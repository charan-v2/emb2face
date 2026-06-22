from __future__ import annotations

import subprocess
import sys
from types import MethodType
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from PIL import Image

from .embeddings import arcface_from_image, load_adaface_model, load_arcface_app, setup_device, source_embeddings
from .models import LinearAdapter, MLPAdapter
from .utils import cosine_similarity_np, l2_normalize


def ensure_arc2face_repo(cfg: dict) -> Path:
    repo_dir = Path(cfg["arc2face_local_dir"]).expanduser()
    if repo_dir.exists():
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/foivospar/Arc2Face.git", str(repo_dir)], check=True)
    return repo_dir


def load_arc2face_pipeline(cfg: dict, device: torch.device):
    repo_dir = ensure_arc2face_repo(cfg)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from arc2face import CLIPTextModelWrapper, project_face_embs

    model_dir = Path(cfg["output_root"]).expanduser() / ".cache" / "arc2face_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "arc2face/config.json",
        "arc2face/diffusion_pytorch_model.safetensors",
        "encoder/config.json",
        "encoder/pytorch_model.bin",
    ]:
        hf_hub_download(repo_id=cfg["arc2face_repo"], filename=filename, local_dir=str(model_dir))

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    encoder = CLIPTextModelWrapper.from_pretrained(model_dir, subfolder="encoder", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder="arc2face", torch_dtype=dtype)

    original_encoder_forward = encoder.text_model.encoder.forward

    def encoder_forward_compat(self, *args, **kwargs):
        kwargs.pop("return_dict", None)
        return original_encoder_forward(*args, **kwargs)

    encoder.text_model.encoder.forward = MethodType(encoder_forward_compat, encoder.text_model.encoder)

    pipeline = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        text_encoder=encoder,
        unet=unet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    return pipeline, project_face_embs


def load_best_adapter(cfg: dict, device: torch.device):
    model_dir = cfg["output_root"] / f"models_{cfg['adapter_run_mode']}"
    candidates = sorted(model_dir.glob("best_*_adapter.pt"))
    if not candidates:
        raise FileNotFoundError(f"No adapter checkpoint found in {model_dir}")
    ckpt = torch.load(candidates[0], map_location=device, weights_only=False)
    acfg = ckpt.get("config", {})
    atype = acfg.get("adapter_type", "linear")
    adapter = (
        LinearAdapter(512)
        if atype == "linear"
        else MLPAdapter(512, acfg.get("hidden_dim", 1024), acfg.get("dropout", 0.1))
    ).to(device)
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()
    return adapter, candidates[0]


def collect_image_rows(
    input_path: Path,
    exts: Iterable[str],
    max_images_per_identity: int | None = None,
):
    if input_path.is_file():
        return [{"identity": input_path.stem, "image_path": input_path}]

    rows = []

    root_images = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if root_images:
        if max_images_per_identity is not None:
            root_images = root_images[:max_images_per_identity]
        for p in root_images:
            rows.append({"identity": input_path.name, "image_path": p})

    for identity_dir in sorted([p for p in input_path.iterdir() if p.is_dir()]):
        paths = sorted([p for p in identity_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
        if max_images_per_identity is not None:
            paths = paths[:max_images_per_identity]
        for p in paths:
            rows.append({"identity": identity_dir.name, "image_path": p})

    return rows


def _load_device(cfg: dict) -> torch.device:
    device = setup_device(cfg)
    return device


def generate_from_embedding(
    pipeline,
    project_face_embs,
    emb512: np.ndarray,
    n: int,
    seed: int,
    device: torch.device,
    cfg: dict,
):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    id_emb = torch.from_numpy(np.asarray(emb512, dtype=np.float32)).to(device, dtype)[None]
    id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)
    proj = project_face_embs(pipeline, id_emb)
    g = torch.Generator(device=device).manual_seed(seed)
    return pipeline(
        prompt_embeds=proj,
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        num_images_per_prompt=n,
        generator=g,
    ).images


def _score_generated_image(source_arc_emb: np.ndarray, image_path: Path, arc_app) -> float | None:
    recon_arc = arcface_from_image(image_path, arc_app)
    if recon_arc is None:
        return None
    return cosine_similarity_np(source_arc_emb, recon_arc)


def _score_generated_images(source_arc_emb: np.ndarray, image_paths: list[Path], arc_app) -> list[float | None]:
    return [_score_generated_image(source_arc_emb, p, arc_app) for p in image_paths]


def _save_single_image_figure(
    input_path: Path,
    source_rgb: np.ndarray,
    adapter_image,
    arcface_image,
    adapter_score: float | None,
    arcface_score: float | None,
    output_path: Path,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(source_rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(adapter_image)
    axes[1].set_title("AdaFace -> Arc2Face")
    axes[1].axis("off")
    axes[1].text(
        0.5,
        -0.08,
        f"Similarity: {adapter_score:.4f}" if adapter_score is not None else "Similarity: N/A",
        ha="center",
        transform=axes[1].transAxes,
        fontsize=11,
    )

    axes[2].imshow(arcface_image)
    axes[2].set_title("ArcFace -> Arc2Face")
    axes[2].axis("off")
    axes[2].text(
        0.5,
        -0.08,
        f"Similarity: {arcface_score:.4f}" if arcface_score is not None else "Similarity: N/A",
        ha="center",
        transform=axes[2].transAxes,
        fontsize=11,
    )

    fig.suptitle(input_path.name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_single_image_inference(
    cfg: dict,
    input_image: Path,
    output_dir: Path | None = None,
    num_images_per_prompt: int | None = None,
    seed: int | None = None,
):
    device = _load_device(cfg)
    adapter, adapter_path = load_best_adapter(cfg, device)
    arc_app, _ = load_arcface_app(cfg)
    ada_model = load_adaface_model(cfg, device)
    pipeline, project_face_embs = load_arc2face_pipeline(cfg, device)

    input_image = Path(input_image)
    if not input_image.exists() or not input_image.is_file():
        raise FileNotFoundError(f"Input image not found: {input_image}")

    output_dir = Path(output_dir) if output_dir is not None else cfg["output_root"] / f"inference_{cfg['runmode'].lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = output_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)

    n_images = num_images_per_prompt if num_images_per_prompt is not None else cfg["num_recon_per_image"]
    base_seed = seed if seed is not None else cfg["seed"]

    emb = source_embeddings(input_image, arc_app, ada_model, device)
    if emb is None:
        raise ValueError(f"Could not extract a face embedding from {input_image}")

    source_arc = emb["arcface"]
    source_rgb = np.array(Image.open(input_image).convert("RGB"))

    ada_input = torch.from_numpy(emb["adaface"]).float().unsqueeze(0).to(device)
    arc_input = torch.from_numpy(source_arc).float().unsqueeze(0).to(device)

    with torch.no_grad():
        adapter_pred = adapter(ada_input).cpu().numpy()[0].astype(np.float32)
    adapter_pred = l2_normalize(adapter_pred).astype(np.float32)

    arcface_pred = arc_input.cpu().numpy()[0].astype(np.float32)
    arcface_pred = l2_normalize(arcface_pred).astype(np.float32)

    stem = input_image.stem
    adapter_dir = recon_dir / f"{stem}_adaface_adapter"
    arc_dir = recon_dir / f"{stem}_arcface_native"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    arc_dir.mkdir(parents=True, exist_ok=True)

    adapter_images = generate_from_embedding(
        pipeline,
        project_face_embs,
        adapter_pred,
        n_images,
        base_seed,
        device,
        cfg,
    )
    arcface_images = generate_from_embedding(
        pipeline,
        project_face_embs,
        arcface_pred,
        n_images,
        base_seed + 1,
        device,
        cfg,
    )

    adapter_saved = []
    for k, image in enumerate(adapter_images):
        out_path = adapter_dir / f"{stem}_adaface_adapter_{k}.png"
        image.save(out_path)
        adapter_saved.append(out_path)

    arc_saved = []
    for k, image in enumerate(arcface_images):
        out_path = arc_dir / f"{stem}_arcface_native_{k}.png"
        image.save(out_path)
        arc_saved.append(out_path)

    adapter_score = _score_generated_image(source_arc, adapter_saved[0], arc_app) if adapter_saved else None
    arcface_score = _score_generated_image(source_arc, arc_saved[0], arc_app) if arc_saved else None

    comparison_path = output_dir / f"{stem}_comparison.png"
    _save_single_image_figure(
        input_path=input_image,
        source_rgb=source_rgb,
        adapter_image=adapter_images[0],
        arcface_image=arcface_images[0],
        adapter_score=adapter_score,
        arcface_score=arcface_score,
        output_path=comparison_path,
    )

    result = pd.DataFrame(
        [
            {
                "image_path": str(input_image),
                "status": "ok",
                "adapter_path": str(adapter_path),
                "comparison_path": str(comparison_path),
                "adapter_recon_path": str(adapter_saved[0]) if adapter_saved else None,
                "arcface_recon_path": str(arc_saved[0]) if arc_saved else None,
                "adapter_similarity": adapter_score,
                "arcface_similarity": arcface_score,
            }
        ]
    )
    result.to_csv(output_dir / "single_image_inference.csv", index=False)
    return {
        "output_dir": output_dir,
        "comparison_path": comparison_path,
        "adapter_path": adapter_path,
        "results": result,
    }


def run_inference_pipeline(
    cfg: dict,
    input_dir: Path,
    output_dir: Path | None = None,
    num_images_per_prompt: int | None = None,
    seed: int | None = None,
    max_images_per_identity: int | None = None,
):
    device = _load_device(cfg)
    adapter, adapter_path = load_best_adapter(cfg, device)
    arc_app, _ = load_arcface_app(cfg)
    ada_model = load_adaface_model(cfg, device)
    pipeline, project_face_embs = load_arc2face_pipeline(cfg, device)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else cfg["output_root"] / f"inference_{cfg['runmode'].lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = output_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)

    n_images = num_images_per_prompt if num_images_per_prompt is not None else cfg["num_recon_per_image"]
    base_seed = seed if seed is not None else cfg["seed"]

    rows = collect_image_rows(input_dir, cfg["image_extensions"], max_images_per_identity=max_images_per_identity)
    if not rows:
        raise ValueError(f"No images found under {input_dir}")

    result_rows = []
    for idx, row in enumerate(tqdm(rows, desc="Inference")):
        image_path = Path(row["image_path"])
        emb = source_embeddings(image_path, arc_app, ada_model, device)
        if emb is None:
            result_rows.append(
                {
                    "identity": row.get("identity"),
                    "image_path": str(image_path),
                    "status": "failed",
                    "reason": "face_or_embedding_extraction_failed",
                }
            )
            continue

        source_arc = emb["arcface"]
        ada = torch.from_numpy(emb["adaface"]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            arc_pred = adapter(ada).cpu().numpy()[0].astype(np.float32)
        arc_pred = l2_normalize(arc_pred).astype(np.float32)

        stem = image_path.stem
        out_subdir = recon_dir / stem
        out_subdir.mkdir(parents=True, exist_ok=True)
        images = generate_from_embedding(
            pipeline,
            project_face_embs,
            arc_pred,
            n_images,
            base_seed + idx,
            device,
            cfg,
        )

        saved_paths = []
        for k, image in enumerate(images):
            out_path = out_subdir / f"{stem}_arc2face_{k}.png"
            image.save(out_path)
            saved_paths.append(str(out_path))

        scores = _score_generated_images(source_arc, [Path(p) for p in saved_paths], arc_app)

        result_rows.append(
            {
                "identity": row.get("identity"),
                "image_path": str(image_path),
                "status": "ok",
                "arcface_source_path": str(image_path),
                "output_paths": ";".join(saved_paths),
                "output_scores": ";".join("" if s is None else f"{s:.6f}" for s in scores),
                "first_output_score": scores[0] if scores else None,
                "arcface_pred_norm": float(np.linalg.norm(arc_pred)),
                "adapter_path": str(adapter_path),
            }
        )

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(output_dir / "inference_index.csv", index=False)
    return {
        "output_dir": output_dir,
        "recon_dir": recon_dir,
        "adapter_path": adapter_path,
        "results": result_df,
    }
