from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime
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
from .models import LinearAdapter, MLPAdapter, ResidualMLPAdapter
from .utils import cosine_similarity_np, l2_normalize


LOGGER = logging.getLogger(__name__)


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
    explicit_ckpt = cfg.get("inference_adapter_checkpoint")
    if explicit_ckpt is not None:
        ckpt_path = Path(explicit_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Explicit adapter checkpoint not found: {ckpt_path}")
    else:
        model_dir = cfg["output_root"] / f"models_{cfg['adapter_run_mode']}"
        candidates = sorted(model_dir.glob("best_*_adapter.pt"))
        if not candidates:
            raise FileNotFoundError(f"No adapter checkpoint found in {model_dir}")
        ckpt_path = candidates[0]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    acfg = ckpt.get("config", {})
    atype = acfg.get("adapter_type", "linear")
    hidden_dim = acfg.get("hidden_dim", 1024)
    dropout = acfg.get("dropout", 0.1)
    if atype == "linear":
        adapter = LinearAdapter(512)
    elif atype == "mlp":
        adapter = MLPAdapter(512, hidden_dim, dropout)
    elif atype == "residual_mlp":
        adapter = ResidualMLPAdapter(512, hidden_dim, dropout)
    else:
        raise ValueError(f"Unsupported adapter_type in checkpoint: {atype}")
    adapter = adapter.to(device)
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()
    return adapter, ckpt_path


def collect_identity_images(input_dir: Path, exts: Iterable[str]) -> dict[str, list[Path]]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    identity_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not identity_dirs:
        raise ValueError(f"No identity folders found under {input_dir}")

    mapping: dict[str, list[Path]] = {}
    for identity_dir in identity_dirs:
        paths = sorted([p for p in identity_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
        if paths:
            mapping[identity_dir.name] = paths

    if not mapping:
        raise ValueError(f"No images found under {input_dir}")

    return mapping


def sample_image_rows(
    input_dir: Path,
    exts: Iterable[str],
    num_identities: int | None,
    images_per_identity: int,
) -> list[dict]:
    identity_images = collect_identity_images(input_dir, exts)
    identities = sorted(identity_images.keys())
    # Sampling is intentionally non-deterministic so the selected identities/images
    # do not depend on the inference seed.
    rng = np.random.default_rng()

    if num_identities is None or num_identities >= len(identities):
        selected_identities = list(rng.permutation(identities))
    else:
        selected_identities = list(rng.choice(identities, size=num_identities, replace=False))

    rows = []
    for identity_order, identity in enumerate(selected_identities):
        paths = identity_images[identity]
        take = min(images_per_identity, len(paths))
        if take <= 0:
            continue
        if take == len(paths):
            selected_paths = list(paths)
        else:
            selected_paths = [paths[i] for i in rng.choice(len(paths), size=take, replace=False)]
        for image_order, image_path in enumerate(selected_paths):
            rows.append(
                {
                    "identity": identity,
                    "identity_order": identity_order,
                    "image_order": image_order,
                    "image_path": image_path,
                }
            )

    return rows


def _load_device(cfg: dict) -> torch.device:
    device = setup_device(cfg)
    return device


def _random_seed() -> int:
    return random.SystemRandom().randint(0, 2**32 - 1)


def _make_run_dir(base_dir: Path) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def generate_from_embedding(
    pipeline,
    project_face_embs,
    emb512: np.ndarray,
    n: int,
    seed: int | None,
    device: torch.device,
    cfg: dict,
):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    id_emb = torch.from_numpy(np.asarray(emb512, dtype=np.float32)).to(device, dtype)[None]
    id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)
    proj = project_face_embs(pipeline, id_emb)
    if seed is None:
        seed = _random_seed()
    g = torch.Generator(device=device).manual_seed(seed)
    return pipeline(
        prompt_embeds=proj,
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        num_images_per_prompt=n,
        generator=g,
    ).images


def _score_generated_image(source_arc_emb: np.ndarray, image_path: Path, arc_app) -> tuple[float | None, str]:
    import cv2

    if not image_path.exists():
        return None, "generated_image_missing"

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None, "generated_image_unreadable"

    recon_arc, _ = None, None
    try:
        recon_arc = arcface_from_image(image_path, arc_app)
    except Exception as exc:
        LOGGER.warning("ArcFace scoring failed for %s: %s", image_path, exc)
        return None, f"arcface_scoring_error:{exc.__class__.__name__}"

    if recon_arc is None:
        return None, "face_not_detected_by_arcface"

    return cosine_similarity_np(source_arc_emb, recon_arc), "ok"


def _score_generated_images(source_arc_emb: np.ndarray, image_paths: list[Path], arc_app) -> list[tuple[float | None, str]]:
    return [_score_generated_image(source_arc_emb, p, arc_app) for p in image_paths]


def _score_text(score: float | None, reason: str | None) -> str:
    if score is not None:
        return f"Similarity: {score:.4f}"
    if reason:
        return f"Similarity: N/A ({reason})"
    return "Similarity: N/A"


def _generate_reconstruction_with_retries(
    *,
    cfg: dict,
    pipeline,
    project_face_embs,
    embedding: np.ndarray,
    label: str,
    stem: str,
    sample_dir: Path,
    source_arc: np.ndarray,
    arc_app,
    device: torch.device,
    base_seed: int,
    max_retries: int,
) -> tuple[list[object], list[Path], float | None, str, int]:
    attempts_dir = sample_dir / f"{label}_attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    final_images = None
    final_paths = None
    final_score = None
    final_reason = "no_generated_image"
    attempts = 0

    for attempt in range(max_retries + 1):
        seed = base_seed + attempt
        images = generate_from_embedding(
            pipeline,
            project_face_embs,
            embedding,
            int(cfg["num_recon_per_image"]),
            seed,
            device,
            cfg,
        )
        attempt_dir = attempts_dir / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_paths = []
        for idx, image in enumerate(images):
            attempt_path = attempt_dir / f"{stem}_{label}_{idx}.png"
            image.save(attempt_path)
            attempt_paths.append(attempt_path)
        attempts += 1

        score, reason = _score_generated_image(source_arc, attempt_paths[0], arc_app)
        LOGGER.info(
            "[inference-retry] label=%s attempt=%d/%d seed=%d score=%s reason=%s",
            label,
            attempt + 1,
            max_retries + 1,
            seed,
            "ok" if score is not None else "N/A",
            reason,
        )

        final_images = images
        final_paths = [sample_dir / f"{stem}_{label}_{idx}.png" for idx in range(len(images))]
        final_score = score
        final_reason = reason
        if score is not None:
            for image, path in zip(images, final_paths):
                image.save(path)
            return final_images, final_paths, final_score, final_reason, attempts

    if final_images is not None and final_paths is not None:
        for image, path in zip(final_images, final_paths):
            image.save(path)
    return final_images or [], final_paths or [], final_score, final_reason, attempts


def _process_sample(
    *,
    cfg: dict,
    sample: dict,
    adapter,
    adapter_path: Path,
    arc_app,
    ada_model,
    pipeline,
    project_face_embs,
    device: torch.device,
    recon_dir: Path,
    figures_dir: Path | None,
    num_images_per_prompt: int,
    base_seed: int,
    sample_index: int,
    save_comparison_figures: bool,
):
    image_path = Path(sample["image_path"])
    emb = source_embeddings(image_path, arc_app, ada_model, device)
    if emb is None:
        return {
            "identity": sample.get("identity"),
            "identity_order": sample.get("identity_order"),
            "image_order": sample.get("image_order"),
            "image_path": str(image_path),
            "status": "failed",
            "reason": "face_or_embedding_extraction_failed",
        }

    source_arc = emb["arcface"]
    source_ada = emb["adaface"]
    source_rgb = np.array(Image.open(image_path).convert("RGB"))

    ada_input = torch.from_numpy(emb["adaface"]).float().unsqueeze(0).to(device)
    arc_input = torch.from_numpy(source_arc).float().unsqueeze(0).to(device)

    with torch.no_grad():
        adapter_pred = adapter(ada_input).cpu().numpy()[0].astype(np.float32)
    adapter_pred = l2_normalize(adapter_pred).astype(np.float32)

    arcface_pred = arc_input.cpu().numpy()[0].astype(np.float32)
    arcface_pred = l2_normalize(arcface_pred).astype(np.float32)

    adaface_pred = l2_normalize(source_ada.astype(np.float32)).astype(np.float32)

    stem = image_path.stem
    identity = sample.get("identity", image_path.parent.name)
    sample_recon_dir = recon_dir / identity / stem
    sample_recon_dir.mkdir(parents=True, exist_ok=True)

    max_retries = int(cfg.get("inference_score_max_retries", 0))
    adaface_images, adaface_paths, adaface_score, adaface_reason, adaface_attempts = _generate_reconstruction_with_retries(
        cfg=cfg,
        pipeline=pipeline,
        project_face_embs=project_face_embs,
        embedding=adaface_pred,
        label="adaface_native",
        stem=stem,
        sample_dir=sample_recon_dir,
        source_arc=source_arc,
        arc_app=arc_app,
        device=device,
        base_seed=base_seed + sample_index * 10 + 1,
        max_retries=max_retries,
    )
    adapter_images, adapter_paths_out, adapter_score, adapter_reason, adapter_attempts = _generate_reconstruction_with_retries(
        cfg=cfg,
        pipeline=pipeline,
        project_face_embs=project_face_embs,
        embedding=adapter_pred,
        label="adaface_adapter",
        stem=stem,
        sample_dir=sample_recon_dir,
        source_arc=source_arc,
        arc_app=arc_app,
        device=device,
        base_seed=base_seed + sample_index * 10,
        max_retries=max_retries,
    )
    arcface_images, arc_paths, arcface_score, arcface_reason, arcface_attempts = _generate_reconstruction_with_retries(
        cfg=cfg,
        pipeline=pipeline,
        project_face_embs=project_face_embs,
        embedding=arcface_pred,
        label="arcface_native",
        stem=stem,
        sample_dir=sample_recon_dir,
        source_arc=source_arc,
        arc_app=arc_app,
        device=device,
        base_seed=base_seed + sample_index * 10 + 2,
        max_retries=max_retries,
    )

    for label, score, reason, path in [
        ("adaface_native", adaface_score, adaface_reason, adaface_paths[0] if adaface_paths else None),
        ("adapter", adapter_score, adapter_reason, adapter_paths_out[0] if adapter_paths_out else None),
        ("arcface", arcface_score, arcface_reason, arc_paths[0] if arc_paths else None),
    ]:
        if score is None:
            LOGGER.warning("Similarity unavailable for %s on %s: %s", label, path, reason)

    comparison_path = None
    if save_comparison_figures and figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = figures_dir / identity / f"{stem}_comparison.png"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        _save_single_image_figure(
            input_path=image_path,
            source_rgb=source_rgb,
            adaface_image=adaface_images[0] if adaface_images else source_rgb,
            adapter_image=adapter_images[0] if adapter_images else source_rgb,
            arcface_image=arcface_images[0] if arcface_images else source_rgb,
            adaface_score=adaface_score,
            adapter_score=adapter_score,
            arcface_score=arcface_score,
            adaface_reason=adaface_reason,
            adapter_reason=adapter_reason,
            arcface_reason=arcface_reason,
            output_path=comparison_path,
        )

    return {
        "identity": identity,
        "identity_order": sample.get("identity_order"),
        "image_order": sample.get("image_order"),
        "image_path": str(image_path),
        "status": "ok",
        "arcface_source_path": str(image_path),
        "adaface_native_recon_path": str(adaface_paths[0]) if adaface_paths else None,
        "adapter_recon_path": str(adapter_paths_out[0]) if adapter_paths_out else None,
        "arcface_recon_path": str(arc_paths[0]) if arc_paths else None,
        "adaface_native_similarity_reason": adaface_reason,
        "adapter_similarity_reason": adapter_reason,
        "arcface_similarity_reason": arcface_reason,
        "adaface_native_similarity": adaface_score,
        "adapter_similarity": adapter_score,
        "arcface_similarity": arcface_score,
        "adaface_native_generation_attempts": adaface_attempts,
        "adapter_generation_attempts": adapter_attempts,
        "arcface_generation_attempts": arcface_attempts,
        "comparison_path": str(comparison_path) if comparison_path else None,
        "adapter_path": str(adapter_path),
    }


def build_summary(result_df: pd.DataFrame, requested_identities: int | None, images_per_identity: int, save_comparison_figures: bool):
    ok = result_df[result_df["status"] == "ok"].copy()
    summary = {
        "num_rows": int(len(result_df)),
        "num_ok": int(len(ok)),
        "num_failed": int((result_df["status"] != "ok").sum()),
        "requested_identities": requested_identities,
        "images_per_identity": images_per_identity,
        "save_comparison_figures": save_comparison_figures,
        "mean_adaface_native_similarity": float(ok["adaface_native_similarity"].mean()) if len(ok) else float("nan"),
        "mean_adapter_similarity": float(ok["adapter_similarity"].mean()) if len(ok) else float("nan"),
        "mean_arcface_similarity": float(ok["arcface_similarity"].mean()) if len(ok) else float("nan"),
    }
    return pd.DataFrame([summary])


def _save_single_image_figure(
    input_path: Path,
    source_rgb: np.ndarray,
    adaface_image,
    adapter_image,
    arcface_image,
    adaface_score: float | None,
    adapter_score: float | None,
    arcface_score: float | None,
    output_path: Path,
    adaface_reason: str | None = None,
    adapter_reason: str | None = None,
    arcface_reason: str | None = None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(source_rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(adaface_image)
    axes[1].set_title("AdaFace -> Arc2Face")
    axes[1].axis("off")
    axes[1].text(
        0.5,
        -0.08,
        _score_text(adaface_score, adaface_reason),
        ha="center",
        transform=axes[1].transAxes,
        fontsize=11,
    )

    axes[2].imshow(adapter_image)
    axes[2].set_title("AdaFace -> ArcFace -> Arc2Face")
    axes[2].axis("off")
    axes[2].text(
        0.5,
        -0.08,
        _score_text(adapter_score, adapter_reason),
        ha="center",
        transform=axes[2].transAxes,
        fontsize=11,
    )

    axes[3].imshow(arcface_image)
    axes[3].set_title("ArcFace -> Arc2Face")
    axes[3].axis("off")
    axes[3].text(
        0.5,
        -0.08,
        _score_text(arcface_score, arcface_reason),
        ha="center",
        transform=axes[3].transAxes,
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

    base_output_dir = Path(output_dir) if output_dir is not None else cfg["output_root"] / f"inference_{cfg['runmode'].lower()}"
    output_dir, run_id = _make_run_dir(base_output_dir)
    recon_dir = output_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)

    n_images = num_images_per_prompt if num_images_per_prompt is not None else cfg["num_recon_per_image"]
    base_seed = seed if seed is not None else cfg.get("seed")
    if base_seed is None:
        base_seed = _random_seed()
    max_retries = int(cfg.get("inference_score_max_retries", 0))

    emb = source_embeddings(input_image, arc_app, ada_model, device)
    if emb is None:
        raise ValueError(f"Could not extract a face embedding from {input_image}")

    source_arc = emb["arcface"]
    source_ada = emb["adaface"]
    source_rgb = np.array(Image.open(input_image).convert("RGB"))

    ada_input = torch.from_numpy(emb["adaface"]).float().unsqueeze(0).to(device)
    arc_input = torch.from_numpy(source_arc).float().unsqueeze(0).to(device)

    with torch.no_grad():
        adapter_pred = adapter(ada_input).cpu().numpy()[0].astype(np.float32)
    adapter_pred = l2_normalize(adapter_pred).astype(np.float32)

    arcface_pred = arc_input.cpu().numpy()[0].astype(np.float32)
    arcface_pred = l2_normalize(arcface_pred).astype(np.float32)

    adaface_pred = source_ada.astype(np.float32)
    adaface_pred = l2_normalize(adaface_pred).astype(np.float32)

    stem = input_image.stem
    adapter_dir = recon_dir / f"{stem}_adaface_adapter"
    adaface_dir = recon_dir / f"{stem}_adaface_native"
    arc_dir = recon_dir / f"{stem}_arcface_native"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adaface_dir.mkdir(parents=True, exist_ok=True)
    arc_dir.mkdir(parents=True, exist_ok=True)

    adapter_images, adapter_paths_out, adapter_score, adapter_reason, adapter_attempts = _generate_reconstruction_with_retries(
        cfg=cfg,
        pipeline=pipeline,
        project_face_embs=project_face_embs,
        embedding=adapter_pred,
        label="adaface_adapter",
        stem=stem,
        sample_dir=adapter_dir,
        source_arc=source_arc,
        arc_app=arc_app,
        device=device,
        base_seed=base_seed,
        max_retries=max_retries,
    )
    adaface_images, adaface_paths, adaface_score, adaface_reason, adaface_attempts = _generate_reconstruction_with_retries(
        cfg=cfg,
        pipeline=pipeline,
        project_face_embs=project_face_embs,
        embedding=adaface_pred,
        label="adaface_native",
        stem=stem,
        sample_dir=adaface_dir,
        source_arc=source_arc,
        arc_app=arc_app,
        device=device,
        base_seed=base_seed + 1,
        max_retries=max_retries,
    )
    arcface_images, arc_paths, arcface_score, arcface_reason, arcface_attempts = _generate_reconstruction_with_retries(
        cfg=cfg,
        pipeline=pipeline,
        project_face_embs=project_face_embs,
        embedding=arcface_pred,
        label="arcface_native",
        stem=stem,
        sample_dir=arc_dir,
        source_arc=source_arc,
        arc_app=arc_app,
        device=device,
        base_seed=base_seed + 2,
        max_retries=max_retries,
    )

    comparison_path = output_dir / f"{stem}_comparison.png"
    _save_single_image_figure(
        input_path=input_image,
        source_rgb=source_rgb,
        adaface_image=adaface_images[0] if adaface_images else source_rgb,
        adapter_image=adapter_images[0] if adapter_images else source_rgb,
        arcface_image=arcface_images[0] if arcface_images else source_rgb,
        adaface_score=adaface_score,
        adapter_score=adapter_score,
        arcface_score=arcface_score,
        adaface_reason=adaface_reason,
        adapter_reason=adapter_reason,
        arcface_reason=arcface_reason,
        output_path=comparison_path,
    )

    result = pd.DataFrame(
        [
            {
            "image_path": str(input_image),
                "status": "ok",
                "run_id": run_id,
                "adapter_path": str(adapter_path),
                "comparison_path": str(comparison_path),
                "adaface_native_recon_path": str(adaface_paths[0]) if adaface_paths else None,
                "adapter_recon_path": str(adapter_paths_out[0]) if adapter_paths_out else None,
                "arcface_recon_path": str(arc_paths[0]) if arc_paths else None,
                "adaface_native_similarity": adaface_score,
                "adapter_similarity": adapter_score,
                "arcface_similarity": arcface_score,
                "adaface_native_generation_attempts": adaface_attempts,
                "adapter_generation_attempts": adapter_attempts,
                "arcface_generation_attempts": arcface_attempts,
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
    num_identities: int | None = None,
    images_per_identity: int | None = None,
    num_images_per_prompt: int | None = None,
    save_comparison_figures: bool | None = None,
    seed: int | None = None,
):
    device = _load_device(cfg)
    adapter, adapter_path = load_best_adapter(cfg, device)
    arc_app, _ = load_arcface_app(cfg)
    ada_model = load_adaface_model(cfg, device)
    pipeline, project_face_embs = load_arc2face_pipeline(cfg, device)

    input_dir = Path(input_dir)
    base_output_dir = Path(output_dir) if output_dir is not None else cfg["output_root"] / f"inference_{cfg['runmode'].lower()}"
    output_dir, run_id = _make_run_dir(base_output_dir)
    recon_dir = output_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)

    n_images = num_images_per_prompt if num_images_per_prompt is not None else cfg["num_recon_per_image"]
    selected_num_identities = num_identities if num_identities is not None else cfg["inference_num_identities"]
    selected_images_per_identity = images_per_identity if images_per_identity is not None else cfg["inference_images_per_identity"]
    selected_save_figures = save_comparison_figures if save_comparison_figures is not None else cfg["save_comparison_figures"]
    base_seed = seed if seed is not None else cfg.get("seed")
    if base_seed is None:
        base_seed = _random_seed()

    rows = sample_image_rows(
        input_dir,
        cfg["image_extensions"],
        selected_num_identities,
        selected_images_per_identity,
    )
    if not rows:
        raise ValueError(f"No images found under {input_dir}")

    selected_df = pd.DataFrame(rows)
    selected_df["run_id"] = run_id
    selected_df.to_csv(output_dir / "selected_samples.csv", index=False)

    figures_dir = output_dir / "figures" if selected_save_figures else None
    result_rows = []
    for idx, row in enumerate(tqdm(rows, desc="Inference")):
        row_result = _process_sample(
            cfg=cfg,
            sample=row,
            adapter=adapter,
            adapter_path=adapter_path,
            arc_app=arc_app,
            ada_model=ada_model,
            pipeline=pipeline,
            project_face_embs=project_face_embs,
            device=device,
            recon_dir=recon_dir,
            figures_dir=figures_dir,
            num_images_per_prompt=n_images,
            base_seed=base_seed,
            sample_index=idx,
            save_comparison_figures=selected_save_figures,
        )
        result_rows.append(row_result)

    result_df = pd.DataFrame(result_rows)
    result_df["run_id"] = run_id
    result_df.to_csv(output_dir / "inference_report.csv", index=False)
    summary_df = build_summary(result_df, selected_num_identities, selected_images_per_identity, selected_save_figures)
    summary_df["run_id"] = run_id
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    return {
        "output_dir": output_dir,
        "run_id": run_id,
        "recon_dir": recon_dir,
        "adapter_path": adapter_path,
        "results": result_df,
        "summary": summary_df,
    }
