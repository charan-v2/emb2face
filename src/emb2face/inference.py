from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime
import subprocess
import sys
from types import MethodType
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from PIL import Image

from .embeddings import load_adaface_model, load_arcface_app, setup_device, source_embeddings
from .face_backends import estimate_face_yaw_degrees, select_largest_face
from .models import LinearAdapter, MLPAdapter, ResidualMLPAdapter
from .utils import l2_normalize


LOGGER = logging.getLogger(__name__)


@dataclass
class LoadedAdapter:
    name: str
    path: Path
    model: nn.Module


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


def _adapter_display_name(ckpt_path: Path, ckpt: dict | None = None) -> str:
    if ckpt is not None:
        acfg = ckpt.get("config", {})
        atype = acfg.get("adapter_type")
        if atype:
            return str(atype)

    stem = ckpt_path.stem
    if stem.startswith("best_") and stem.endswith("_adapter"):
        stem = stem[len("best_") : -len("_adapter")]
    elif stem.startswith("best_"):
        stem = stem[len("best_") :]
    elif stem.endswith("_adapter"):
        stem = stem[: -len("_adapter")]
    return stem or "adapter"


def _build_adapter_from_checkpoint(ckpt_path: Path, device: torch.device) -> LoadedAdapter:
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
    return LoadedAdapter(name=_adapter_display_name(ckpt_path, ckpt), path=ckpt_path, model=adapter)


def load_adapters(cfg: dict, device: torch.device) -> list[LoadedAdapter]:
    explicit_multi = cfg.get("inference_adapter_checkpoints")
    explicit_single = cfg.get("inference_adapter_checkpoint")

    if explicit_multi is not None:
        ckpt_paths = [Path(p) for p in explicit_multi]
    elif explicit_single is not None:
        ckpt_paths = [Path(explicit_single)]
    else:
        model_dir = cfg["output_root"] / f"models_{cfg['adapter_run_mode']}"
        candidates = sorted(model_dir.glob("best_*_adapter.pt"))
        if not candidates:
            raise FileNotFoundError(f"No adapter checkpoint found in {model_dir}")
        ckpt_paths = [candidates[0]]

    adapters = [_build_adapter_from_checkpoint(path, device) for path in ckpt_paths]
    seen: dict[str, int] = {}
    for item in adapters:
        base_name = item.name
        count = seen.get(base_name, 0)
        if count:
            item.name = f"{base_name}_{count + 1}"
        seen[base_name] = count + 1
    return adapters


def load_best_adapter(cfg: dict, device: torch.device):
    adapter = load_adapters(cfg, device)[0]
    return adapter.model, adapter.path


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


def _image_is_frontal(image_path: Path, arc_app, max_yaw_degrees: float, require_single_face: bool) -> dict[str, object] | None:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None

    try:
        faces = list(arc_app.get(img_bgr))
    except Exception:
        return None

    if not faces:
        return None
    if require_single_face and len(faces) != 1:
        return None

    face = select_largest_face(faces)
    yaw = estimate_face_yaw_degrees(face, img_bgr.shape)
    if yaw is None:
        return None
    if abs(yaw) > float(max_yaw_degrees):
        return None

    return {
        "image_path": str(image_path),
        "face_count": int(len(faces)),
        "yaw_degrees": float(yaw),
        "abs_yaw_degrees": float(abs(yaw)),
    }


def build_pose_filtered_manifest(
    input_dir: Path,
    exts: Iterable[str],
    arc_app,
    max_yaw_degrees: float,
    require_single_face: bool,
) -> tuple[dict[str, list[Path]], pd.DataFrame]:
    identity_images = collect_identity_images(input_dir, exts)
    filtered: dict[str, list[Path]] = {}
    rows: list[dict[str, object]] = []

    for identity, paths in tqdm(identity_images.items(), desc="Pose filtering"):
        kept_paths: list[Path] = []
        for image_path in paths:
            pose_row = _image_is_frontal(
                image_path,
                arc_app=arc_app,
                max_yaw_degrees=max_yaw_degrees,
                require_single_face=require_single_face,
            )
            if pose_row is None:
                rows.append(
                    {
                        "identity": identity,
                        "image_path": str(image_path),
                        "keep": False,
                        "face_count": None,
                        "yaw_degrees": None,
                        "abs_yaw_degrees": None,
                        "reason": "no_usable_face_or_yaw_exceeds_threshold",
                    }
                )
                continue

            rows.append(
                {
                    "identity": identity,
                    **pose_row,
                    "keep": True,
                    "reason": "kept",
                }
            )
            kept_paths.append(image_path)

        if kept_paths:
            filtered[identity] = kept_paths

    manifest_df = pd.DataFrame(rows)
    if not manifest_df.empty:
        manifest_df["keep"] = manifest_df["keep"].astype(bool)
    return filtered, manifest_df


def sample_image_rows_from_mapping(
    identity_images: dict[str, list[Path]],
    num_identities: int | None,
    images_per_identity: int,
    *,
    arc_app,
    max_yaw_degrees: float,
    require_single_face: bool,
    seed: int | None,
) -> tuple[list[dict], pd.DataFrame]:
    if not identity_images:
        raise ValueError("No identity folders found after scanning the dataset")

    target_identities = len(identity_images) if num_identities is None else min(num_identities, len(identity_images))
    rng = np.random.default_rng(seed)
    identity_order_pool = list(rng.permutation(sorted(identity_images.keys())))

    selected_rows: list[dict] = []
    manifest_rows: list[dict[str, object]] = []
    selected_identity_count = 0

    for identity in identity_order_pool:
        if selected_identity_count >= target_identities:
            break

        candidate_paths = list(identity_images[identity])
        if not candidate_paths:
            continue
        candidate_paths = list(rng.permutation(candidate_paths))

        accepted_paths: list[Path] = []
        for candidate_index, image_path in enumerate(candidate_paths):
            pose_row = _image_is_frontal(
                image_path,
                arc_app=arc_app,
                max_yaw_degrees=max_yaw_degrees,
                require_single_face=require_single_face,
            )
            if pose_row is None:
                manifest_rows.append(
                    {
                        "identity": identity,
                        "image_path": str(image_path),
                        "candidate_index": candidate_index,
                        "keep": False,
                        "face_count": None,
                        "yaw_degrees": None,
                        "abs_yaw_degrees": None,
                        "reason": "rejected_by_pose_filter",
                    }
                )
                continue

            manifest_rows.append(
                {
                    "identity": identity,
                    "image_path": str(image_path),
                    "candidate_index": candidate_index,
                    "keep": True,
                    **pose_row,
                    "reason": "kept",
                }
            )
            accepted_paths.append(image_path)
            if len(accepted_paths) >= images_per_identity:
                break

        if len(accepted_paths) < images_per_identity:
            continue

        for image_order, image_path in enumerate(accepted_paths[:images_per_identity]):
            selected_rows.append(
                {
                    "identity": identity,
                    "identity_order": selected_identity_count,
                    "image_order": image_order,
                    "image_path": image_path,
                }
            )
        selected_identity_count += 1

    if selected_identity_count < target_identities:
        raise ValueError(
            f"Could only find {selected_identity_count} identities with at least {images_per_identity} usable images "
            f"out of {target_identities} requested. Try relaxing the pose threshold or using a larger dataset."
        )

    manifest_df = pd.DataFrame(manifest_rows)
    if not manifest_df.empty:
        manifest_df["keep"] = manifest_df["keep"].astype(bool)
    return selected_rows, manifest_df


def sample_image_rows(
    input_dir: Path,
    exts: Iterable[str],
    num_identities: int | None,
    images_per_identity: int,
    *,
    arc_app,
    max_yaw_degrees: float,
    require_single_face: bool,
    seed: int | None,
) -> tuple[list[dict], pd.DataFrame]:
    identity_images = collect_identity_images(input_dir, exts)
    return sample_image_rows_from_mapping(
        identity_images,
        num_identities,
        images_per_identity,
        arc_app=arc_app,
        max_yaw_degrees=max_yaw_degrees,
        require_single_face=require_single_face,
        seed=seed,
    )


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


def _generate_reconstruction(
    *,
    cfg: dict,
    pipeline,
    project_face_embs,
    embedding: np.ndarray,
    label: str,
    stem: str,
    sample_dir: Path,
    device: torch.device,
    base_seed: int,
    num_images_per_prompt: int,
) -> tuple[list[object], list[Path]]:
    images = generate_from_embedding(
        pipeline,
        project_face_embs,
        embedding,
        int(num_images_per_prompt),
        base_seed,
        device,
        cfg,
    )
    paths: list[Path] = []
    for idx, image in enumerate(images):
        path = sample_dir / f"{stem}_{label}_{idx}.png"
        image.save(path)
        paths.append(path)
    return images, paths


def _pretty_method_name(method: str) -> str:
    mapping = {
        "adapter": "Adapter",
        "linear": "Linear Adapter",
        "mlp": "MLP Adapter",
        "residual_mlp": "Residual MLP Adapter",
        "adaface_native": "AdaFace Native",
        "arcface_native": "ArcFace Native",
    }
    if method in mapping:
        return mapping[method]
    return method.replace("_", " ").title()


def _process_sample(
    *,
    cfg: dict,
    sample: dict,
    adapters: Sequence[LoadedAdapter],
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

    stem = image_path.stem
    identity = sample.get("identity", image_path.parent.name)
    sample_recon_dir = recon_dir / identity / stem
    sample_recon_dir.mkdir(parents=True, exist_ok=True)

    adapter_images_by_name: dict[str, object] = {}
    adapter_paths_by_name: dict[str, list[str]] = {}
    adapter_paths_meta: dict[str, str] = {}
    for adapter_index, adapter_item in enumerate(adapters):
        with torch.no_grad():
            adapter_pred = adapter_item.model(ada_input).cpu().numpy()[0].astype(np.float32)
        adapter_pred = l2_normalize(adapter_pred).astype(np.float32)

        adapter_images, adapter_paths = _generate_reconstruction(
            cfg=cfg,
            pipeline=pipeline,
            project_face_embs=project_face_embs,
            embedding=adapter_pred,
            label=f"{adapter_item.name}_adapter",
            stem=stem,
            sample_dir=sample_recon_dir,
            device=device,
            base_seed=base_seed + sample_index * 1000 + 100 + adapter_index,
            num_images_per_prompt=num_images_per_prompt,
        )
        adapter_images_by_name[adapter_item.name] = adapter_images
        adapter_paths_by_name[adapter_item.name] = [str(p) for p in adapter_paths]
        adapter_paths_meta[adapter_item.name] = str(adapter_item.path)

    arcface_pred = arc_input.cpu().numpy()[0].astype(np.float32)
    arcface_pred = l2_normalize(arcface_pred).astype(np.float32)

    adaface_pred = l2_normalize(source_ada.astype(np.float32)).astype(np.float32)

    adaface_images, adaface_paths = _generate_reconstruction(
        cfg=cfg,
        pipeline=pipeline,
        project_face_embs=project_face_embs,
        embedding=adaface_pred,
        label="adaface_native",
        stem=stem,
        sample_dir=sample_recon_dir,
        device=device,
        base_seed=base_seed + sample_index * 1000 + 1,
        num_images_per_prompt=num_images_per_prompt,
    )
    arcface_images, arc_paths = _generate_reconstruction(
        cfg=cfg,
        pipeline=pipeline,
        project_face_embs=project_face_embs,
        embedding=arcface_pred,
        label="arcface_native",
        stem=stem,
        sample_dir=sample_recon_dir,
        device=device,
        base_seed=base_seed + sample_index * 1000 + 2,
        num_images_per_prompt=num_images_per_prompt,
    )

    comparison_path = None
    if save_comparison_figures and figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = figures_dir / identity / f"{stem}_comparison.png"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        _save_comparison_figure(
            input_path=image_path,
            source_rgb=source_rgb,
            adaface_image=adaface_images[0] if adaface_images else source_rgb,
            adapter_images={name: images[0] if images else source_rgb for name, images in adapter_images_by_name.items()},
            arcface_image=arcface_images[0] if arcface_images else source_rgb,
            output_path=comparison_path,
        )

    result = {
        "identity": identity,
        "identity_order": sample.get("identity_order"),
        "image_order": sample.get("image_order"),
        "image_path": str(image_path),
        "status": "ok",
        "arcface_source_path": str(image_path),
        "adaface_native_recon_path": str(adaface_paths[0]) if adaface_paths else None,
        "adaface_native_recon_paths": json.dumps([str(p) for p in adaface_paths]),
        "arcface_recon_path": str(arc_paths[0]) if arc_paths else None,
        "arcface_recon_paths": json.dumps([str(p) for p in arc_paths]),
        "num_generated_images": int(len(adaface_images)),
        "comparison_path": str(comparison_path) if comparison_path else None,
        "adapter_path": json.dumps(adapter_paths_meta) if len(adapter_paths_meta) > 1 else next(iter(adapter_paths_meta.values()), None),
        "adapter_paths": json.dumps(adapter_paths_meta),
    }
    for adapter_name, adapter_paths in adapter_paths_by_name.items():
        result[f"{adapter_name}_recon_path"] = adapter_paths[0] if adapter_paths else None
        result[f"{adapter_name}_recon_paths"] = json.dumps(adapter_paths)
        result[f"{adapter_name}_adapter_path"] = adapter_paths_meta.get(adapter_name)

    return result


def build_summary(
    result_df: pd.DataFrame,
    requested_identities: int | None,
    images_per_identity: int,
    save_comparison_figures: bool,
    *,
    selected_identity_count: int,
    pose_manifest_rows: int,
    pose_kept_rows: int,
    max_yaw_degrees: float,
    require_single_face: bool,
    adapter_count: int,
):
    ok = result_df[result_df["status"] == "ok"].copy()
    summary = {
        "num_rows": int(len(result_df)),
        "num_ok": int(len(ok)),
        "num_failed": int((result_df["status"] != "ok").sum()),
        "requested_identities": requested_identities,
        "selected_identities": int(selected_identity_count),
        "images_per_identity": images_per_identity,
        "save_comparison_figures": save_comparison_figures,
        "pose_manifest_rows": int(pose_manifest_rows),
        "pose_kept_rows": int(pose_kept_rows),
        "pose_rejected_rows": int(max(pose_manifest_rows - pose_kept_rows, 0)),
        "pose_max_yaw_degrees": float(max_yaw_degrees),
        "pose_require_single_face": bool(require_single_face),
        "adapter_count": int(adapter_count),
        "num_generated_images": float(ok["num_generated_images"].mean()) if len(ok) else float("nan"),
    }
    return pd.DataFrame([summary])


def _save_comparison_figure(
    input_path: Path,
    source_rgb: np.ndarray,
    adaface_image,
    adapter_images: dict[str, object],
    arcface_image,
    output_path: Path,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("Input", source_rgb), ("AdaFace -> Arc2Face", adaface_image)]
    for adapter_name, adapter_image in adapter_images.items():
        panels.append((f"{_pretty_method_name(adapter_name)} -> Arc2Face", adapter_image))
    panels.append(("ArcFace -> Arc2Face", arcface_image))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

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
    adapters = load_adapters(cfg, device)
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

    emb = source_embeddings(input_image, arc_app, ada_model, device)
    if emb is None:
        raise ValueError(f"Could not extract a face embedding from {input_image}")

    source_arc = emb["arcface"]
    source_ada = emb["adaface"]
    source_rgb = np.array(Image.open(input_image).convert("RGB"))

    ada_input = torch.from_numpy(emb["adaface"]).float().unsqueeze(0).to(device)
    arc_input = torch.from_numpy(source_arc).float().unsqueeze(0).to(device)

    stem = input_image.stem
    adapter_dir = recon_dir / f"{stem}_adapters"
    adaface_dir = recon_dir / f"{stem}_adaface_native"
    arc_dir = recon_dir / f"{stem}_arcface_native"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adaface_dir.mkdir(parents=True, exist_ok=True)
    arc_dir.mkdir(parents=True, exist_ok=True)

    adapter_images_by_name: dict[str, object] = {}
    adapter_paths_by_name: dict[str, list[str]] = {}
    adapter_paths_meta: dict[str, str] = {}
    for adapter_index, adapter_item in enumerate(adapters):
        with torch.no_grad():
            adapter_pred = adapter_item.model(ada_input).cpu().numpy()[0].astype(np.float32)
        adapter_pred = l2_normalize(adapter_pred).astype(np.float32)

        adapter_images, adapter_paths_out = _generate_reconstruction(
            cfg=cfg,
            pipeline=pipeline,
            project_face_embs=project_face_embs,
            embedding=adapter_pred,
            label=f"{adapter_item.name}_adapter",
            stem=stem,
            sample_dir=adapter_dir,
            device=device,
            base_seed=base_seed + adapter_index,
            num_images_per_prompt=n_images,
        )
        adapter_images_by_name[adapter_item.name] = adapter_images
        adapter_paths_by_name[adapter_item.name] = [str(p) for p in adapter_paths_out]
        adapter_paths_meta[adapter_item.name] = str(adapter_item.path)

    arcface_pred = arc_input.cpu().numpy()[0].astype(np.float32)
    arcface_pred = l2_normalize(arcface_pred).astype(np.float32)

    adaface_pred = source_ada.astype(np.float32)
    adaface_pred = l2_normalize(adaface_pred).astype(np.float32)

    adaface_images, adaface_paths = _generate_reconstruction(
        cfg=cfg,
        pipeline=pipeline,
        project_face_embs=project_face_embs,
        embedding=adaface_pred,
        label="adaface_native",
        stem=stem,
        sample_dir=adaface_dir,
        device=device,
        base_seed=base_seed + 1,
        num_images_per_prompt=n_images,
    )
    arcface_images, arc_paths = _generate_reconstruction(
        cfg=cfg,
        pipeline=pipeline,
        project_face_embs=project_face_embs,
        embedding=arcface_pred,
        label="arcface_native",
        stem=stem,
        sample_dir=arc_dir,
        device=device,
        base_seed=base_seed + 2,
        num_images_per_prompt=n_images,
    )

    comparison_path = output_dir / f"{stem}_comparison.png"
    _save_comparison_figure(
        input_path=input_image,
        source_rgb=source_rgb,
        adaface_image=adaface_images[0] if adaface_images else source_rgb,
        adapter_images={name: images[0] if images else source_rgb for name, images in adapter_images_by_name.items()},
        arcface_image=arcface_images[0] if arcface_images else source_rgb,
        output_path=comparison_path,
    )

    result = pd.DataFrame(
        [
            {
                "image_path": str(input_image),
                "status": "ok",
                "run_id": run_id,
                "adapter_path": json.dumps(adapter_paths_meta) if len(adapter_paths_meta) > 1 else next(iter(adapter_paths_meta.values()), None),
                "adapter_paths": json.dumps(adapter_paths_meta),
                "comparison_path": str(comparison_path),
                "adaface_native_recon_path": str(adaface_paths[0]) if adaface_paths else None,
                "adaface_native_recon_paths": json.dumps([str(p) for p in adaface_paths]),
                "arcface_recon_path": str(arc_paths[0]) if arc_paths else None,
                "arcface_recon_paths": json.dumps([str(p) for p in arc_paths]),
                "num_generated_images": int(len(adaface_images)),
            }
        ]
    )
    for adapter_name, adapter_paths in adapter_paths_by_name.items():
        result.loc[0, f"{adapter_name}_recon_path"] = adapter_paths[0] if adapter_paths else None
        result.loc[0, f"{adapter_name}_recon_paths"] = json.dumps(adapter_paths)
        result.loc[0, f"{adapter_name}_adapter_path"] = adapter_paths_meta.get(adapter_name)
    result.to_csv(output_dir / "single_image_inference.csv", index=False)
    return {
        "output_dir": output_dir,
        "comparison_path": comparison_path,
        "adapter_paths": adapter_paths_meta,
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
    adapters = load_adapters(cfg, device)
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

    rows, pose_manifest_df = sample_image_rows(
        input_dir,
        cfg["image_extensions"],
        selected_num_identities,
        selected_images_per_identity,
        arc_app=arc_app,
        max_yaw_degrees=cfg["inference_max_yaw_degrees"],
        require_single_face=cfg["inference_pose_require_single_face"],
        seed=base_seed,
    )
    if not rows:
        raise ValueError(f"No images found under {input_dir} after pose filtering")

    selected_df = pd.DataFrame(rows)
    if not selected_df.empty:
        kept_meta = pose_manifest_df[pose_manifest_df["keep"]].copy()
        kept_meta["image_path"] = kept_meta["image_path"].astype(str)
        selected_df["image_path"] = selected_df["image_path"].astype(str)
        selected_df = selected_df.merge(kept_meta.drop(columns=["keep"]), on=["identity", "image_path"], how="left")
    selected_df["run_id"] = run_id
    pose_manifest_df["run_id"] = run_id
    pose_manifest_df.to_csv(output_dir / "pose_filter_report.csv", index=False)
    selected_df.to_csv(output_dir / "selected_samples.csv", index=False)

    figures_dir = output_dir / "figures" if selected_save_figures else None
    result_rows = []
    for idx, row in enumerate(tqdm(rows, desc="Inference")):
        row_result = _process_sample(
            cfg=cfg,
            sample=row,
            adapters=adapters,
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
    summary_df = build_summary(
        result_df,
        selected_num_identities,
        selected_images_per_identity,
        selected_save_figures,
        selected_identity_count=int(selected_df["identity"].nunique()) if not selected_df.empty else 0,
        pose_manifest_rows=int(len(pose_manifest_df)),
        pose_kept_rows=int(pose_manifest_df["keep"].sum()) if not pose_manifest_df.empty else 0,
        max_yaw_degrees=float(cfg["inference_max_yaw_degrees"]),
        require_single_face=bool(cfg["inference_pose_require_single_face"]),
        adapter_count=len(adapters),
    )
    summary_df["run_id"] = run_id
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    return {
        "output_dir": output_dir,
        "run_id": run_id,
        "recon_dir": recon_dir,
        "adapter_paths": {item.name: str(item.path) for item in adapters},
        "results": result_df,
        "summary": summary_df,
    }
