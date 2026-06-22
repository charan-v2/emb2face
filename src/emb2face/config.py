from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "runmode": "FULL",  # "DEBUG" or "FULL"
    "dataset_root": "./data/webface_112x112",
    "output_root": "./outputs/webface_arcada_adapter",
    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp"],
    "seed": 42,
    "device": "auto",  # "auto", "cuda", "mps", or "cpu"
    "det_size": [640, 640],
    "require_single_face": False,
    "min_images_per_identity": 2,
    "debug_max_identities": 25,
    "debug_max_images_per_identity": 8,
    "full_max_identities": None,
    "full_max_images_per_identity": None,
    "train_id_fraction": 0.70,
    "val_id_fraction": 0.15,
    "test_id_fraction": 0.15,
    "batch_size": 256,
    "num_epochs": 25,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "mse_loss_weight": 1.0,
    "cosine_loss_weight": 1.0,
    "hidden_dim": 1024,
    "dropout": 0.10,
    "adapter_type": "linear",  # "linear" or "mlp"
    "pairs_per_split": 3000,
    "experiments": [
        "exp1_arcface_baseline",
        "exp2_adaface_wrongspace",
        "exp3_adapter_mapped",
    ],
    "eval_source": "casia_test",  # "casia_test" or "external"
    "eval_dataset_root": "./data/celeb_eval",
    "eval_identities": 100,
    "eval_images_per_identity": 5,
    "debug_eval_identities": 5,
    "debug_eval_images_per_identity": 2,
    "num_recon_per_image": 1,
    "num_inference_steps": 25,
    "guidance_scale": 3.0,
    "impostors_per_probe": 20,
    "adapter_run_mode": "full",
    "use_test_split_from_notebook1": True,
    "adaface_repo": "minchul/cvlface_adaface_ir50_ms1mv2",
    "arc2face_repo": "FoivosPar/Arc2Face",
    "insight_root": "~/.insightface",
    "arc2face_local_dir": "./.cache/arc2face_models",
}


def _coerce_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    for key in ("dataset_root", "output_root", "eval_dataset_root", "insight_root", "arc2face_local_dir"):
        if key in cfg and cfg[key] is not None:
            cfg[key] = Path(cfg[key]).expanduser()
    return cfg


def resolve_device(device: str | None = None) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_config(path: str | None = None, overrides: Mapping[str, Any] | None = None):
    cfg = DEFAULT_CONFIG.copy()
    if path is not None:
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})

    runmode = str(cfg["runmode"]).upper()
    if runmode not in {"DEBUG", "FULL"}:
        raise ValueError("runmode must be DEBUG or FULL")
    cfg["runmode"] = runmode

    device_value = cfg.get("device", "auto")
    cfg["device"] = resolve_device(None if device_value is None else str(device_value))
    cfg = _coerce_paths(cfg)

    max_ids = cfg["debug_max_identities"] if runmode == "DEBUG" else cfg["full_max_identities"]
    max_imgs = cfg["debug_max_images_per_identity"] if runmode == "DEBUG" else cfg["full_max_images_per_identity"]
    cfg["max_identities"] = max_ids
    cfg["max_images_per_identity"] = max_imgs

    output_root = cfg["output_root"]
    cfg["emb_dir"] = output_root / f"embeddings_{runmode.lower()}"
    cfg["model_dir"] = output_root / f"models_{runmode.lower()}"
    cfg["report_dir"] = output_root / f"reports_{runmode.lower()}"

    for p in (cfg["output_root"], cfg["emb_dir"], cfg["model_dir"], cfg["report_dir"]):
        p.mkdir(parents=True, exist_ok=True)

    return cfg
