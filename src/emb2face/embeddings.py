from __future__ import annotations

import importlib.util
import os
import pickle
import shutil
import zipfile
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image as PILImage
from tqdm.auto import tqdm
from torchvision.transforms import Compose, Normalize, ToTensor

from .dataset import collect_dataset
from .utils import l2_normalize


_REC_MODEL_CACHE: dict[int, Any] = {}


def setup_device(cfg: dict) -> torch.device:
    device = str(cfg.get("device", "auto"))
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def ensure_antelopev2_pack(cfg: dict) -> Path:
    insight_root = Path(cfg["insight_root"]).expanduser()
    model_root = insight_root / "models"
    ant_dir = model_root / "antelopev2"
    ant_dir.mkdir(parents=True, exist_ok=True)

    arcface_path = ant_dir / "arcface.onnx"
    if arcface_path.exists():
        return ant_dir

    model_root.mkdir(parents=True, exist_ok=True)
    zip_path = Path("antelopev2.zip")
    import gdown

    gdown.download(
        id="18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8",
        output=str(zip_path),
        quiet=True,
    )
    if ant_dir.exists():
        shutil.rmtree(ant_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(model_root)

    hf_hub_download(repo_id=cfg["arc2face_repo"], filename="arcface.onnx", local_dir=str(ant_dir))
    glintr = ant_dir / "glintr100.onnx"
    if glintr.exists():
        glintr.unlink()
    return ant_dir


def load_arcface_app(cfg: dict):
    from insightface.app import FaceAnalysis

    ant_dir = ensure_antelopev2_pack(cfg)
    app = FaceAnalysis(
        name="antelopev2",
        root=str(Path(cfg["insight_root"]).expanduser()),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(
        ctx_id=0 if torch.cuda.is_available() else -1,
        det_size=tuple(cfg["det_size"]),
    )
    return app, ant_dir


def load_adaface_model(cfg: dict, device: torch.device):
    repo_id = cfg["adaface_repo"]
    local_dir = Path("/tmp") / repo_id.split("/")[-1]
    if not local_dir.exists():
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False)

    cwd = os.getcwd()
    try:
        os.chdir(local_dir)
        if str(local_dir) not in sys.path:
            sys.path.insert(0, str(local_dir))
        spec = importlib.util.spec_from_file_location("cvlface_wrapper", local_dir / "wrapper.py")
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if not hasattr(module.CVLFaceRecognitionModel, "all_tied_weights_keys"):
            module.CVLFaceRecognitionModel.all_tied_weights_keys = {}
        model = module.CVLFaceRecognitionModel.from_pretrained(str(local_dir), trust_remote_code=True)
    finally:
        os.chdir(cwd)

    return model.eval().to(device)


def _adaface_transform():
    return Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def adaface_from_crop(crop_rgb_112: np.ndarray, ada_model, device: torch.device) -> tuple[np.ndarray, float]:
    tensor = _adaface_transform()(PILImage.fromarray(crop_rgb_112)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = ada_model(tensor)
    feat = out[0] if isinstance(out, (tuple, list)) else out
    feat = feat.detach().cpu().numpy()[0].astype(np.float32)
    return l2_normalize(feat).astype(np.float32), float(np.linalg.norm(feat))


def get_largest_face(faces):
    return sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]


def get_rec_model(arc_app):
    cache_key = id(arc_app)
    if cache_key not in _REC_MODEL_CACHE:
        rec_model = None
        for key in arc_app.models:
            model = arc_app.models[key]
            if hasattr(model, "get_feat"):
                rec_model = model
                break
        if rec_model is None:
            raise RuntimeError(f"No recognition model found in arc_app.models. Keys: {list(arc_app.models.keys())}")
        _REC_MODEL_CACHE[cache_key] = rec_model
    return _REC_MODEL_CACHE[cache_key]


def aligned_arcface_and_crop(img_bgr: np.ndarray, arc_app) -> tuple[np.ndarray | None, np.ndarray | None]:
    from insightface.utils import face_align

    h, w = img_bgr.shape[:2]
    if h <= 128 and w <= 128:
        crop_bgr = cv2.resize(img_bgr, (112, 112)) if (h != 112 or w != 112) else img_bgr
    else:
        faces = arc_app.get(img_bgr)
        if len(faces) == 0:
            return None, None
        face = get_largest_face(faces)
        crop_bgr = face_align.norm_crop(img_bgr, landmark=face.kps, image_size=112)

    emb = get_rec_model(arc_app).get_feat([crop_bgr])[0]
    emb = l2_normalize(emb.astype(np.float32)).astype(np.float32)
    return emb, cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def source_embeddings(path: str | Path, arc_app, ada_model, device: torch.device):
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return None
    arc, crop_rgb = aligned_arcface_and_crop(img_bgr, arc_app)
    if arc is None:
        return None
    ada, ada_raw_norm = adaface_from_crop(crop_rgb, ada_model, device)
    return {"arcface": arc, "adaface": ada, "adaface_raw_norm": ada_raw_norm, "crop_rgb": crop_rgb}


def arcface_from_image(path: str | Path, arc_app):
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return None
    arc, _ = aligned_arcface_and_crop(img_bgr, arc_app)
    return arc


def collect_and_extract_embeddings(cfg: dict):
    dataset_root = Path(cfg["dataset_root"])
    chkpt_records = cfg["emb_dir"] / "ckpt_records.pkl"
    chkpt_arc = cfg["emb_dir"] / "ckpt_arc_embs.npy"
    chkpt_ada = cfg["emb_dir"] / "ckpt_ada_embs.npy"
    final_arc = cfg["emb_dir"] / "arcface_embeddings.npy"
    final_ada = cfg["emb_dir"] / "adaface_embeddings.npy"
    final_csv = cfg["report_dir"] / "paired_metadata.csv"

    if final_arc.exists() and final_ada.exists() and final_csv.exists():
        arc_embs = np.load(final_arc)
        ada_embs = np.load(final_ada)
        paired_df = pd.read_csv(final_csv)
        return arc_embs, ada_embs, paired_df

    device = setup_device(cfg)
    arc_app, _ = load_arcface_app(cfg)
    ada_model = load_adaface_model(cfg, device)

    df = collect_dataset(
        dataset_root,
        cfg["image_extensions"],
        max_identities=cfg["max_identities"],
        max_images_per_identity=cfg["max_images_per_identity"],
        min_images_per_identity=cfg["min_images_per_identity"],
    )
    if df.empty:
        raise ValueError(f"No images found under {dataset_root}")

    records, arc_embs, ada_embs = [], [], []
    done_paths = set()
    fail_counter = Counter()

    if chkpt_records.exists() and chkpt_arc.exists() and chkpt_ada.exists():
        with open(chkpt_records, "rb") as f:
            records = pickle.load(f)
        arc_embs = list(np.load(chkpt_arc))
        ada_embs = list(np.load(chkpt_ada))
        done_paths = {r["image_path"] for r in records}

    remaining_df = df[~df["image_path"].isin(done_paths)].reset_index(drop=True)

    for i, row in enumerate(tqdm(remaining_df.to_dict("records"), desc="Extracting paired embeddings")):
        result = source_embeddings(row["image_path"], arc_app, ada_model, device)
        if result is None:
            fail_counter["extract_failed"] += 1
            continue

        rec = dict(row)
        rec["adaface_raw_norm"] = result["adaface_raw_norm"]
        rec["face_count"] = 1
        rec["arc_index"] = len(arc_embs)
        rec["ada_index"] = len(ada_embs)
        records.append(rec)
        arc_embs.append(result["arcface"])
        ada_embs.append(result["adaface"])

        if (i + 1) % 500 == 0:
            np.save(chkpt_arc, np.stack(arc_embs).astype(np.float32))
            np.save(chkpt_ada, np.stack(ada_embs).astype(np.float32))
            with open(chkpt_records, "wb") as f:
                pickle.dump(records, f)

    paired_df = pd.DataFrame(records)
    arcarr = np.stack(arc_embs).astype(np.float32) if arc_embs else np.empty((0, 512), dtype=np.float32)
    adaarr = np.stack(ada_embs).astype(np.float32) if ada_embs else np.empty((0, 512), dtype=np.float32)

    paired_df.to_csv(final_csv, index=False)
    np.save(final_arc, arcarr)
    np.save(final_ada, adaarr)

    for p in (chkpt_records, chkpt_arc, chkpt_ada):
        if p.exists():
            p.unlink()

    if len(fail_counter):
        print("Embedding extraction failures:", dict(fail_counter))

    return arcarr, adaarr, paired_df
