from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from .face_backends import build_face_detector, build_face_embedder, extract_face_embedding
from .utils import det_curve_points, verification_summary


matplotlib.use("Agg")
LOGGER = logging.getLogger(__name__)


@dataclass
class CachedFaceEmbedding:
    embedding: np.ndarray
    face_count: int
    confidence: float | None


def _parse_path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v).strip()]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v is not None and str(v).strip()]
        except Exception:
            pass
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    return [text]


def _discover_method_columns(report_df: pd.DataFrame) -> list[tuple[str, str]]:
    methods: list[tuple[str, str]] = []
    for col in report_df.columns:
        if col.endswith("_recon_path"):
            methods.append((col[: -len("_recon_path")], col))
        elif col.endswith("_recon_paths"):
            methods.append((col[: -len("_recon_paths")], col))
    seen = set()
    ordered: list[tuple[str, str]] = []
    for method, col in methods:
        if method not in seen:
            ordered.append((method, col))
            seen.add(method)
    return ordered


def _explode_report_rows(report_df: pd.DataFrame, selected_methods: list[str] | None = None) -> list[dict[str, Any]]:
    method_columns = _discover_method_columns(report_df)
    if selected_methods is not None:
        selected = {m for m in selected_methods}
        method_columns = [item for item in method_columns if item[0] in selected]

    long_rows: list[dict[str, Any]] = []
    for source_row_index, row in report_df.reset_index(drop=True).iterrows():
        source_path = row.get("image_path")
        identity = row.get("identity")
        for method, col in method_columns:
            recon_paths = _parse_path_list(row.get(col))
            for recon_index, recon_path in enumerate(recon_paths):
                long_rows.append(
                    {
                        "source_row_index": int(source_row_index),
                        "recon_row_index": int(source_row_index),
                        "method": method,
                        "identity": identity,
                        "source_path": str(source_path),
                        "recon_path": str(recon_path),
                        "recon_path_index": int(recon_index),
                    }
                )
    return long_rows


def _embedding_column_name(role: str) -> str:
    if role == "reconstruction":
        return "recon_embedding"
    return f"{role}_embedding"


def _cache_face_embedding(face: Any) -> CachedFaceEmbedding:
    confidence = getattr(face, "confidence", None)
    if confidence is None:
        confidence = getattr(face, "det_score", None)
    if confidence is not None:
        confidence = float(confidence)
    if confidence is not None and np.isnan(confidence):
        confidence = None
    return CachedFaceEmbedding(
        embedding=np.asarray(face.embedding, dtype=np.float32).reshape(-1),
        face_count=int(face.face_count),
        confidence=confidence,
    )


def _load_face_cache(cache_path: Path) -> dict[str, CachedFaceEmbedding | None]:
    if not cache_path.exists():
        return {}
    with open(cache_path, "rb") as f:
        raw_cache = pickle.load(f)
    cache: dict[str, CachedFaceEmbedding | None] = {}
    for path_str, entry in raw_cache.items():
        if entry is None:
            cache[str(path_str)] = None
            continue
        if isinstance(entry, CachedFaceEmbedding):
            cache[str(path_str)] = entry
            continue
        if hasattr(entry, "embedding") and hasattr(entry, "face_count"):
            cache[str(path_str)] = CachedFaceEmbedding(
                embedding=np.asarray(entry.embedding, dtype=np.float32).reshape(-1),
                face_count=int(entry.face_count),
                confidence=None if getattr(entry, "confidence", None) is None else float(entry.confidence),
            )
            continue
        if isinstance(entry, dict) and "embedding" in entry and "face_count" in entry:
            confidence = entry.get("confidence")
            if confidence is not None:
                confidence = float(confidence)
            cache[str(path_str)] = CachedFaceEmbedding(
                embedding=np.asarray(entry["embedding"], dtype=np.float32).reshape(-1),
                face_count=int(entry["face_count"]),
                confidence=confidence,
            )
            continue
        cache[str(path_str)] = None
    return cache


def _save_face_cache(cache_path: Path, cache: dict[str, CachedFaceEmbedding | None]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=str(cache_path.parent), delete=False) as tmp:
        pickle.dump(cache, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, cache_path)


def _extract_face_rows(
    rows: list[dict[str, Any]],
    *,
    path_key: str,
    role: str,
    detector,
    embedder,
    require_single_face: bool,
    cache: dict[str, CachedFaceEmbedding | None],
    cache_path: Path | None = None,
    cache_save_every: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    total = len(rows)
    for row_index, row in enumerate(rows, start=1):
        path = row.get(path_key)
        if path is None:
            failed_rows.append({**row, "role": role, "reason": f"{role}_path_missing"})
            continue

        path_str = str(path)
        if path_str not in cache:
            try:
                extracted_face = extract_face_embedding(
                    path_str,
                    detector=detector,
                    embedder=embedder,
                    require_single_face=require_single_face,
                )
                cache[path_str] = None if extracted_face is None else _cache_face_embedding(extracted_face)
            except ValueError as exc:
                cache[path_str] = None
                failed_rows.append(
                    {
                        **row,
                        "role": role,
                        "reason": f"{role}_face_extraction_failed",
                        "error": str(exc),
                    }
                )
                continue
        face = cache[path_str]

        if face is None:
            failed_rows.append({**row, "role": role, "reason": f"{role}_face_extraction_failed"})
        else:
            valid_rows.append(
                {
                    **row,
                    _embedding_column_name(role): face.embedding,
                    f"{role}_face_count": face.face_count,
                    f"{role}_confidence": face.confidence,
                }
            )

        if (
            cache_path is not None
            and cache_save_every > 0
            and (row_index % cache_save_every == 0 or row_index == total)
        ):
            _save_face_cache(cache_path, cache)

        if total and (row_index == total or row_index % 500 == 0):
            LOGGER.info("[score-1a2b] extracted %s embeddings %d/%d", role, row_index, total)

    valid_df = pd.DataFrame(valid_rows)
    failed_df = pd.DataFrame(failed_rows)
    return valid_df, failed_df


def _build_source_lookup(source_valid_df: pd.DataFrame) -> tuple[dict[int, int], dict[str, list[int]]]:
    source_pos_by_row_index: dict[int, int] = {}
    same_identity_rows: dict[str, list[int]] = {}
    for pos, row in source_valid_df.reset_index(drop=True).iterrows():
        row_index = int(row["source_row_index"])
        identity = str(row["identity"])
        source_pos_by_row_index[row_index] = int(pos)
        same_identity_rows.setdefault(identity, []).append(row_index)
    return source_pos_by_row_index, same_identity_rows


def _sample_impostor_source_rows(
    source_valid_df: pd.DataFrame,
    max_impostors_per_reconstruction: int | None,
    seed: int | None,
) -> dict[int, list[int]]:
    source_rows = source_valid_df.reset_index(drop=True)
    source_row_indices = source_rows["source_row_index"].astype(int).tolist()
    source_identities = source_rows["identity"].astype(str).tolist()
    rng = np.random.default_rng(0 if seed is None else int(seed))

    impostors_by_source_row: dict[int, list[int]] = {}
    for pos, source_row_index in enumerate(source_row_indices):
        source_identity = source_identities[pos]
        candidates = [idx for idx, identity in zip(source_row_indices, source_identities) if identity != source_identity]
        if not candidates:
            impostors_by_source_row[source_row_index] = []
            continue

        if max_impostors_per_reconstruction is None or max_impostors_per_reconstruction <= 0:
            chosen = candidates
        elif len(candidates) <= max_impostors_per_reconstruction:
            chosen = candidates
        else:
            chosen = rng.choice(candidates, size=max_impostors_per_reconstruction, replace=False).tolist()
        impostors_by_source_row[source_row_index] = sorted(int(idx) for idx in chosen)

    return impostors_by_source_row


def _build_type_i_pairs(
    valid_recon_df: pd.DataFrame,
    source_valid_df: pd.DataFrame,
    source_pos_by_row_index: dict[int, int],
    impostors_by_source_row: dict[int, list[int]],
) -> pd.DataFrame:
    if valid_recon_df.empty or source_valid_df.empty:
        return pd.DataFrame()

    source_rows = source_valid_df.reset_index(drop=True)
    source_embs = np.stack(source_rows["source_embedding"].to_list()).astype(np.float32)
    source_emb_by_row_index = {
        int(row["source_row_index"]): source_embs[pos]
        for pos, row in source_rows.iterrows()
    }
    source_identity_by_row_index = {
        int(row["source_row_index"]): str(row["identity"])
        for _, row in source_rows.iterrows()
    }
    source_path_by_row_index = {
        int(row["source_row_index"]): str(row["source_path"])
        for _, row in source_rows.iterrows()
    }

    pair_rows: list[dict[str, Any]] = []
    for _, recon_row in valid_recon_df.iterrows():
        recon_source_row_index = int(recon_row["source_row_index"])
        source_pos = source_pos_by_row_index.get(recon_source_row_index)
        if source_pos is None:
            continue

        recon_emb = np.asarray(recon_row["recon_embedding"], dtype=np.float32)
        exact_source_emb = source_embs[source_pos]
        pair_rows.append(
            {
                "source_row_index": recon_source_row_index,
                "recon_row_index": int(recon_row["recon_row_index"]),
                "source_identity": source_identity_by_row_index[recon_source_row_index],
                "recon_identity": str(recon_row["identity"]),
                "source_path": source_path_by_row_index[recon_source_row_index],
                "recon_path": str(recon_row["recon_path"]),
                "label": 1,
                "score": float(recon_emb @ exact_source_emb),
                "comparison_type": "type_i_genuine",
            }
        )

        for impostor_source_row_index in impostors_by_source_row.get(recon_source_row_index, []):
            impostor_pos = source_pos_by_row_index.get(impostor_source_row_index)
            if impostor_pos is None:
                continue
            pair_rows.append(
                {
                    "source_row_index": impostor_source_row_index,
                    "recon_row_index": int(recon_row["recon_row_index"]),
                    "source_identity": source_identity_by_row_index[impostor_source_row_index],
                    "recon_identity": str(recon_row["identity"]),
                    "source_path": source_path_by_row_index[impostor_source_row_index],
                    "recon_path": str(recon_row["recon_path"]),
                    "label": 0,
                    "score": float(recon_emb @ source_emb_by_row_index[impostor_source_row_index]),
                    "comparison_type": "type_i_impostor",
                }
            )

    return pd.DataFrame(pair_rows)


def _build_type_ii_pairs(
    valid_recon_df: pd.DataFrame,
    source_valid_df: pd.DataFrame,
    source_pos_by_row_index: dict[int, int],
    same_identity_rows: dict[str, list[int]],
    impostors_by_source_row: dict[int, list[int]],
) -> pd.DataFrame:
    if valid_recon_df.empty or source_valid_df.empty:
        return pd.DataFrame()

    source_rows = source_valid_df.reset_index(drop=True)
    source_embs = np.stack(source_rows["source_embedding"].to_list()).astype(np.float32)
    source_emb_by_row_index = {
        int(row["source_row_index"]): source_embs[pos]
        for pos, row in source_rows.iterrows()
    }
    source_identity_by_row_index = {
        int(row["source_row_index"]): str(row["identity"])
        for _, row in source_rows.iterrows()
    }
    source_path_by_row_index = {
        int(row["source_row_index"]): str(row["source_path"])
        for _, row in source_rows.iterrows()
    }

    pair_rows: list[dict[str, Any]] = []
    for _, recon_row in valid_recon_df.iterrows():
        recon_source_row_index = int(recon_row["source_row_index"])
        source_pos = source_pos_by_row_index.get(recon_source_row_index)
        if source_pos is None:
            continue

        recon_identity = str(recon_row["identity"])
        recon_emb = np.asarray(recon_row["recon_embedding"], dtype=np.float32)

        for genuine_source_row_index in same_identity_rows.get(recon_identity, []):
            if genuine_source_row_index == recon_source_row_index:
                continue
            genuine_pos = source_pos_by_row_index.get(genuine_source_row_index)
            if genuine_pos is None:
                continue
            pair_rows.append(
                {
                    "source_row_index": genuine_source_row_index,
                    "recon_row_index": int(recon_row["recon_row_index"]),
                    "source_identity": source_identity_by_row_index[genuine_source_row_index],
                    "recon_identity": recon_identity,
                    "source_path": source_path_by_row_index[genuine_source_row_index],
                    "recon_path": str(recon_row["recon_path"]),
                    "label": 1,
                    "score": float(recon_emb @ source_emb_by_row_index[genuine_source_row_index]),
                    "comparison_type": "type_ii_genuine",
                }
            )

        for impostor_source_row_index in impostors_by_source_row.get(recon_source_row_index, []):
            impostor_pos = source_pos_by_row_index.get(impostor_source_row_index)
            if impostor_pos is None:
                continue
            if source_identity_by_row_index[impostor_source_row_index] == recon_identity:
                continue
            pair_rows.append(
                {
                    "source_row_index": impostor_source_row_index,
                    "recon_row_index": int(recon_row["recon_row_index"]),
                    "source_identity": source_identity_by_row_index[impostor_source_row_index],
                    "recon_identity": recon_identity,
                    "source_path": source_path_by_row_index[impostor_source_row_index],
                    "recon_path": str(recon_row["recon_path"]),
                    "label": 0,
                    "score": float(recon_emb @ source_emb_by_row_index[impostor_source_row_index]),
                    "comparison_type": "type_ii_impostor",
                }
            )

    return pd.DataFrame(pair_rows)


def _score_method_rows(
    rows: list[dict[str, Any]],
    source_valid_df: pd.DataFrame,
    source_pos_by_row_index: dict[int, int],
    same_identity_rows: dict[str, list[int]],
    impostors_by_source_row: dict[int, list[int]],
    detector,
    embedder,
    require_single_face: bool,
    face_cache: dict[str, Any],
    cache_path: Path | None = None,
    cache_save_every: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_recon_df, failed_rows_df = _extract_face_rows(
        rows,
        path_key="recon_path",
        role="reconstruction",
        detector=detector,
        embedder=embedder,
        require_single_face=require_single_face,
        cache=face_cache,
        cache_path=cache_path,
        cache_save_every=cache_save_every,
    )

    if valid_recon_df.empty:
        return valid_recon_df, failed_rows_df, pd.DataFrame(), pd.DataFrame()

    type_i_df = _build_type_i_pairs(
        valid_recon_df,
        source_valid_df,
        source_pos_by_row_index,
        impostors_by_source_row,
    )
    type_ii_df = _build_type_ii_pairs(
        valid_recon_df,
        source_valid_df,
        source_pos_by_row_index,
        same_identity_rows,
        impostors_by_source_row,
    )

    return valid_recon_df, failed_rows_df, type_i_df, type_ii_df


def _score_to_summary(
    method: str,
    pair_df: pd.DataFrame,
    cfg: dict,
    detector_backend: str,
    embedder_backend: str,
    num_valid_rows: int,
    num_failed_rows: int,
) -> dict[str, Any]:
    labels = pair_df["label"].to_numpy()
    scores = pair_df["score"].to_numpy()
    summary = verification_summary(labels, scores)
    genuine = pair_df[pair_df["label"] == 1]["score"]
    impostor = pair_df[pair_df["label"] == 0]["score"]
    return {
        "method": method,
        "num_valid_rows": int(num_valid_rows),
        "num_failed_rows": int(num_failed_rows),
        "num_pairs": int(len(pair_df)),
        "num_genuine": int((pair_df["label"] == 1).sum()),
        "num_impostor": int((pair_df["label"] == 0).sum()),
        "mean_genuine_score": float(genuine.mean()) if len(genuine) else float("nan"),
        "mean_impostor_score": float(impostor.mean()) if len(impostor) else float("nan"),
        **summary,
        "detector_backend": detector_backend,
        "embedder_backend": embedder_backend,
        "require_single_face": bool(cfg.get("score_require_single_face", cfg.get("require_single_face", False))),
    }


def _save_det_plot(det_curve_df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, sub in det_curve_df.groupby("method"):
        ax.plot(sub["far"], sub["fnmr"], label=method)
    ax.set_xlabel("FAR")
    ax.set_ylabel("FNMR")
    ax.set_title("DET Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def _find_latest_run_dir(base_dir: Path) -> Path | None:
    if not base_dir.exists() or not base_dir.is_dir():
        return None
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def run_score_pipeline(
    cfg: dict,
    input_run_dir: Path | None,
    output_dir: Path | None = None,
    selected_methods: list[str] | None = None,
):
    LOGGER.info("[score-1a2b] starting score pipeline")
    if input_run_dir is None:
        configured = cfg.get("score_input_run_dir")
        if configured is not None:
            input_run_dir = Path(configured)
            LOGGER.info("[score-1a2b] using score_input_run_dir from config: %s", input_run_dir)
        else:
            base_dir = cfg.get("inference_output_dir")
            if base_dir is None:
                base_dir = cfg["output_root"] / f"inference_{cfg['runmode'].lower()}"
            input_run_dir = _find_latest_run_dir(Path(base_dir))
            if input_run_dir is None:
                raise FileNotFoundError(
                    "Could not infer a score input run directory. Set score_input_run_dir in the config or pass --input-run-dir."
                )
            LOGGER.info("[score-1a2b] auto-selected latest inference run: %s", input_run_dir)

    input_run_dir = Path(input_run_dir)
    report_path = input_run_dir / "inference_report.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"Inference report not found: {report_path}")
    LOGGER.info("[score-1a2b] reading inference report: %s", report_path)

    report_df = pd.read_csv(report_path)
    LOGGER.info("[score-1a2b] loaded %d report row(s) and %d column(s)", len(report_df), len(report_df.columns))
    long_rows = _explode_report_rows(report_df, selected_methods=selected_methods)
    if not long_rows:
        raise ValueError(f"No reconstruction path columns found in {report_path}")
    LOGGER.info("[score-1a2b] expanded to %d reconstruction row(s)", len(long_rows))

    score_dir = Path(output_dir) if output_dir is not None else input_run_dir / "biometric_eval"
    score_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("[score-1a2b] writing outputs to %s", score_dir)
    face_cache_path = score_dir / "face_embedding_cache.pkl"
    face_cache = _load_face_cache(face_cache_path)
    if face_cache:
        LOGGER.info("[score-1a2b] loaded %d cached face embedding(s) from %s", len(face_cache), face_cache_path)
    face_cache_save_every = int(cfg.get("score_face_cache_save_every", 500) or 0)
    LOGGER.info("[score-1a2b] face cache checkpoint interval=%s", "disabled" if face_cache_save_every <= 0 else face_cache_save_every)

    detector_backend = cfg.get("score_detector_backend", "insightface")
    embedder_backend = cfg.get("score_embedder_backend", "insightface")
    LOGGER.info(
        "[score-1a2b] building detector=%s embedder=%s",
        detector_backend,
        embedder_backend,
    )
    detector = build_face_detector(cfg)
    embedder = build_face_embedder(cfg)
    require_single_face = bool(cfg.get("score_require_single_face", cfg.get("require_single_face", False)))
    LOGGER.info("[score-1a2b] require_single_face=%s", require_single_face)

    source_rows = [
        {
            "source_row_index": int(source_row_index),
            "identity": row.get("identity"),
            "source_path": str(row.get("image_path")),
        }
        for source_row_index, row in report_df.reset_index(drop=True).iterrows()
    ]
    LOGGER.info("[score-1a2b] extracting source gallery embeddings for %d row(s)", len(source_rows))
    source_valid_df, source_failed_df = _extract_face_rows(
        source_rows,
        path_key="source_path",
        role="source",
        detector=detector,
        embedder=embedder,
        require_single_face=require_single_face,
        cache=face_cache,
        cache_path=face_cache_path,
        cache_save_every=face_cache_save_every,
    )
    LOGGER.info(
        "[score-1a2b] source gallery produced %d valid row(s) and %d failed row(s)",
        len(source_valid_df),
        len(source_failed_df),
    )
    source_pos_by_row_index, same_identity_rows = _build_source_lookup(source_valid_df)
    max_impostors_per_reconstruction = cfg.get("score_max_impostors_per_reconstruction")
    if max_impostors_per_reconstruction is not None:
        max_impostors_per_reconstruction = int(max_impostors_per_reconstruction)
    impostor_sampling_seed = cfg.get("score_impostor_sampling_seed", cfg.get("seed"))
    impostors_by_source_row = _sample_impostor_source_rows(
        source_valid_df,
        max_impostors_per_reconstruction=max_impostors_per_reconstruction,
        seed=impostor_sampling_seed,
    )
    LOGGER.info(
        "[score-1a2b] sampled up to %s impostor source row(s) per reconstruction using seed=%s",
        "all" if not max_impostors_per_reconstruction or max_impostors_per_reconstruction <= 0 else max_impostors_per_reconstruction,
        impostor_sampling_seed,
    )

    type_i_summary_rows: list[dict[str, Any]] = []
    type_ii_summary_rows: list[dict[str, Any]] = []
    type_i_frames: list[pd.DataFrame] = []
    type_ii_frames: list[pd.DataFrame] = []
    det_frames: list[pd.DataFrame] = []
    failed_frames: list[pd.DataFrame] = []
    if not source_failed_df.empty:
        failed_frames.append(source_failed_df)

    long_df = pd.DataFrame(long_rows)
    LOGGER.info("[score-1a2b] scoring %d method(s): %s", long_df["method"].nunique(), ", ".join(sorted(long_df["method"].unique())))
    for method, method_rows in long_df.groupby("method"):
        LOGGER.info("[score-1a2b] scoring method=%s with %d row(s)", method, len(method_rows))
        valid_df, failed_df, type_i_df, type_ii_df = _score_method_rows(
            method_rows.to_dict("records"),
            source_valid_df=source_valid_df,
            source_pos_by_row_index=source_pos_by_row_index,
            same_identity_rows=same_identity_rows,
            impostors_by_source_row=impostors_by_source_row,
            detector=detector,
            embedder=embedder,
            require_single_face=require_single_face,
            face_cache=face_cache,
            cache_path=face_cache_path,
            cache_save_every=face_cache_save_every,
        )
        LOGGER.info(
            "[score-1a2b] method=%s produced %d valid reconstruction row(s), %d failed row(s), %d type-I pair(s), %d type-II pair(s)",
            method,
            len(valid_df),
            len(failed_df),
            len(type_i_df),
            len(type_ii_df),
        )

        if len(failed_df):
            failed_df = failed_df.copy()
            failed_df["method"] = method
            failed_frames.append(failed_df)

        if not type_i_df.empty:
            type_i_df = type_i_df.copy()
            type_i_df["method"] = method
            type_i_frames.append(type_i_df)
            LOGGER.info("[score-1a2b] computing DET curve for method=%s from %d type-I pair(s)", method, len(type_i_df))
            det = det_curve_points(type_i_df["label"].to_numpy(), type_i_df["score"].to_numpy())
            det_df = pd.DataFrame(det)
            det_df["method"] = method
            det_frames.append(det_df)
            type_i_summary_rows.append(
                {
                    "method": method,
                    "comparison_type": "type_i",
                    "num_valid_rows": int(len(valid_df)),
                    "num_failed_rows": int(len(failed_df)),
                    "num_pairs": int(len(type_i_df)),
                    "mean_exact_score": float(type_i_df["score"].mean()),
                    "min_exact_score": float(type_i_df["score"].min()),
                    "max_exact_score": float(type_i_df["score"].max()),
                    "detector_backend": detector_backend,
                    "embedder_backend": embedder_backend,
                    "require_single_face": require_single_face,
                }
            )
        else:
            type_i_summary_rows.append(
                {
                    "method": method,
                    "comparison_type": "type_i",
                    "num_valid_rows": int(len(valid_df)),
                    "num_failed_rows": int(len(failed_df)),
                    "num_pairs": 0,
                    "mean_exact_score": float("nan"),
                    "min_exact_score": float("nan"),
                    "max_exact_score": float("nan"),
                    "detector_backend": detector_backend,
                    "embedder_backend": embedder_backend,
                    "require_single_face": require_single_face,
                }
            )

        if not type_ii_df.empty:
            type_ii_df = type_ii_df.copy()
            type_ii_df["method"] = method
            type_ii_frames.append(type_ii_df)
            type_ii_summary_rows.append(
                {
                    **_score_to_summary(
                        method,
                        type_ii_df,
                        cfg,
                        detector_backend,
                        embedder_backend,
                        num_valid_rows=len(valid_df),
                        num_failed_rows=len(failed_df),
                    ),
                    "comparison_type": "type_ii",
                }
            )
        else:
            LOGGER.info("[score-1a2b] method=%s has no type-II pairs after filtering", method)
            type_ii_summary_rows.append(
                {
                    "method": method,
                    "comparison_type": "type_ii",
                    "num_valid_rows": int(len(valid_df)),
                    "num_failed_rows": int(len(failed_df)),
                    "num_pairs": 0,
                    "num_genuine": 0,
                    "num_impostor": 0,
                    "mean_genuine_score": float("nan"),
                    "mean_impostor_score": float("nan"),
                    "eer": float("nan"),
                    "eer_threshold": float("nan"),
                    "far_at_eer": float("nan"),
                    "frr_at_eer": float("nan"),
                    "fmr_at_eer": float("nan"),
                    "fnmr_at_eer": float("nan"),
                    "detector_backend": detector_backend,
                    "embedder_backend": embedder_backend,
                    "require_single_face": require_single_face,
                }
            )

    type_i_summary_df = pd.DataFrame(type_i_summary_rows)
    type_ii_summary_df = pd.DataFrame(type_ii_summary_rows)
    summary_df = type_i_summary_df
    LOGGER.info("[score-1a2b] concatenating outputs")
    type_i_scores_df = pd.concat(type_i_frames, ignore_index=True) if type_i_frames else pd.DataFrame()
    type_ii_scores_df = pd.concat(type_ii_frames, ignore_index=True) if type_ii_frames else pd.DataFrame()
    det_curve_df = pd.concat(det_frames, ignore_index=True) if det_frames else pd.DataFrame()
    failed_df = pd.concat(failed_frames, ignore_index=True) if failed_frames else pd.DataFrame()

    LOGGER.info("[score-1a2b] writing verification_eval.csv and summary.csv")
    summary_df.to_csv(score_dir / "verification_eval.csv", index=False)
    summary_df.to_csv(score_dir / "summary.csv", index=False)
    if not type_i_summary_df.empty:
        LOGGER.info("[score-1a2b] writing typeI_summary.csv with %d row(s)", len(type_i_summary_df))
        type_i_summary_df.to_csv(score_dir / "typeI_summary.csv", index=False)
    if not type_ii_summary_df.empty:
        LOGGER.info("[score-1a2b] writing typeII_summary.csv with %d row(s)", len(type_ii_summary_df))
        type_ii_summary_df.to_csv(score_dir / "typeII_summary.csv", index=False)
    if not type_i_scores_df.empty:
        LOGGER.info("[score-1a2b] writing verification_scores.csv and typeI_scores.csv with %d row(s)", len(type_i_scores_df))
        type_i_scores_df.to_csv(score_dir / "verification_scores.csv", index=False)
        type_i_scores_df.to_csv(score_dir / "typeI_scores.csv", index=False)
    if not type_ii_scores_df.empty:
        LOGGER.info("[score-1a2b] writing typeII_scores.csv with %d row(s)", len(type_ii_scores_df))
        type_ii_scores_df.to_csv(score_dir / "typeII_scores.csv", index=False)
    if not det_curve_df.empty:
        LOGGER.info("[score-1a2b] writing det_curve.csv and det_curve.png with %d row(s)", len(det_curve_df))
        det_curve_df.to_csv(score_dir / "det_curve.csv", index=False)
        _save_det_plot(det_curve_df, score_dir / "det_curve.png")
    if not failed_df.empty:
        LOGGER.info("[score-1a2b] writing failed_rows.csv with %d row(s)", len(failed_df))
        failed_df.to_csv(score_dir / "failed_rows.csv", index=False)

    LOGGER.info("[score-1a2b] score pipeline complete")
    return {
        "output_dir": score_dir,
        "summary": summary_df,
        "scores": type_i_scores_df,
        "type_i_scores": type_i_scores_df,
        "type_ii_scores": type_ii_scores_df,
        "det_curve": det_curve_df,
        "failed": failed_df,
    }
