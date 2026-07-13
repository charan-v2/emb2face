from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from .face_backends import build_face_detector, build_face_embedder, extract_face_embedding
from .utils import det_curve_points, verification_summary


matplotlib.use("Agg")


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


def _score_method_rows(
    rows: list[dict[str, Any]],
    detector,
    embedder,
    require_single_face: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_cache: dict[str, Any] = {}
    recon_cache: dict[str, Any] = {}

    valid_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for row in rows:
        source_path = row["source_path"]
        recon_path = row["recon_path"]

        if source_path not in source_cache:
            source_face = extract_face_embedding(
                source_path,
                detector=detector,
                embedder=embedder,
                require_single_face=require_single_face,
            )
            source_cache[source_path] = source_face
        source_face = source_cache[source_path]

        if recon_path not in recon_cache:
            recon_face = extract_face_embedding(
                recon_path,
                detector=detector,
                embedder=embedder,
                require_single_face=require_single_face,
            )
            recon_cache[recon_path] = recon_face
        recon_face = recon_cache[recon_path]

        if source_face is None:
            failed_rows.append({**row, "role": "source", "reason": "source_face_extraction_failed"})
            continue
        if recon_face is None:
            failed_rows.append({**row, "role": "reconstruction", "reason": "reconstruction_face_extraction_failed"})
            continue

        valid_rows.append(
            {
                **row,
                "source_embedding": source_face.embedding,
                "recon_embedding": recon_face.embedding,
                "source_face_count": source_face.face_count,
                "recon_face_count": recon_face.face_count,
                "source_confidence": source_face.confidence,
                "recon_confidence": recon_face.confidence,
            }
        )

    if not valid_rows:
        return (
            pd.DataFrame(valid_rows),
            pd.DataFrame(failed_rows),
            pd.DataFrame(),
        )

    source_embs = np.stack([r["source_embedding"] for r in valid_rows]).astype(np.float32)
    recon_embs = np.stack([r["recon_embedding"] for r in valid_rows]).astype(np.float32)
    identities = np.asarray([r["identity"] for r in valid_rows])

    scores = source_embs @ recon_embs.T
    labels = (identities[:, None] == identities[None, :]).astype(np.int32)

    n = len(valid_rows)
    pair_rows = pd.DataFrame(
        {
            "source_index": np.repeat(np.arange(n), n),
            "recon_index": np.tile(np.arange(n), n),
            "source_identity": np.repeat(identities, n),
            "recon_identity": np.tile(identities, n),
            "source_path": np.repeat([r["source_path"] for r in valid_rows], n),
            "recon_path": np.tile([r["recon_path"] for r in valid_rows], n),
            "label": labels.reshape(-1),
            "score": scores.reshape(-1),
        }
    )

    valid_df = pd.DataFrame(valid_rows).drop(columns=["source_embedding", "recon_embedding"])
    failed_df = pd.DataFrame(failed_rows)
    return valid_df, failed_df, pair_rows


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


def run_score_pipeline(
    cfg: dict,
    input_run_dir: Path,
    output_dir: Path | None = None,
    selected_methods: list[str] | None = None,
):
    input_run_dir = Path(input_run_dir)
    report_path = input_run_dir / "inference_report.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"Inference report not found: {report_path}")

    report_df = pd.read_csv(report_path)
    long_rows = _explode_report_rows(report_df, selected_methods=selected_methods)
    if not long_rows:
        raise ValueError(f"No reconstruction path columns found in {report_path}")

    score_dir = Path(output_dir) if output_dir is not None else input_run_dir / "biometric_eval"
    score_dir.mkdir(parents=True, exist_ok=True)

    detector_backend = cfg.get("score_detector_backend", "insightface")
    embedder_backend = cfg.get("score_embedder_backend", "insightface")
    detector = build_face_detector(cfg)
    embedder = build_face_embedder(cfg)
    require_single_face = bool(cfg.get("score_require_single_face", cfg.get("require_single_face", False)))

    summary_rows: list[dict[str, Any]] = []
    pair_frames: list[pd.DataFrame] = []
    det_frames: list[pd.DataFrame] = []
    failed_frames: list[pd.DataFrame] = []

    for method, method_rows in pd.DataFrame(long_rows).groupby("method"):
        valid_df, failed_df, pair_df = _score_method_rows(
            method_rows.to_dict("records"),
            detector=detector,
            embedder=embedder,
            require_single_face=require_single_face,
        )

        if len(failed_df):
            failed_df = failed_df.copy()
            failed_df["method"] = method
            failed_frames.append(failed_df)

        if pair_df.empty:
            summary_rows.append(
                {
                    "method": method,
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
            continue

        pair_df = pair_df.copy()
        pair_df["method"] = method
        pair_frames.append(pair_df)

        det = det_curve_points(pair_df["label"].to_numpy(), pair_df["score"].to_numpy())
        det_df = pd.DataFrame(det)
        det_df["method"] = method
        det_frames.append(det_df)

        summary_rows.append(
            _score_to_summary(
                method,
                pair_df,
                cfg,
                detector_backend,
                embedder_backend,
                num_valid_rows=len(valid_df),
                num_failed_rows=len(failed_df),
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    pair_scores_df = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    det_curve_df = pd.concat(det_frames, ignore_index=True) if det_frames else pd.DataFrame()
    failed_df = pd.concat(failed_frames, ignore_index=True) if failed_frames else pd.DataFrame()

    summary_df.to_csv(score_dir / "verification_eval.csv", index=False)
    summary_df.to_csv(score_dir / "summary.csv", index=False)
    if not pair_scores_df.empty:
        pair_scores_df.to_csv(score_dir / "verification_scores.csv", index=False)
    if not det_curve_df.empty:
        det_curve_df.to_csv(score_dir / "det_curve.csv", index=False)
        _save_det_plot(det_curve_df, score_dir / "det_curve.png")
    if not failed_df.empty:
        failed_df.to_csv(score_dir / "failed_rows.csv", index=False)

    return {
        "output_dir": score_dir,
        "summary": summary_df,
        "scores": pair_scores_df,
        "det_curve": det_curve_df,
        "failed": failed_df,
    }
