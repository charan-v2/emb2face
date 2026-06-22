from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .utils import compute_eer, cosine_similarity_np, l2_normalize, sample_verification_pairs


def predict_embeddings(df_subset, ada_embs, arc_embs, adapter, device):
    preds, gts = [], []
    adapter.eval()
    with torch.no_grad():
        for _, row in df_subset.iterrows():
            a = torch.from_numpy(ada_embs[int(row["ada_index"])]).float().unsqueeze(0).to(device)
            pred = adapter(a).cpu().numpy()[0]
            preds.append(l2_normalize(pred.astype(np.float32)))
            gts.append(arc_embs[int(row["arc_index"])])
    return np.stack(preds), np.stack(gts)


def eval_embedding_alignment(val_df, test_df, ada_embs, arc_embs, adapter, report_dir: Path, cfg: dict):
    device = torch.device(cfg["device"] if cfg["device"] in {"cuda", "mps"} else "cpu")
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if cfg["device"] == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")

    adapter.to(device)

    val_pred, val_gt = predict_embeddings(val_df, ada_embs, arc_embs, adapter, device)
    test_pred, test_gt = predict_embeddings(test_df, ada_embs, arc_embs, adapter, device)

    rows = [
        {
            "split": "val",
            "mean_cosine_to_target_arcface": float(
                np.mean([cosine_similarity_np(a, b) for a, b in zip(val_pred, val_gt)])
            ),
        },
        {
            "split": "test",
            "mean_cosine_to_target_arcface": float(
                np.mean([cosine_similarity_np(a, b) for a, b in zip(test_pred, test_gt)])
            ),
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(report_dir / "embedding_eval.csv", index=False)
    return df


def get_split_mats(df_subset, arc_embs, ada_embs, pred_override=None):
    df_reset = df_subset.reset_index(drop=True)
    arc = np.stack([arc_embs[int(r["arc_index"])] for _, r in df_reset.iterrows()])
    ada = np.stack([ada_embs[int(r["ada_index"])] for _, r in df_reset.iterrows()])
    return arc, ada, pred_override


def eval_verification(df_subset, emb_matrix, split_name: str, label: str, cfg: dict):
    local_df = df_subset.reset_index(drop=True).copy()
    labels, scores = sample_verification_pairs(
        local_df,
        emb_matrix,
        n_pairs=cfg["pairs_per_split"],
        seed=cfg.get("seed"),
    )
    eer, threshold = compute_eer(labels, scores)
    return {
        "split": split_name,
        "space": label,
        "num_pairs": len(scores),
        "mean_score": float(np.mean(scores)),
        "eer": eer,
        "eer_threshold": threshold,
    }


def eval_all_verification(val_df, test_df, arc_embs, ada_embs, adapter, report_dir: Path, cfg: dict):
    device = torch.device(cfg["device"] if cfg["device"] in {"cuda", "mps"} else "cpu")
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if cfg["device"] == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")

    adapter.to(device)

    val_pred, _ = predict_embeddings(val_df, ada_embs, arc_embs, adapter, device)
    test_pred, _ = predict_embeddings(test_df, ada_embs, arc_embs, adapter, device)

    val_arc, val_ada, _ = get_split_mats(val_df, arc_embs, ada_embs)
    test_arc, test_ada, _ = get_split_mats(test_df, arc_embs, ada_embs)

    rows = [
        eval_verification(val_df, val_arc, "val", "arcface_native", cfg),
        eval_verification(val_df, val_ada, "val", "adaface_raw", cfg),
        eval_verification(val_df, val_pred, "val", "adaface_to_arcface_adapter", cfg),
        eval_verification(test_df, test_arc, "test", "arcface_native", cfg),
        eval_verification(test_df, test_ada, "test", "adaface_raw", cfg),
        eval_verification(test_df, test_pred, "test", "adaface_to_arcface_adapter", cfg),
    ]

    df = pd.DataFrame(rows)
    df.to_csv(report_dir / "verification_eval.csv", index=False)
    return df
