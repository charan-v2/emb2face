from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .embeddings import collect_and_extract_embeddings, setup_device
from .evaluate import eval_all_verification, eval_embedding_alignment
from .models import LinearAdapter, MLPAdapter, PairDataset


def cosine_loss(pred, target):
    pred = F.normalize(pred, dim=1)
    target = F.normalize(target, dim=1)
    return 1 - (pred * target).sum(dim=1).mean()


def total_loss(pred, target):
    mse = F.mse_loss(pred, target)
    cos = cosine_loss(pred, target)
    return mse + cos, mse.detach(), cos.detach()


def split_identities(paired_df: pd.DataFrame, cfg: dict):
    identities = sorted(paired_df["identity"].unique())
    train_ids, temp_ids = train_test_split(
        identities,
        test_size=1 - cfg["train_id_fraction"],
        random_state=cfg["seed"],
    )
    val_ratio_within_temp = cfg["val_id_fraction"] / (cfg["val_id_fraction"] + cfg["test_id_fraction"])
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=1 - val_ratio_within_temp,
        random_state=cfg["seed"],
    )

    train_id_set = set(train_ids)
    val_id_set = set(val_ids)

    def label_split(identity):
        if identity in train_id_set:
            return "train"
        if identity in val_id_set:
            return "val"
        return "test"

    paired_df = paired_df.copy()
    paired_df["split"] = paired_df["identity"].map(label_split)

    train_df = paired_df[paired_df["split"] == "train"].reset_index(drop=True)
    val_df = paired_df[paired_df["split"] == "val"].reset_index(drop=True)
    test_df = paired_df[paired_df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


def get_adapter(cfg: dict, device: torch.device):
    if cfg["adapter_type"] == "linear":
        model = LinearAdapter(512)
    elif cfg["adapter_type"] == "mlp":
        model = MLPAdapter(512, cfg["hidden_dim"], cfg["dropout"])
    else:
        raise ValueError("adapter_type must be linear or mlp")
    return model.to(device)


def train_adapter(arc_embs: np.ndarray, ada_embs: np.ndarray, paired_df: pd.DataFrame, cfg: dict):
    device = setup_device(cfg)
    train_df, val_df, test_df = split_identities(paired_df, cfg)

    train_loader = DataLoader(
        PairDataset(train_df, ada_embs, arc_embs),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        PairDataset(val_df, ada_embs, arc_embs),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model_dir = cfg["model_dir"]
    best_path = model_dir / f"best_{cfg['adapter_type']}_adapter.pt"
    resume_path = model_dir / f"resume_{cfg['adapter_type']}_adapter.pt"

    adapter = get_adapter(cfg, device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    history = []
    best_val = float("inf")
    start_epoch = 1

    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        adapter.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        history = ckpt["history"]
        best_val = ckpt["best_val"]
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        adapter.train()
        train_loss_sum, train_n = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = adapter(x)
            loss, _, _ = total_loss(pred, y)
            loss.backward()
            optimizer.step()
            bsz = x.size(0)
            train_loss_sum += loss.item() * bsz
            train_n += bsz

        adapter.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} val", leave=False):
                x, y = x.to(device), y.to(device)
                pred = adapter(x)
                loss, _, _ = total_loss(pred, y)
                bsz = x.size(0)
                val_loss_sum += loss.item() * bsz
                val_n += bsz

        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_n, 1),
            "val_loss": val_loss_sum / max(val_n, 1),
        }
        history.append(row)
        print(row)

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(
                {"config": cfg, "state_dict": adapter.state_dict(), "history": history},
                best_path,
            )

        torch.save(
            {
                "config": cfg,
                "state_dict": adapter.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "epoch": epoch,
                "best_val": best_val,
            },
            resume_path,
        )

        pd.DataFrame(history).to_csv(cfg["report_dir"] / "training_history.csv", index=False)

    print("Best model saved to:", best_path)
    if resume_path.exists():
        resume_path.unlink()
        print("[done] Resume checkpoint removed.")

    return adapter, train_df, val_df, test_df


def run_training_pipeline(cfg: dict):
    arc_embs, ada_embs, paired_df = collect_and_extract_embeddings(cfg)
    train_df, val_df, test_df = split_identities(paired_df, cfg)
    paired_df.to_csv(cfg["report_dir"] / "paired_metadata_with_splits.csv", index=False)

    adapter, _, _, _ = train_adapter(arc_embs, ada_embs, paired_df, cfg)

    best_path = cfg["model_dir"] / f"best_{cfg['adapter_type']}_adapter.pt"
    checkpoint = torch.load(best_path, map_location=setup_device(cfg))
    adapter.load_state_dict(checkpoint["state_dict"])
    adapter.eval()

    embedding_eval = eval_embedding_alignment(val_df, test_df, ada_embs, arc_embs, adapter, cfg["report_dir"], cfg)
    verification_df = eval_all_verification(val_df, test_df, arc_embs, ada_embs, adapter, cfg["report_dir"], cfg)

    return {
        "arc_embs": arc_embs,
        "ada_embs": ada_embs,
        "paired_df": paired_df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "adapter": adapter,
        "embedding_eval": embedding_eval,
        "verification_df": verification_df,
    }
