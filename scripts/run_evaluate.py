import numpy as np
import pandas as pd
import torch

from emb2face.config import load_config
from emb2face.train import split_identities, get_adapter
from emb2face.evaluate import eval_embedding_alignment, eval_all_verification


def main():
    cfg = load_config("config/default.yaml")
    emb_dir = cfg["emb_dir"]
    report_dir = cfg["report_dir"]

    arc = np.load(emb_dir / "arcface_embeddings.npy")
    ada = np.load(emb_dir / "adaface_embeddings.npy")
    paired = pd.read_csv(report_dir / "paired_metadata.csv")

    train_df, val_df, test_df = split_identities(paired, cfg)

    if cfg["device"] == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif cfg["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    adapter = get_adapter(cfg, device)
    best_path = cfg["model_dir"] / f"best_{cfg['adapter_type']}_adapter.pt"
    ckpt = torch.load(best_path, map_location=device)
    adapter.load_state_dict(ckpt["state_dict"])

    emb_eval = eval_embedding_alignment(val_df, test_df, ada, arc, adapter, report_dir, cfg)
    verif_eval = eval_all_verification(val_df, test_df, arc, ada, adapter, report_dir, cfg)
    print("Embedding eval:")
    print(emb_eval)
    print("Verification eval:")
    print(verif_eval)


if __name__ == "__main__":
    main()
