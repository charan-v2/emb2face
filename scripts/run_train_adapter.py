import numpy as np
import pandas as pd

from emb2face.config import load_config
from emb2face.train import train_adapter


def main():
    cfg = load_config("config/default.yaml")
    emb_dir = cfg["emb_dir"]
    report_dir = cfg["report_dir"]

    arc = np.load(emb_dir / "arcface_embeddings.npy")
    ada = np.load(emb_dir / "adaface_embeddings.npy")
    paired = pd.read_csv(report_dir / "paired_metadata.csv")

    adapter, train_df, val_df, test_df = train_adapter(arc, ada, paired, cfg)
    print("Training complete.")


if __name__ == "__main__":
    main()
