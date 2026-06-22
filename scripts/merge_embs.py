from pathlib import Path
import numpy as np
import pandas as pd

base = Path("outputs")

run_a = base / "00_100" / "webface_arcada_adapter"
run_b = base / "101_150" / "webface_arcada_adapter"
merged = base / "webface_arcada_adapter"

out_emb = merged / "embeddings_full"
out_rep = merged / "reports_full"
out_mod = merged / "models_full"

out_emb.mkdir(parents=True, exist_ok=True)
out_rep.mkdir(parents=True, exist_ok=True)
out_mod.mkdir(parents=True, exist_ok=True)

arc_a = np.load(run_a / "embeddings_full" / "arcface_embeddings.npy")
ada_a = np.load(run_a / "embeddings_full" / "adaface_embeddings.npy")
df_a = pd.read_csv(run_a / "reports_full" / "paired_metadata.csv")

arc_b = np.load(run_b / "embeddings_full" / "arcface_embeddings.npy")
ada_b = np.load(run_b / "embeddings_full" / "adaface_embeddings.npy")
df_b = pd.read_csv(run_b / "reports_full" / "paired_metadata.csv")

df_b = df_b.copy()
df_b["arc_index"] += len(arc_a)
df_b["ada_index"] += len(ada_a)

arc = np.concatenate([arc_a, arc_b], axis=0)
ada = np.concatenate([ada_a, ada_b], axis=0)
df = pd.concat([df_a, df_b], ignore_index=True)

np.save(out_emb / "arcface_embeddings.npy", arc)
np.save(out_emb / "adaface_embeddings.npy", ada)
df.to_csv(out_rep / "paired_metadata.csv", index=False)

print("Merged:")
print("  arc:", arc.shape)
print("  ada:", ada.shape)
print("  rows:", len(df))