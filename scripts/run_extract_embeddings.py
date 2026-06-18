from emb2face.config import load_config
from emb2face.dataset import collect_dataset
from emb2face.embeddings import run_embedding_extraction


def main():
    cfg = load_config("config/default.yaml")
    df = collect_dataset(
        cfg["dataset_root"],
        cfg["image_extensions"],
        max_identities=cfg["max_identities"],
        max_images_per_identity=cfg["max_images_per_identity"],
        min_images_per_identity=cfg["min_images_per_identity"],
    )
    arc, ada, paired = run_embedding_extraction(df, cfg)
    print("Done. Shapes:", arc.shape, ada.shape, "rows:", len(paired))


if __name__ == "__main__":
    main()
