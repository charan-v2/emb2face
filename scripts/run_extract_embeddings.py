from emb2face.config import load_config
from emb2face.embeddings import collect_and_extract_embeddings


def main():
    cfg = load_config("config/default.yaml")
    arc, ada, paired = collect_and_extract_embeddings(cfg)
    print("Done. Shapes:", arc.shape, ada.shape, "rows:", len(paired))


if __name__ == "__main__":
    main()
