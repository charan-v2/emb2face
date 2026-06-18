from pathlib import Path
from typing import Iterable, List, Dict

import pandas as pd
from tqdm.auto import tqdm


def collect_dataset(
    root_dir: Path,
    exts: Iterable[str],
    max_identities=None,
    max_images_per_identity=None,
    min_images_per_identity: int = 2,
) -> pd.DataFrame:
    rows: List[Dict] = []
    identity_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    if max_identities is not None:
        identity_dirs = identity_dirs[:max_identities]

    for ident_dir in tqdm(identity_dirs, desc="Scanning identities"):
        image_paths = sorted(
            [p for p in ident_dir.rglob("*") if p.suffix.lower() in exts]
        )
        if max_images_per_identity is not None:
            image_paths = image_paths[:max_images_per_identity]
        if len(image_paths) < min_images_per_identity:
            continue

        for p in image_paths:
            rows.append(
                {
                    "identity": ident_dir.name,
                    "image_path": str(p),
                    "relative_path": str(p.relative_to(root_dir)),
                }
            )

    return pd.DataFrame(rows)
