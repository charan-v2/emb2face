from collections import defaultdict
import random
from typing import Tuple, List

import numpy as np
from sklearn.metrics import roc_curve


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)


def cosine_similarity_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def compute_eer(labels, scores) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thr = float(thresholds[idx])
    return eer, thr


def far_frr_at(labels, scores, threshold: float) -> Tuple[float, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    gen = scores[labels == 1]
    imp = scores[labels == 0]
    frr = float(np.mean(gen < threshold)) if len(gen) else float("nan")
    far = float(np.mean(imp >= threshold)) if len(imp) else float("nan")
    return far, frr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def sample_verification_pairs(df_split, embeddings: np.ndarray, n_pairs: int = 3000, seed: int = 42):
    rng = random.Random(seed)
    by_identity = defaultdict(list)

    for idx, row in df_split.reset_index(drop=True).iterrows():
        by_identity[row["identity"]].append(idx)

    identities = [k for k, v in by_identity.items() if len(v) >= 2]
    if len(identities) < 2:
        return [], []

    genuine_pairs, impostor_pairs = [], []
    target_each = n_pairs // 2

    while len(genuine_pairs) < target_each:
        ident = rng.choice(identities)
        i, j = rng.sample(by_identity[ident], 2)
        genuine_pairs.append((i, j, 1))

    all_ids = list(by_identity.keys())
    while len(impostor_pairs) < target_each:
        id1, id2 = rng.sample(all_ids, 2)
        i = rng.choice(by_identity[id1])
        j = rng.choice(by_identity[id2])
        impostor_pairs.append((i, j, 0))

    pairs = genuine_pairs + impostor_pairs
    rng.shuffle(pairs)

    labels, scores = [], []
    for i, j, y in pairs:
        labels.append(y)
        scores.append(cosine_similarity_np(embeddings[i], embeddings[j]))
    return labels, scores
