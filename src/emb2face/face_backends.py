from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .embeddings import get_rec_model, load_arcface_app, l2_normalize


_BACKEND_ALIASES = {
    "insightface": "insightface",
    "faceanalysis": "insightface",
    "arcface": "insightface",
    "retinaface": "retinaface",
    "uniface": "uniface",
    "faceanalyzer": "uniface",
    "face_analyzer": "uniface",
}

_INSIGHTFACE_CACHE: dict[tuple[str, tuple[int, int]], tuple[Any, Any]] = {}
_UNIFACE_DETECTOR_CACHE: Any | None = None
_UNIFACE_ANALYZER_CACHE: Any | None = None


def normalize_backend_name(name: str) -> str:
    normalized = _BACKEND_ALIASES.get(str(name).strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported face backend: {name}")
    return normalized


def _face_bbox(face: Any) -> np.ndarray | None:
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return None
    arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return None
    return arr[:4]


def _face_landmarks(face: Any) -> np.ndarray | None:
    for attr in ("kps", "landmarks"):
        landmarks = getattr(face, attr, None)
        if landmarks is not None:
            arr = np.asarray(landmarks, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= 5:
                return arr[:5, :2]
    return None


def face_landmarks(face: Any) -> np.ndarray | None:
    return _face_landmarks(face)


def _face_area(face: Any) -> float:
    bbox = _face_bbox(face)
    if bbox is None:
        return 0.0
    return float(max(bbox[2] - bbox[0], 0.0) * max(bbox[3] - bbox[1], 0.0))


def select_largest_face(faces: list[Any], require_single_face: bool = False) -> Any:
    if not faces:
        raise ValueError("No faces were detected")
    if require_single_face and len(faces) != 1:
        raise ValueError(f"Expected exactly one face but detected {len(faces)}")
    return max(faces, key=_face_area)


def align_face_from_landmarks(img_bgr: np.ndarray, landmarks: np.ndarray | None, image_size: int = 112) -> np.ndarray | None:
    if landmarks is None:
        return None
    from insightface.utils import face_align

    return face_align.norm_crop(img_bgr, landmark=landmarks, image_size=image_size)


def estimate_yaw_degrees(landmarks: np.ndarray | None, image_shape: tuple[int, int, int] | tuple[int, int]) -> float | None:
    if landmarks is None:
        return None

    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 5 or arr.shape[1] < 2:
        return None

    image_points = arr[:5, :2].astype(np.float32)
    model_points = np.array(
        [
            (-30.0, 30.0, 30.0),
            (30.0, 30.0, 30.0),
            (0.0, 0.0, 0.0),
            (-25.0, -30.0, 30.0),
            (25.0, -30.0, 30.0),
        ],
        dtype=np.float32,
    )

    height, width = image_shape[:2]
    focal_length = float(max(width, height))
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    try:
        success, rvec, tvec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None

    if not success:
        return None

    try:
        rotation_mat, _ = cv2.Rodrigues(rvec)
        pose_mat = cv2.hconcat((rotation_mat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    except cv2.error:
        return None

    yaw = float(euler_angles[1, 0])
    if not math.isfinite(yaw):
        return None
    return yaw


def estimate_face_yaw_degrees(face: Any, image_shape: tuple[int, int, int] | tuple[int, int]) -> float | None:
    return estimate_yaw_degrees(face_landmarks(face), image_shape)


@dataclass
class ExtractedFace:
    embedding: np.ndarray
    crop_rgb: np.ndarray
    face_count: int
    bbox: np.ndarray | None
    landmarks: np.ndarray | None
    confidence: float | None


class InsightFaceBackend:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        cache_key = (
            str(Path(cfg["insight_root"]).expanduser()),
            tuple(cfg.get("det_size", (640, 640))),
        )
        if cache_key not in _INSIGHTFACE_CACHE:
            app, _ = load_arcface_app(cfg)
            _INSIGHTFACE_CACHE[cache_key] = (app, get_rec_model(app))
        self.app, self.rec_model = _INSIGHTFACE_CACHE[cache_key]

    def detect(self, img_bgr: np.ndarray):
        return list(self.app.get(img_bgr))

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        feat = self.rec_model.get_feat([crop_bgr])[0]
        return l2_normalize(np.asarray(feat, dtype=np.float32)).astype(np.float32)


class UniFaceDetectorBackend:
    def __init__(self):
        global _UNIFACE_DETECTOR_CACHE
        if _UNIFACE_DETECTOR_CACHE is not None:
            self.detector = _UNIFACE_DETECTOR_CACHE
            return
        try:
            from uniface.detection import RetinaFace
        except ImportError as exc:
            raise ImportError(
                "UniFace is not installed. Install it with `pip install uniface[cpu]` or `pip install uniface[gpu]`."
            ) from exc

        _UNIFACE_DETECTOR_CACHE = RetinaFace()
        self.detector = _UNIFACE_DETECTOR_CACHE

    def detect(self, img_bgr: np.ndarray):
        return list(self.detector.detect(img_bgr))

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        return None


class UniFaceAnalyzerBackend:
    def __init__(self):
        global _UNIFACE_ANALYZER_CACHE
        if _UNIFACE_ANALYZER_CACHE is not None:
            self.analyzer = _UNIFACE_ANALYZER_CACHE
            return
        try:
            from uniface import FaceAnalyzer
        except ImportError as exc:
            raise ImportError(
                "UniFace is not installed. Install it with `pip install uniface[cpu]` or `pip install uniface[gpu]`."
            ) from exc

        _UNIFACE_ANALYZER_CACHE = FaceAnalyzer()
        self.analyzer = _UNIFACE_ANALYZER_CACHE

    def detect(self, img_bgr: np.ndarray):
        return list(self.analyzer.analyze(img_bgr))

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        faces = list(self.analyzer.analyze(crop_bgr))
        if not faces:
            return None
        face = select_largest_face(faces)
        emb = getattr(face, "embedding", None)
        if emb is None:
            return None
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if emb.size == 0:
            return None
        return l2_normalize(emb).astype(np.float32)


def build_face_detector(cfg: dict):
    backend = normalize_backend_name(cfg.get("score_detector_backend", "insightface"))
    if backend == "insightface":
        return InsightFaceBackend(cfg)
    if backend == "retinaface":
        return UniFaceDetectorBackend()
    if backend == "uniface":
        return UniFaceAnalyzerBackend()
    raise ValueError(f"Unsupported detector backend: {backend}")


def build_face_embedder(cfg: dict):
    backend = normalize_backend_name(cfg.get("score_embedder_backend", "insightface"))
    if backend == "insightface":
        return InsightFaceBackend(cfg)
    if backend == "uniface":
        return UniFaceAnalyzerBackend()
    raise ValueError(f"Unsupported embedder backend: {backend}")


def extract_face_embedding(
    image_path: str | Path,
    detector,
    embedder,
    require_single_face: bool = False,
    image_size: int = 112,
):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None

    faces = detector.detect(img_bgr)
    if not faces:
        return None

    face = select_largest_face(faces, require_single_face=require_single_face)
    landmarks = _face_landmarks(face)
    crop_bgr = align_face_from_landmarks(img_bgr, landmarks, image_size=image_size)
    if crop_bgr is None:
        return None

    embedding = embedder.embed(crop_bgr)
    if embedding is None:
        return None

    bbox = _face_bbox(face)
    confidence = getattr(face, "confidence", None)
    if confidence is None:
        confidence = getattr(face, "det_score", None)
    if confidence is not None:
        confidence = float(confidence)
    if confidence is not None and np.isnan(confidence):
        confidence = None

    return ExtractedFace(
        embedding=np.asarray(embedding, dtype=np.float32).reshape(-1),
        crop_rgb=cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
        face_count=len(faces),
        bbox=bbox,
        landmarks=landmarks,
        confidence=confidence,
    )
