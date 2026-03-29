from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np


WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

LANDMARK_COUNT = 21
LANDMARK_DIMENSIONS = 3
EPSILON = 1e-6

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
PALM_POINTS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
PALM_EDGES = [(WRIST, INDEX_MCP), (WRIST, MIDDLE_MCP), (WRIST, PINKY_MCP), (INDEX_MCP, PINKY_MCP)]
ANGLE_TRIPLETS = [
    (WRIST, THUMB_CMC, THUMB_MCP),
    (THUMB_CMC, THUMB_MCP, THUMB_IP),
    (THUMB_MCP, THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP, INDEX_PIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP),
    (INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (WRIST, MIDDLE_MCP, MIDDLE_PIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (WRIST, RING_MCP, RING_PIP),
    (RING_MCP, RING_PIP, RING_DIP),
    (RING_PIP, RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP, PINKY_PIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP),
    (PINKY_PIP, PINKY_DIP, PINKY_TIP),
]
FINGER_SPREAD_PAIRS = [
    (THUMB_TIP, INDEX_TIP),
    (INDEX_TIP, MIDDLE_TIP),
    (MIDDLE_TIP, RING_TIP),
    (RING_TIP, PINKY_TIP),
]
CANONICAL_COLUMNS = [
    f"lm_{landmark_idx:02d}_{axis}"
    for landmark_idx in range(LANDMARK_COUNT)
    for axis in ("x", "y", "z")
]


def image_landmark_columns() -> list[str]:
    return CANONICAL_COLUMNS.copy()


def feature_names() -> list[str]:
    names: list[str] = []
    for landmark_idx in range(LANDMARK_COUNT):
        for axis in ("x", "y", "z"):
            names.append(f"norm_lm_{landmark_idx:02d}_{axis}")

    for start_idx, end_idx in combinations(FINGERTIPS, 2):
        names.append(f"dist_tip_{start_idx:02d}_{end_idx:02d}")

    for fingertip_idx in FINGERTIPS:
        names.append(f"dist_tip_{fingertip_idx:02d}_wrist")

    for fingertip_idx in FINGERTIPS:
        names.append(f"dist_tip_{fingertip_idx:02d}_palm_center")

    for vertex_a, vertex_b, vertex_c in ANGLE_TRIPLETS:
        names.append(f"angle_{vertex_a:02d}_{vertex_b:02d}_{vertex_c:02d}")

    for left_tip, right_tip in FINGER_SPREAD_PAIRS:
        names.append(f"spread_angle_{left_tip:02d}_{right_tip:02d}")

    names.extend(
        [
            "palm_width",
            "palm_height",
            "palm_depth",
            "tip_span_x",
            "tip_span_y",
            "tip_span_z",
            "tip_polygon_area_xy",
            "mean_tip_radius",
            "std_tip_radius",
        ]
    )
    return names


def row_to_landmarks(row: dict[str, str]) -> np.ndarray:
    values: list[float] = []
    for column in CANONICAL_COLUMNS:
        raw_value = row.get(column, "")
        if raw_value in ("", None):
            values.append(np.nan)
        else:
            values.append(float(raw_value))
    return np.asarray(values, dtype=np.float32).reshape(LANDMARK_COUNT, LANDMARK_DIMENSIONS)


def landmarks_from_mediapipe(hand_landmarks: Iterable[object]) -> np.ndarray:
    values: list[list[float]] = []
    for landmark in hand_landmarks:
        values.append([float(landmark.x), float(landmark.y), float(landmark.z)])
    return np.asarray(values, dtype=np.float32)


def engineer_features(landmarks: np.ndarray, handedness: str | None = None) -> np.ndarray:
    if landmarks.shape != (LANDMARK_COUNT, LANDMARK_DIMENSIONS):
        raise ValueError(
            f"Expected landmarks shape {(LANDMARK_COUNT, LANDMARK_DIMENSIONS)}, got {landmarks.shape}"
        )

    if not np.isfinite(landmarks).all():
        return np.full(len(feature_names()), np.nan, dtype=np.float32)

    canonical = landmarks.astype(np.float32, copy=True)
    if (handedness or "").strip().lower() == "left":
        canonical[:, 0] *= -1.0

    canonical -= canonical[WRIST]
    scale = _hand_scale(canonical)
    normalized = canonical / max(scale, EPSILON)

    palm_center = np.mean(normalized[PALM_POINTS], axis=0)
    tip_points = normalized[FINGERTIPS]
    tip_radii = np.linalg.norm(tip_points - palm_center, axis=1)

    features: list[float] = normalized.reshape(-1).tolist()

    for start_idx, end_idx in combinations(FINGERTIPS, 2):
        features.append(_distance(normalized[start_idx], normalized[end_idx]))

    for fingertip_idx in FINGERTIPS:
        features.append(_distance(normalized[fingertip_idx], normalized[WRIST]))

    for fingertip_idx in FINGERTIPS:
        features.append(_distance(normalized[fingertip_idx], palm_center))

    for vertex_a, vertex_b, vertex_c in ANGLE_TRIPLETS:
        features.append(_joint_angle(normalized[vertex_a], normalized[vertex_b], normalized[vertex_c]))

    for left_tip, right_tip in FINGER_SPREAD_PAIRS:
        features.append(_vector_angle(normalized[left_tip] - palm_center, normalized[right_tip] - palm_center))

    palm_box = np.ptp(normalized[PALM_POINTS], axis=0)
    tip_box = np.ptp(tip_points, axis=0)
    features.extend(
        [
            float(palm_box[0]),
            float(palm_box[1]),
            float(palm_box[2]),
            float(tip_box[0]),
            float(tip_box[1]),
            float(tip_box[2]),
            _polygon_area_xy(tip_points),
            float(np.mean(tip_radii)),
            float(np.std(tip_radii)),
        ]
    )

    return np.asarray(features, dtype=np.float32)


def _hand_scale(points: np.ndarray) -> float:
    distances = [_distance(points[start_idx], points[end_idx]) for start_idx, end_idx in PALM_EDGES]
    return float(np.mean(distances))


def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def _joint_angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    return _vector_angle(point_a - point_b, point_c - point_b)


def _vector_angle(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_product = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if norm_product < EPSILON:
        return 0.0
    cosine = float(np.clip(np.dot(vector_a, vector_b) / norm_product, -1.0, 1.0))
    return float(np.arccos(cosine))


def _polygon_area_xy(points: np.ndarray) -> float:
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    shifted_x = np.roll(x_coords, -1)
    shifted_y = np.roll(y_coords, -1)
    area = 0.5 * abs(np.dot(x_coords, shifted_y) - np.dot(y_coords, shifted_x))
    return float(area)
