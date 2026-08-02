from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web
from pymongo import MongoClient
from xgboost import XGBClassifier

DEFAULT_CLASSIFIER_FILENAME = "xgboost_hand_classifier.json"
DEFAULT_LABELS_FILENAME = "xgboost_hand_classifier_labels.json"
SERVER_ROOT = Path(__file__).resolve().parent
DEFAULT_SERVER_CLASSIFIER_PATH = SERVER_ROOT / "models" / DEFAULT_CLASSIFIER_FILENAME
DEFAULT_SERVER_LABELS_PATH = SERVER_ROOT / "models" / DEFAULT_LABELS_FILENAME

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
PALM_EDGES = [
    (WRIST, INDEX_MCP),
    (WRIST, MIDDLE_MCP),
    (WRIST, PINKY_MCP),
    (INDEX_MCP, PINKY_MCP),
]
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


def feature_count() -> int:
    normalized_landmark_features = LANDMARK_COUNT * LANDMARK_DIMENSIONS
    fingertip_pair_distances = len(list(combinations(FINGERTIPS, 2)))
    fingertip_wrist_distances = len(FINGERTIPS)
    fingertip_palm_distances = len(FINGERTIPS)
    joint_angles = len(ANGLE_TRIPLETS)
    finger_spreads = len(FINGER_SPREAD_PAIRS)
    global_shape_features = 9
    return (
        normalized_landmark_features
        + fingertip_pair_distances
        + fingertip_wrist_distances
        + fingertip_palm_distances
        + joint_angles
        + finger_spreads
        + global_shape_features
    )


def engineer_features(landmarks: np.ndarray, handedness: str | None = None) -> np.ndarray:
    if landmarks.shape != (LANDMARK_COUNT, LANDMARK_DIMENSIONS):
        raise ValueError(
            f"Expected landmarks shape {(LANDMARK_COUNT, LANDMARK_DIMENSIONS)}, got {landmarks.shape}"
        )

    if not np.isfinite(landmarks).all():
        return np.full(feature_count(), np.nan, dtype=np.float32)

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
        features.append(
            _joint_angle(
                normalized[vertex_a],
                normalized[vertex_b],
                normalized[vertex_c],
            )
        )

    for left_tip, right_tip in FINGER_SPREAD_PAIRS:
        features.append(
            _vector_angle(
                normalized[left_tip] - palm_center,
                normalized[right_tip] - palm_center,
            )
        )

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
    distances = [
        _distance(points[start_idx], points[end_idx])
        for start_idx, end_idx in PALM_EDGES
    ]
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


@dataclass
class Prediction:
    label: str
    confidence: float
    handedness: str


class LandmarkMudraClassifier:
    def __init__(
        self,
        classifier_path: Path,
        labels_path: Path,
        confidence_threshold: float,
    ) -> None:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        self.classes = payload["classes"]
        self.expected_feature_count = len(
            payload.get("feature_names", [])) or feature_count()
        self.confidence_threshold = confidence_threshold

        self.model = XGBClassifier()
        self.model.load_model(classifier_path)

    def predict(self, hands: list[dict]) -> Prediction | None:
        best_hand = self._select_best_hand(hands)
        if best_hand is None:
            return None

        handedness = str(best_hand.get("handedness", "")).strip()
        landmarks = self._hand_landmarks(best_hand)
        if landmarks is None:
            return None

        features = engineer_features(landmarks, handedness)
        if features.shape[0] != self.expected_feature_count:
            raise ValueError(
                "Engineered feature count does not match the classifier metadata: "
                f"{features.shape[0]} != {self.expected_feature_count}"
            )
        probabilities = self.model.predict_proba(
            np.asarray([features], dtype=np.float32)
        )[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        label = self.classes[best_index]
        if confidence < self.confidence_threshold:
            label = "uncertain"
        return Prediction(label=label, confidence=confidence, handedness=handedness)

    @staticmethod
    def _select_best_hand(hands: list[dict]) -> dict | None:
        if not hands:
            return None
        return hands[0]

    @staticmethod
    def _hand_landmarks(hand: dict) -> np.ndarray | None:
        points = hand.get("landmarks", [])
        if len(points) != 21:
            return None

        ordered_points = sorted(
            points,
            key=lambda point: int(point.get("index", 0)),
        )
        if len(ordered_points) != 21:
            return None

        landmarks = []
        for expected_index, point in enumerate(ordered_points):
            if int(point.get("index", expected_index)) != expected_index:
                return None
            landmarks.append(
                [
                    float(point.get("x", 0.0)),
                    float(point.get("y", 0.0)),
                    float(point.get("z", 0.0)),
                ]
            )

        return np.asarray(landmarks, dtype=np.float32)


def display_mudra_name(label: str | None) -> str:
    if not label:
        return "No hand"
    if label == "Pataka":
        return "Pathakam"
    if label == "Hamsasya":
        return "Hamsasyam"
    if label == "Ardhachandra":
        return "Ardhachandram"
    if label == "Chatura":
        return "Chathuram"
    if label == "Katakamukha_1":
        return "Katakaamukham"
    if label == "Ardhapataka":
        return "Ardhapathakam"
    if label == "Kartarimukha":
        return "Kartharimukham"
    if label == "Kapitta":
        return "Kapitham"
    if label == "Shikhara":
        return "Shikaram"
    if label == "Tripataka":
        return "Thripathakam"
    if label == "Mukula":
        return "Mukulam"
    if label == "uncertain":
        return "Uncertain"
    return label.replace("_", " ").title()


def build_response(
    prediction: Prediction | None,
    timestamp: str | None,
) -> str:
    if prediction is None:
        payload = {
            "timestamp": timestamp,
            "label": None,
            "displayLabel": "No hand",
            "confidence": None,
            "handedness": None,
        }
    else:
        payload = {
            "timestamp": timestamp,
            "label": prediction.label,
            "displayLabel": display_mudra_name(prediction.label),
            "confidence": prediction.confidence,
            "handedness": prediction.handedness or None,
        }
    return json.dumps(payload)


async def handle_connection(request: web.Request) -> web.WebSocketResponse:
    websocket = web.WebSocketResponse()
    await websocket.prepare(request)
    classifier: LandmarkMudraClassifier = request.app["classifier"]
    client = request.remote
    print(f"client connected: {client}")

    try:
        async for websocket_message in websocket:
            if websocket_message.type != WSMsgType.TEXT:
                continue
            payload = json.loads(websocket_message.data)
            hands = payload.get("hands", [])
            timestamp = payload.get("timestamp")
            prediction = classifier.predict(hands)
            await websocket.send_str(build_response(prediction, timestamp))
    except json.JSONDecodeError as error:
        print(f"received invalid JSON: {error}")
    except ValueError as error:
        print(f"invalid input payload: {error}")
    finally:
        print(f"client disconnected: {client}")
    return websocket


def cors_headers(origin: str | None, allowed_origins: list[str]) -> dict[str, str]:
    allowed_origin = "*" if "*" in allowed_origins else origin if origin in allowed_origins else None
    headers = {
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, X-Story-Pin",
    }
    if allowed_origin:
        headers["Access-Control-Allow-Origin"] = allowed_origin
        if allowed_origin != "*":
            headers["Vary"] = "Origin"
    return headers


def story_document(payload: dict) -> dict:
    story = str(payload.get("story", "")).strip()
    if not story or len(story) > 5000:
        raise ValueError("Story must be between 1 and 5000 characters.")
    rasa = str(payload.get("rasa", "")).strip()[:100] or None
    week = payload.get("week")
    if week is not None and (not isinstance(week, int) or week < 1):
        raise ValueError("Week must be a positive integer.")
    return {"story": story, "rasa": rasa, "week": week}


def serialise_story(story: dict) -> dict:
    return {
        "id": str(story["_id"]),
        "story": story["story"],
        "rasa": story.get("rasa"),
        "week": story.get("week"),
        "createdAt": story["createdAt"].isoformat(),
    }


async def handle_api(request: web.Request) -> web.Response:
    headers = cors_headers(request.headers.get("Origin"), request.app["allowed_origins"])
    if request.method == "OPTIONS":
        return web.Response(status=204, headers=headers)
    stories = request.app["stories"]
    story_pin = request.app["story_pin"]
    if request.method == "GET":
        if request.headers.get("X-Story-Pin") != story_pin:
            return web.json_response({"error": "Teacher PIN required."}, status=401, headers=headers)
        return web.json_response({"stories": [
            serialise_story(story) for story in stories.find().sort("createdAt", -1)
        ]}, headers=headers)
    if request.method != "POST":
        return web.json_response({"error": "Method not allowed."}, status=405, headers=headers)
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Request body must be an object.")
        document = story_document(payload)
        document["createdAt"] = datetime.now(timezone.utc)
        document["_id"] = stories.insert_one(document).inserted_id
        return web.json_response({"story": serialise_story(document)}, status=201, headers=headers)
    except (json.JSONDecodeError, ValueError) as error:
        return web.json_response({"error": str(error)}, status=400, headers=headers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the websocket landmark server with XGBoost mudra classification."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    parser.add_argument(
        "--classifier-path",
        type=Path,
        default=DEFAULT_SERVER_CLASSIFIER_PATH,
        help="Path to the trained XGBoost classifier JSON.",
    )
    parser.add_argument(
        "--mongodb-uri",
        default=os.environ.get("MONGODB_URI"),
        help="MongoDB connection string; defaults to MONGODB_URI.",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=["https://natyalab.com"],
        help="Allowed frontend origin; repeat for more than one.",
    )
    parser.add_argument(
        "--story-portal-pin",
        default=os.environ.get("STORY_PORTAL_PIN", "0000"),
        help="Teacher PIN for reading stories; defaults to STORY_PORTAL_PIN or 0000.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=DEFAULT_SERVER_LABELS_PATH,
        help="Path to the label metadata JSON.",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
        help="Minimum classifier probability before a hasta label is shown.",
    )
    return parser


def validate_paths(classifier_path: Path, labels_path: Path) -> None:
    missing_paths = [path for path in (
        classifier_path, labels_path) if not path.is_file()]
    if not missing_paths:
        return

    missing_text = ", ".join(str(path) for path in missing_paths)
    raise FileNotFoundError(
        "Missing required model assets. Bundle the classifier and labels JSON next to "
        f"the server under `server/models/` or pass explicit paths. Missing: {missing_text}"
    )


async def main() -> None:
    args = build_parser().parse_args()
    validate_paths(args.classifier_path, args.labels_path)
    if not args.mongodb_uri:
        raise ValueError("Set MONGODB_URI before starting the story API.")
    mongo_client = MongoClient(args.mongodb_uri)
    stories = mongo_client["natya_lab"]["stories"]
    stories.create_index("createdAt")
    classifier = LandmarkMudraClassifier(
        classifier_path=args.classifier_path,
        labels_path=args.labels_path,
        confidence_threshold=args.classification_threshold,
    )
    app = web.Application()
    app["classifier"] = classifier
    app["stories"] = stories
    app["allowed_origins"] = args.cors_origin
    app["story_pin"] = args.story_portal_pin
    app.router.add_get("/", handle_connection)
    app.router.add_route("*", "/api/stories", handle_api)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    print(f"server listening on ws://{args.host}:{args.port} and http://{args.host}:{args.port}/api/stories")
    try:
        await asyncio.Future()
    finally:
        mongo_client.close()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
