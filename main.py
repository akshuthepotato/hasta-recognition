from __future__ import annotations

import argparse
import json
import queue
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from xgboost import XGBClassifier

from geometry_features import engineer_features, landmarks_from_mediapipe


DEFAULT_MODEL_PATH = Path("models") / "hand_landmarker.task"
DEFAULT_CLASSIFIER_PATH = Path("models") / "xgboost_hand_classifier.json"
DEFAULT_LABELS_PATH = Path("models") / "xgboost_hand_classifier_labels.json"


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass
class Prediction:
    label: str
    confidence: float
    handedness: str
    handedness_score: float


@dataclass
class FrameResult:
    timestamp_ms: int
    result: HandLandmarkerResult
    prediction: Prediction | None


class LiveMudraClassifier:
    def __init__(
        self,
        classifier_path: Path,
        labels_path: Path,
        confidence_threshold: float,
    ) -> None:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        self.classes = payload["classes"]
        self.feature_names = payload["feature_names"]
        self.expected_feature_count = len(self.feature_names)
        self.confidence_threshold = confidence_threshold

        self.model = XGBClassifier()
        self.model.load_model(classifier_path)

    def predict(self, result: HandLandmarkerResult) -> Prediction | None:
        hand_index = self._select_best_hand(result)
        if hand_index is None:
            return None

        handedness_category = result.handedness[hand_index][0] if result.handedness[hand_index] else None
        handedness = handedness_category.category_name if handedness_category else ""
        features = engineer_features(
            landmarks_from_mediapipe(result.hand_landmarks[hand_index]),
            handedness,
        )
        if len(features) != self.expected_feature_count:
            raise ValueError(
                f"Expected {self.expected_feature_count} features, got {len(features)}"
            )

        probabilities = self.model.predict_proba(
            np.asarray([features], dtype=np.float32)
        )[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])

        label = self.classes[best_index]
        if confidence < self.confidence_threshold:
            label = "uncertain"

        return Prediction(
            label=label,
            confidence=confidence,
            handedness=handedness_category.category_name if handedness_category else "",
            handedness_score=handedness_category.score if handedness_category else 0.0,
        )

    @staticmethod
    def _select_best_hand(result: HandLandmarkerResult) -> int | None:
        if not result.hand_landmarks:
            return None

        def score_for(index: int) -> float:
            categories = result.handedness[index]
            return categories[0].score if categories else 0.0

        return max(range(len(result.hand_landmarks)), key=score_for)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run live MediaPipe hand tracking and classify mudras from a webcam feed."
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="OpenCV webcam device index.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MediaPipe hand landmarker .task file.",
    )
    parser.add_argument(
        "--classifier-path",
        type=Path,
        default=DEFAULT_CLASSIFIER_PATH,
        help="Path to the trained XGBoost classifier JSON.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="Path to the label metadata JSON.",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=1,
        help="Maximum number of hands to track.",
    )
    parser.add_argument(
        "--min-hand-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum palm detection confidence.",
    )
    parser.add_argument(
        "--min-hand-presence-confidence",
        type=float,
        default=0.5,
        help="Minimum hand presence confidence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum tracking confidence.",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
        help="Minimum classifier probability before a mudra label is shown.",
    )
    return parser


def draw_hand_landmarks(frame: np.ndarray, result: HandLandmarkerResult) -> None:
    height, width = frame.shape[:2]
    for hand_landmarks in result.hand_landmarks:
        points: list[tuple[int, int]] = []
        for landmark in hand_landmarks:
            x = min(max(int(landmark.x * width), 0), width - 1)
            y = min(max(int(landmark.y * height), 0), height - 1)
            points.append((x, y))

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (255, 200, 0), 2)

        for point in points:
            cv2.circle(frame, point, 4, (0, 140, 255), -1)


def draw_overlay(
    frame: np.ndarray,
    prediction: Prediction | None,
    fps: float,
) -> None:
    lines = [f"FPS: {fps:.1f}"]
    if prediction is None:
        lines.append("Mudra: no hand detected")
    else:
        lines.append(f"Mudra: {prediction.label} ({prediction.confidence:.2f})")
        if prediction.handedness:
            lines.append(
                f"Hand: {prediction.handedness} ({prediction.handedness_score:.2f})"
            )

    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30


def build_landmarker(
    args: argparse.Namespace,
    result_queue: queue.Queue[FrameResult],
    classifier: LiveMudraClassifier,
) -> HandLandmarker:
    def print_result(
        result: HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        del output_image
        prediction = classifier.predict(result)
        frame_result = FrameResult(
            timestamp_ms=timestamp_ms,
            result=result,
            prediction=prediction,
        )
        try:
            result_queue.put_nowait(frame_result)
        except queue.Full:
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
            result_queue.put_nowait(frame_result)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.model_path.resolve())),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        result_callback=print_result,
    )
    return HandLandmarker.create_from_options(options)


def validate_paths(*paths: Path) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path.resolve()}")


def run(args: argparse.Namespace) -> int:
    validate_paths(args.model_path, args.classifier_path, args.labels_path)

    classifier = LiveMudraClassifier(
        classifier_path=args.classifier_path.resolve(),
        labels_path=args.labels_path.resolve(),
        confidence_threshold=args.classification_threshold,
    )
    result_queue: queue.Queue[FrameResult] = queue.Queue(maxsize=1)

    capture = cv2.VideoCapture(args.camera_id)
    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open webcam {args.camera_id}. Check camera permissions and device index."
        )

    latest_result: FrameResult | None = None
    previous_frame_time = time.perf_counter()

    with build_landmarker(args, result_queue, classifier) as landmarker:
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Failed to read a frame from the webcam.")

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.time() * 1000)
                landmarker.detect_async(mp_image, timestamp_ms)

                while True:
                    try:
                        candidate = result_queue.get_nowait()
                    except queue.Empty:
                        break
                    if latest_result is None or candidate.timestamp_ms >= latest_result.timestamp_ms:
                        latest_result = candidate

                now = time.perf_counter()
                fps = 1.0 / max(now - previous_frame_time, 1e-6)
                previous_frame_time = now

                if latest_result is not None:
                    draw_hand_landmarks(frame, latest_result.result)
                    draw_overlay(frame, latest_result.prediction, fps)
                else:
                    draw_overlay(frame, None, fps)

                cv2.imshow("Hasta Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
