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


class HoldStateMachine:
    def __init__(
        self,
        hold_duration: float,
        max_gap: float = 0.6,       # tolerate missing label for this long
        max_mismatch: float = 0.6,  # tolerate wrong label for this long
    ):
        self.hold_duration = hold_duration
        self.max_gap = max_gap
        self.max_mismatch = max_mismatch

        self.hold_label = None
        self.hold_start_time = None
        self.paused = False

        # New tracking
        self.last_seen_time = None
        self.mismatch_start_time = None

    def reset(self):
        self.hold_label = None
        self.hold_start_time = None
        self.paused = False
        self.last_seen_time = None
        self.mismatch_start_time = None

    def update(self, label: str | None):
        now = time.perf_counter()
        progress = 0.0

        valid = label and label != "uncertain"

        # --- CASE 1: no active hold yet ---
        if self.hold_label is None:
            if valid:
                self.hold_label = label
                self.hold_start_time = now
                self.last_seen_time = now
            return 0.0, False, None

        # --- CASE 2: we are tracking a label ---
        if valid and label == self.hold_label:
            # Correct label → continue
            self.last_seen_time = now
            self.mismatch_start_time = None

        elif valid and label != self.hold_label:
            # Wrong label → start mismatch timer
            if self.mismatch_start_time is None:
                self.mismatch_start_time = now

            if now - self.mismatch_start_time > self.max_mismatch:
                # Too long mismatch → reset to new label
                self.hold_label = label
                self.hold_start_time = now
                self.last_seen_time = now
                self.mismatch_start_time = None
                self.paused = False
                return 0.0, False, None

        else:
            # Missing / uncertain label
            if self.last_seen_time is not None:
                if now - self.last_seen_time > self.max_gap:
                    # Too long gap → reset
                    self.reset()
                    return 0.0, False, None

        # --- Compute progress ---
        if self.hold_start_time is not None:
            elapsed = now - self.hold_start_time
            progress = min(elapsed / self.hold_duration, 1.0)

            if elapsed >= self.hold_duration:
                self.paused = True

        return progress, self.paused, self.hold_label


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


# ---------------- NEW: PROGRESS CIRCLE ---------------- #

def draw_progress_circle(frame, result, progress, paused):
    if not result.hand_landmarks:
        return

    height, width = frame.shape[:2]

    # Border inset from edges
    margin = 10
    thickness = 6

    # Rectangle corners
    top_left = (margin, margin)
    top_right = (width - margin, margin)
    bottom_right = (width - margin, height - margin)
    bottom_left = (margin, height - margin)

    color_active = (0, 255, 255)
    color_paused = (0, 255, 0)

    color = color_paused if paused else color_active

    # Total perimeter
    top_len = top_right[0] - top_left[0]
    right_len = bottom_right[1] - top_right[1]
    bottom_len = bottom_right[0] - bottom_left[0]
    left_len = bottom_left[1] - top_left[1]

    perimeter = top_len + right_len + bottom_len + left_len

    # Length to draw
    draw_len = perimeter if paused else int(progress * perimeter)

    def draw_segment(p1, p2, length_remaining):
        segment_len = int(((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) ** 0.5)

        if length_remaining <= 0:
            return length_remaining

        if length_remaining >= segment_len:
            cv2.line(frame, p1, p2, color, thickness)
            return length_remaining - segment_len
        else:
            # Partial segment
            ratio = length_remaining / segment_len
            x = int(p1[0] + (p2[0] - p1[0]) * ratio)
            y = int(p1[1] + (p2[1] - p1[1]) * ratio)
            cv2.line(frame, p1, (x, y), color, thickness)
            return 0

    remaining = draw_len

    # Draw in clockwise order
    remaining = draw_segment(top_left, top_right, remaining)
    remaining = draw_segment(top_right, bottom_right, remaining)
    remaining = draw_segment(bottom_right, bottom_left, remaining)
    remaining = draw_segment(bottom_left, top_left, remaining)


# ---------------- DRAW ---------------- #

def draw_hand_landmarks(frame: np.ndarray, result: HandLandmarkerResult) -> None:
    height, width = frame.shape[:2]
    for hand_landmarks in result.hand_landmarks:
        points = []
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (255, 200, 0), 2)

        for point in points:
            cv2.circle(frame, point, 4, (0, 140, 255), -1)


def draw_overlay(frame, prediction, fps):
    def display_hasta_name(label: str) -> str:
        return "Pathaakam" if label == "Pataka" else label.replace("_", " ").title()

    lines = [f"FPS: {fps:.1f}"]

    if prediction is None:
        lines.append("Hasta: no hand")
    else:
        lines.append(
            f"Hasta: {display_hasta_name(prediction.label)} ({prediction.confidence:.2f})"
        )

    y = 30
    for line in lines:
        cv2.putText(frame, line, (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30


# ---------------- MAIN ---------------- #

def run(args):
    classifier = LiveMudraClassifier(
        args.classifier_path,
        args.labels_path,
        args.classification_threshold,
    )

    result_queue = queue.Queue(maxsize=1)
    capture = cv2.VideoCapture(args.camera_id)

    HOLD_DURATION = 5
    hold_sm = HoldStateMachine(HOLD_DURATION)

    latest_result = None
    prev_time = time.perf_counter()

    def callback(result, output_image, timestamp_ms):
        prediction = classifier.predict(result)
        result_queue.put(FrameResult(timestamp_ms, result, prediction))

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.model_path)),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=callback,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, int(time.time() * 1000))

            while not result_queue.empty():
                latest_result = result_queue.get()

            now = time.perf_counter()
            fps = 1 / (now - prev_time)
            prev_time = now

            progress = 0.0
            paused = False

            if latest_result and latest_result.prediction:
                label = latest_result.prediction.label

                progress, paused = hold_sm.update(label)

                draw_hand_landmarks(frame, latest_result.result)
                draw_progress_circle(frame, latest_result.result, progress, paused)
                draw_overlay(frame, latest_result.prediction, fps)
            else:
                hold_sm.reset()
                draw_overlay(frame, None, fps)

            cv2.imshow("Hasta Detection", frame)

            if paused:
                cv2.waitKey(0)

            key = cv2.waitKey(1)
            if key in (27, ord('q')):
                print("quitting")
                break
        print("closing landmarker")

    print("releasing")
    capture.release()
    print("released")
    cv2.destroyAllWindows()
    print("destroying")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run live MediaPipe hand tracking and classify hastas from a webcam feed."
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
        help="Minimum classifier probability before a hasta label is shown.",
    )
    return parser


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
