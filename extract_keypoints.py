from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mediapipe as mp


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_PATH = Path("data") / "hand_landmarks.csv"
DEFAULT_MODEL_PATH = Path("models") / "hand_landmarker.task"
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


@dataclass
class ExtractionRow:
    label: str
    image_path: str
    detected: bool
    handedness: str
    handedness_score: float | None
    landmarks: list[float | None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from the dataset in data/."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory containing one subdirectory per class label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV file to write extracted landmarks to.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MediaPipe hand landmarker .task model.",
    )
    parser.add_argument(
        "--model-url",
        default=DEFAULT_MODEL_URL,
        help="URL used when downloading the default hand landmarker model.",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download the model to --model-path if it does not exist.",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=1,
        help="Maximum number of hands to detect per image.",
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
        "--skip-missing",
        action="store_true",
        help="Skip images where no hand is detected instead of writing empty landmark columns.",
    )
    return parser


def download_model_if_needed(model_path: Path, model_url: str, enabled: bool) -> None:
    if model_path.exists():
        return
    if not enabled:
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Pass --download-model or provide --model-path."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model to {model_path}...", file=sys.stderr)
    urllib.request.urlretrieve(model_url, model_path)


def iter_image_files(data_dir: Path) -> Iterable[Path]:
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def build_header() -> list[str]:
    header = [
        "label",
        "image_path",
        "detected",
        "handedness",
        "handedness_score",
    ]
    for landmark_idx in range(21):
        header.extend(
            [
                f"lm_{landmark_idx:02d}_x",
                f"lm_{landmark_idx:02d}_y",
                f"lm_{landmark_idx:02d}_z",
            ]
        )
    return header


def empty_landmarks() -> list[float | None]:
    return [None] * (21 * 3)


def select_best_hand(result) -> int | None:
    if not result.hand_landmarks:
        return None

    def score_for(index: int) -> float:
        categories = result.handedness[index]
        if not categories:
            return 0.0
        return categories[0].score

    return max(range(len(result.hand_landmarks)), key=score_for)


def extract_row(data_dir: Path, image_path: Path, result) -> ExtractionRow | None:
    label = image_path.parent.name
    relative_path = image_path.relative_to(data_dir).as_posix()
    hand_index = select_best_hand(result)

    if hand_index is None:
        return ExtractionRow(
            label=label,
            image_path=relative_path,
            detected=False,
            handedness="",
            handedness_score=None,
            landmarks=empty_landmarks(),
        )

    handedness_category = result.handedness[hand_index][0] if result.handedness[hand_index] else None
    flattened_landmarks: list[float | None] = []
    for landmark in result.hand_landmarks[hand_index]:
        flattened_landmarks.extend([landmark.x, landmark.y, landmark.z])

    return ExtractionRow(
        label=label,
        image_path=relative_path,
        detected=True,
        handedness=handedness_category.category_name if handedness_category else "",
        handedness_score=handedness_category.score if handedness_category else None,
        landmarks=flattened_landmarks,
    )


def write_rows(rows: Iterable[ExtractionRow], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(build_header())
        for row in rows:
            writer.writerow(
                [
                    row.label,
                    row.image_path,
                    row.detected,
                    row.handedness,
                    row.handedness_score,
                    *row.landmarks,
                ]
            )
            count += 1
    return count


def run(args: argparse.Namespace) -> int:
    data_dir = args.data_dir.resolve()
    output_path = args.output.resolve()
    model_path = args.model_path.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    download_model_if_needed(model_path, args.model_url, args.download_model)

    image_paths = list(iter_image_files(data_dir))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {data_dir}")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    rows: list[ExtractionRow] = []
    detected_count = 0
    with HandLandmarker.create_from_options(options) as landmarker:
        for image_path in image_paths:
            mp_image = mp.Image.create_from_file(str(image_path))
            result = landmarker.detect(mp_image)
            row = extract_row(data_dir, image_path, result)
            if row is None:
                continue
            if row.detected:
                detected_count += 1
            elif args.skip_missing:
                continue
            rows.append(row)

    written_count = write_rows(rows, output_path)
    print(f"Wrote {written_count} rows to {output_path}")
    print(f"Detected at least one hand in {detected_count}/{len(image_paths)} images")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
