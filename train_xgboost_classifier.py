from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from geometry_features import engineer_features, feature_names, row_to_landmarks

DEFAULT_KEYPOINTS_CSV = Path("data") / "hand_landmarks.csv"
DEFAULT_MODEL_PATH = Path("models") / "xgboost_hand_classifier.json"
DEFAULT_LABELS_PATH = Path("models") / "xgboost_hand_classifier_labels.json"
DEFAULT_AUGMENT_COPIES = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost classifier on extracted hand landmark keypoints."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_KEYPOINTS_CSV,
        help="CSV produced by extract_keypoints.py.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to save the trained XGBoost model JSON.",
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="Path to save class label metadata JSON.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split and model.",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Keep rows where no hand was detected. By default they are dropped.",
    )
    parser.add_argument(
        "--augment-copies",
        type=int,
        default=DEFAULT_AUGMENT_COPIES,
        help="Number of augmented samples to create per training sample.",
    )
    parser.add_argument(
        "--rotation-degrees",
        type=float,
        default=20.0,
        help="Maximum absolute rotation applied about each axis during augmentation.",
    )
    parser.add_argument(
        "--translation-range",
        type=float,
        default=0.03,
        help="Maximum absolute translation applied along each axis during augmentation.",
    )
    parser.add_argument(
        "--scale-range",
        type=float,
        default=0.08,
        help="Maximum per-axis scaling delta during augmentation. Actual factors are sampled from [1-range, 1+range].",
    )
    parser.add_argument(
        "--jitter-std",
        type=float,
        default=0.01,
        help="Standard deviation of Gaussian landmark jitter added during augmentation.",
    )
    return parser


def load_dataset(
    csv_path: Path,
    include_missing: bool,
) -> tuple[np.ndarray, np.ndarray, list[str | None], list[str]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    handedness_values: list[str | None] = []
    engineered_feature_names = feature_names()

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}")

        if not any(name.startswith("lm_") for name in reader.fieldnames):
            raise ValueError(f"No landmark columns found in {csv_path}")

        for row in reader:
            detected = row.get("detected", "").strip().lower() == "true"
            if not include_missing and not detected:
                continue

            labels.append(row["label"])
            handedness = row.get("handedness")
            landmarks = row_to_landmarks(row)
            handedness_values.append(handedness)
            rows.append(landmarks)

    if not rows:
        raise ValueError("No training rows available after filtering the dataset.")

    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels),
        handedness_values,
        engineered_feature_names,
    )


def save_label_metadata(
    output_path: Path,
    label_encoder: LabelEncoder,
    feature_names: list[str],
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classes": label_encoder.classes_.tolist(),
        "feature_names": feature_names,
        "input_csv": str(args.input.resolve()),
        "include_missing": args.include_missing,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "augment_copies": args.augment_copies,
        "rotation_degrees": args.rotation_degrees,
        "translation_range": args.translation_range,
        "scale_range": args.scale_range,
        "jitter_std": args.jitter_std,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_model(num_classes: int, random_state: int):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        print(exc)
        raise RuntimeError(
            "XGBoost is installed but could not be loaded. "
            "On macOS this usually means the OpenMP runtime is missing "
            "(`libomp.dylib`). Install libomp, then rerun training."
        ) from exc

    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=0,
    )


def build_feature_matrix(landmarks_batch: np.ndarray, handedness_batch: list[str | None]) -> np.ndarray:
    return np.asarray(
        [
            engineer_features(landmarks, handedness).tolist()
            for landmarks, handedness in zip(landmarks_batch, handedness_batch)
        ],
        dtype=np.float32,
    )


def augment_landmarks(
    landmarks_batch: np.ndarray,
    handedness_batch: list[str | None],
    labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[str | None], np.ndarray]:
    if args.augment_copies <= 0:
        return landmarks_batch, handedness_batch, labels

    rng = np.random.default_rng(args.random_state)
    augmented_landmarks: list[np.ndarray] = [landmarks_batch]
    augmented_handedness: list[str | None] = list(handedness_batch)
    augmented_labels: list[np.ndarray] = [labels]

    finite_mask = np.isfinite(landmarks_batch).all(axis=(1, 2))
    valid_landmarks = landmarks_batch[finite_mask]
    valid_labels = labels[finite_mask]
    valid_handedness = [handedness_batch[idx]
                        for idx, is_valid in enumerate(finite_mask) if is_valid]

    if len(valid_landmarks) == 0:
        return landmarks_batch, handedness_batch, labels

    for _ in range(args.augment_copies):
        transformed_batch = np.asarray(
            [
                apply_random_spatial_transform(
                    landmarks,
                    rng=rng,
                    rotation_degrees=args.rotation_degrees,
                    translation_range=args.translation_range,
                    scale_range=args.scale_range,
                    jitter_std=args.jitter_std,
                )
                for landmarks in valid_landmarks
            ],
            dtype=np.float32,
        )
        augmented_landmarks.append(transformed_batch)
        augmented_handedness.extend(valid_handedness)
        augmented_labels.append(valid_labels)

    return (
        np.concatenate(augmented_landmarks, axis=0),
        augmented_handedness,
        np.concatenate(augmented_labels, axis=0),
    )


def apply_random_spatial_transform(
    landmarks: np.ndarray,
    *,
    rng: np.random.Generator,
    rotation_degrees: float,
    translation_range: float,
    scale_range: float,
    jitter_std: float,
) -> np.ndarray:
    centered = landmarks.astype(np.float32, copy=True)
    anchor = centered[0].copy()
    centered -= anchor

    rotation = random_rotation_matrix(rng, max_degrees=rotation_degrees)
    axis_scale = rng.uniform(1.0 - scale_range, 1.0 +
                             scale_range, size=3).astype(np.float32)
    translation = rng.uniform(-translation_range,
                              translation_range, size=3).astype(np.float32)
    jitter = rng.normal(0.0, jitter_std, size=centered.shape).astype(np.float32)

    transformed = centered @ rotation.T
    transformed *= axis_scale
    transformed += jitter
    transformed += anchor + translation
    return transformed.astype(np.float32)


def random_rotation_matrix(rng: np.random.Generator, max_degrees: float) -> np.ndarray:
    angles = np.deg2rad(rng.uniform(-max_degrees, max_degrees, size=3))
    x_angle, y_angle, z_angle = angles

    rotation_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(x_angle), -np.sin(x_angle)],
            [0.0, np.sin(x_angle), np.cos(x_angle)],
        ],
        dtype=np.float32,
    )
    rotation_y = np.asarray(
        [
            [np.cos(y_angle), 0.0, np.sin(y_angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(y_angle), 0.0, np.cos(y_angle)],
        ],
        dtype=np.float32,
    )
    rotation_z = np.asarray(
        [
            [np.cos(z_angle), -np.sin(z_angle), 0.0],
            [np.sin(z_angle), np.cos(z_angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return rotation_z @ rotation_y @ rotation_x


def validate_args(args: argparse.Namespace) -> None:
    if args.augment_copies < 0:
        raise ValueError("--augment-copies must be >= 0.")
    if args.rotation_degrees < 0:
        raise ValueError("--rotation-degrees must be >= 0.")
    if args.translation_range < 0:
        raise ValueError("--translation-range must be >= 0.")
    if not 0 <= args.scale_range < 1:
        raise ValueError("--scale-range must be in the interval [0, 1).")
    if args.jitter_std < 0:
        raise ValueError("--jitter-std must be >= 0.")


def train(args: argparse.Namespace) -> int:
    validate_args(args)
    print(args)
    input_path = args.input.resolve()
    model_output = args.model_output.resolve()
    labels_output = args.labels_output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    landmark_rows, labels, handedness_values, feature_names = load_dataset(
        input_path, args.include_missing)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    landmarks_train, landmarks_test, handedness_train, handedness_test, y_train, y_test = train_test_split(
        landmark_rows,
        handedness_values,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    augmented_landmarks_train, augmented_handedness_train, y_train_augmented = augment_landmarks(
        landmarks_train,
        handedness_train,
        y_train,
        args,
    )
    x_train = build_feature_matrix(
        augmented_landmarks_train, augmented_handedness_train)
    x_test = build_feature_matrix(landmarks_test, handedness_test)

    model = build_model(
        num_classes=len(label_encoder.classes_),
        random_state=args.random_state,
    )
    model.fit(x_train, y_train_augmented)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_output)
    save_label_metadata(labels_output, label_encoder, feature_names, args)

    print(f"Samples used: {len(landmark_rows)}")
    print(f"Augmented training samples: {len(x_train)}")
    print(f"Classes: {len(label_encoder.classes_)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Saved model to {model_output}")
    print(f"Saved labels to {labels_output}")
    print()
    print(report)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return train(args)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
