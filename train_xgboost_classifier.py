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
    return parser


def load_dataset(csv_path: Path, include_missing: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: list[list[float]] = []
    labels: list[str] = []
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
            landmarks = row_to_landmarks(row)
            rows.append(engineer_features(landmarks, row.get("handedness")).tolist())

    if not rows:
        raise ValueError("No training rows available after filtering the dataset.")

    return np.asarray(rows, dtype=np.float32), np.asarray(labels), engineered_feature_names


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


def train(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    model_output = args.model_output.resolve()
    labels_output = args.labels_output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    x, labels, feature_names = load_dataset(input_path, args.include_missing)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = build_model(
        num_classes=len(label_encoder.classes_),
        random_state=args.random_state,
    )
    model.fit(x_train, y_train)

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

    print(f"Samples used: {len(x)}")
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
