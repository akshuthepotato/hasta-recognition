# Hasta Detection

Simple MediaPipe hand landmark extraction for the image dataset in `data/`.

## Run the extractor

Use the default dataset folder and write a CSV to `data/hand_landmarks.csv`:

```bash
UV_CACHE_DIR=.uv-cache MPLCONFIGDIR=.mplconfig uv run python extract_keypoints.py \
  --download-model
```

If you already have the MediaPipe task model locally, point to it directly:

```bash
UV_CACHE_DIR=.uv-cache MPLCONFIGDIR=.mplconfig uv run python extract_keypoints.py \
  --model-path /absolute/path/to/hand_landmarker.task
```

## Output format

The generated CSV contains:

- `label`: class name from the image's parent folder
- `image_path`: path relative to `data/`
- `detected`: whether MediaPipe found a hand in the image
- `handedness` and `handedness_score`
- `lm_00_x` ... `lm_20_z`: normalized MediaPipe landmarks for the selected hand

Images with no detected hand are still written by default with empty landmark columns. Pass `--skip-missing` if you only want successful detections.

## Train the classifier

Train a multiclass XGBoost classifier from the generated landmark CSV:

```bash
UV_CACHE_DIR=.uv-cache MPLCONFIGDIR=.mplconfig uv run python train_xgboost_classifier.py
```

This saves:

- `models/xgboost_hand_classifier.json`: trained XGBoost model
- `models/xgboost_hand_classifier_labels.json`: class names and feature metadata

On macOS, XGBoost may also require the OpenMP runtime (`libomp.dylib`) to be installed before training can run successfully.
