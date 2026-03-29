from __future__ import annotations

import queue
import sys
import time
from collections.abc import Callable

import cv2
import mediapipe as mp
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from main import (
    BaseOptions,
    DEFAULT_CLASSIFIER_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_MODEL_PATH,
    FrameResult,
    HandLandmarker,
    HandLandmarkerOptions,
    HoldStateMachine,
    LiveMudraClassifier,
    VisionRunningMode,
    draw_hand_landmarks,
    draw_progress_circle,
)


class WebcamViewerTab(QWidget):
    def __init__(
        self,
        camera_id: int = 0,
        interval_ms: int = 30,
        hold_duration: float = 5.0,
        classification_threshold: float = 0.5,
        on_hold_pause: Callable[[str | None], None] | None = None,
    ) -> None:
        super().__init__()

        self.on_hold_pause = on_hold_pause
        self.capture = cv2.VideoCapture(camera_id)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open webcam {camera_id}.")

        self.paused = False
        self.last_frame = None
        self.latest_result: FrameResult | None = None
        self.last_prediction_label: str | None = None
        self.result_queue: queue.Queue[FrameResult] = queue.Queue(maxsize=1)

        self.classifier = LiveMudraClassifier(
            DEFAULT_CLASSIFIER_PATH,
            DEFAULT_LABELS_PATH,
            classification_threshold,
        )
        self.hold_sm = HoldStateMachine(hold_duration)

        self.video_label = QLabel("Waiting for webcam...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background: #111; color: #ddd;")

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)

        layout = QVBoxLayout(self)
        layout.addWidget(self.video_label, stretch=1)
        layout.addWidget(self.pause_button)

        self.landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(DEFAULT_MODEL_PATH)),
                running_mode=VisionRunningMode.LIVE_STREAM,
                result_callback=self._on_detection_result,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(interval_ms)

    def _on_detection_result(self, result, output_image, timestamp_ms: int) -> None:
        del output_image

        prediction = self.classifier.predict(result)
        frame_result = FrameResult(timestamp_ms, result, prediction)

        while True:
            try:
                self.result_queue.put_nowait(frame_result)
                break
            except queue.Full:
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break

    def toggle_pause(self) -> None:
        self.set_paused(not self.paused)

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        self.pause_button.setText("Resume" if self.paused else "Pause")
        if not paused:
            self.hold_sm.reset()
            self.latest_result = None
            self.last_prediction_label = None
            self._drain_result_queue()
            self.update_frame()

    def _drain_result_queue(self) -> None:
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                return

    def _pause_from_hold(self, detected_label) -> None:
        if not self.paused:
            self.pause_button.click()
        if self.on_hold_pause is not None:
            self.on_hold_pause(detected_label)

    def update_frame(self) -> None:
        if self.paused and self.last_frame is not None:
            self.show_frame(self.last_frame)
            return

        ok, frame = self.capture.read()
        if not ok:
            self.video_label.setText("Failed to read from webcam.")
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.landmarker.detect_async(mp_image, int(time.time() * 1000))

        while not self.result_queue.empty():
            try:
                self.latest_result = self.result_queue.get_nowait()
            except queue.Empty:
                break

        progress = 0.0
        should_pause = False

        if self.latest_result and self.latest_result.prediction is not None:
            prediction = self.latest_result.prediction
            self.last_prediction_label = prediction.label
            progress, should_pause, detected_label = self.hold_sm.update(
                prediction.label)

            draw_hand_landmarks(frame, self.latest_result.result)
            draw_progress_circle(
                frame,
                self.latest_result.result,
                progress,
                should_pause,
            )
            self.draw_overlay(frame, prediction.label, prediction.confidence, progress)
        else:
            self.hold_sm.reset()
            self.last_prediction_label = None
            self.draw_overlay(frame, "no hand", None, progress)

        self.last_frame = frame
        self.show_frame(frame)

        if should_pause:
            self._pause_from_hold(detected_label)

    def draw_overlay(
        self,
        frame,
        label: str,
        confidence: float | None,
        progress: float,
    ) -> None:
        lines = []
        if confidence is None:
            lines.append(f"Mudra: {label}")
        else:
            lines.append(f"Mudra: {label} ({confidence:.2f})")
        lines.append(f"Hold: {progress:.0%}")

        y = 30
        for line in lines:
            cv2.putText(
                frame,
                line,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            y += 30

    def show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.last_frame is not None:
            self.show_frame(self.last_frame)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.timer.stop()
        if self.capture.isOpened():
            self.capture.release()
        self.landmarker.close()
        super().closeEvent(event)


class PlaceholderTab(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.label = QLabel("Waiting for an automatic hold pause.")
        self.label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

    def set_hold_label(self, label: str | None) -> None:
        if label:
            self.label.setText(f"Held mudra: {label}")
            # TODO: add more stuff here
        else:
            self.label.setText("Hold completed.")


class AppWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hasta Recognition")
        self.resize(960, 720)

        self.tabs = QTabWidget()
        self.workspace_tab = PlaceholderTab()
        self.viewer_tab = WebcamViewerTab(on_hold_pause=self.handle_hold_pause)

        self.tabs.addTab(self.viewer_tab, "Viewer")
        self.tabs.addTab(self.workspace_tab, "Workspace")
        self.setCentralWidget(self.tabs)

    def handle_hold_pause(self, label: str | None) -> None:
        self.workspace_tab.set_hold_label(label)
        self.tabs.setCurrentWidget(self.workspace_tab)


def main() -> int:
    app = QApplication(sys.argv)
    try:
        window = AppWindow()
    except Exception as exc:
        QMessageBox.critical(None, "Webcam Error", str(exc))
        return 1

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
