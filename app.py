from __future__ import annotations

import sys

import cv2
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


class WebcamViewerTab(QWidget):
    def __init__(self, camera_id: int = 0, interval_ms: int = 30) -> None:
        super().__init__()

        self.capture = cv2.VideoCapture(camera_id)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open webcam {camera_id}.")

        self.paused = False
        self.last_frame = None

        self.video_label = QLabel("Waiting for webcam...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background: #111; color: #ddd;")

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)

        layout = QVBoxLayout(self)
        layout.addWidget(self.video_label, stretch=1)
        layout.addWidget(self.pause_button)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(interval_ms)

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_button.setText("Resume" if self.paused else "Pause")
        if not self.paused:
            self.update_frame()

    def update_frame(self) -> None:
        if self.paused and self.last_frame is not None:
            self.show_frame(self.last_frame)
            return

        ok, frame = self.capture.read()
        if not ok:
            self.video_label.setText("Failed to read from webcam.")
            return

        frame = cv2.flip(frame, 1)
        self.last_frame = frame
        self.show_frame(frame)

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
        super().closeEvent(event)


class PlaceholderTab(QWidget):
    def __init__(self) -> None:
        super().__init__()

        label = QLabel("Future app sections can live here.")
        label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(label)


class AppWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hasta Recognition")
        self.resize(960, 720)

        tabs = QTabWidget()
        tabs.addTab(WebcamViewerTab(), "Viewer")
        tabs.addTab(PlaceholderTab(), "Workspace")
        self.setCentralWidget(tabs)


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
