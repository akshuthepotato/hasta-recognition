from __future__ import annotations

import queue
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from PySide6.QtCore import QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
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

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


@dataclass(frozen=True)
class Interpretation:
    label: str
    description: str | None = None
    video_path: Path | None = None


@dataclass(frozen=True)
class MudraEntry:
    name: str
    sketch_path: Path
    interpretations: tuple[Interpretation, ...]


class ClickableVideoLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._pixmap_rect().contains(
            event.position().toPoint()
        ):
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def _pixmap_rect(self):
        pixmap = self.pixmap()
        if pixmap is None:
            return self.rect()

        x = (self.width() - pixmap.width()) // 2
        y = (self.height() - pixmap.height()) // 2
        return pixmap.rect().translated(x, y)


MUDRA_ARCHIVE: tuple[MudraEntry, ...] = (
    MudraEntry(
        name="PlaceHolder3",
        sketch_path=ASSETS_DIR / "pathaka_sketch.png",
        interpretations=(
            Interpretation(
                label="Blessing",
                description="Used as an open-palmed gesture of blessing, assurance, or calm restraint.",
            ),
            Interpretation(
                label="Stop",
                description="Can be read as a firm stopping gesture or a sign of setting a boundary.",
            ),
            Interpretation(
                label="Mirror",
                video_path=ASSETS_DIR / "pathaka_mirror.MOV",
            ),
        ),
    ),
    MudraEntry(
        name="PlaceHolder2",
        sketch_path=ASSETS_DIR / "pathaka_sketch.png",
        interpretations=(
            Interpretation(
                label="Blessing",
                description="Used as an open-palmed gesture of blessing, assurance, or calm restraint.",
            ),
            Interpretation(
                label="Stop",
                description="Can be read as a firm stopping gesture or a sign of setting a boundary.",
            ),
            Interpretation(
                label="Mirror",
                video_path=ASSETS_DIR / "pathaka_mirror.MOV",
            ),
        ),
    ),
    MudraEntry(
        name="PlaceHolder1",
        sketch_path=ASSETS_DIR / "pathaka_sketch.png",
        interpretations=(
            Interpretation(
                label="Blessing",
                description="Used as an open-palmed gesture of blessing, assurance, or calm restraint.",
            ),
            Interpretation(
                label="Stop",
                description="Can be read as a firm stopping gesture or a sign of setting a boundary.",
            ),
            Interpretation(
                label="Mirror",
                video_path=ASSETS_DIR / "pathaka_mirror.MOV",
            ),
        ),
    ),
    MudraEntry(
        name="Pataka",
        sketch_path=ASSETS_DIR / "pathaka_sketch.png",
        interpretations=(
            Interpretation(
                label="Blessing",
                description="Used as an open-palmed gesture of blessing, assurance, or calm restraint.",
            ),
            Interpretation(
                label="Stop",
                description="Can be read as a firm stopping gesture or a sign of setting a boundary.",
            ),
            Interpretation(
                label="Mirror",
                video_path=ASSETS_DIR / "pathaka_mirror.MOV",
            ),
        ),
    ),
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

        self.video_label = ClickableVideoLabel("Waiting for webcam...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background: #111; color: #ddd;")
        self.video_label.setCursor(Qt.PointingHandCursor)
        self.video_label.clicked.connect(self.toggle_pause)

        layout = QVBoxLayout(self)
        layout.addWidget(self.video_label, stretch=1)

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
            self.toggle_pause()
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

    def draw_paused_overlay(self, frame) -> None:
        cv2.putText(
            frame,
            "Paused",
            (16, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        height, width = frame.shape[:2]
        overlay = frame.copy()
        cv2.circle(
            overlay,
            (width // 2, height // 2),
            min(width, height) // 8,
            (255, 255, 255),
            -1,
        )
        frame[:] = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        bar_height = min(width, height) // 10
        bar_width = max(10, bar_height // 3)
        gap = bar_width
        center_x = width // 2
        center_y = height // 2
        top = center_y - bar_height // 2
        bottom = center_y + bar_height // 2
        cv2.rectangle(
            frame,
            (center_x - gap // 2 - bar_width, top),
            (center_x - gap // 2, bottom),
            (40, 40, 40),
            -1,
        )
        cv2.rectangle(
            frame,
            (center_x + gap // 2, top),
            (center_x + gap // 2 + bar_width, bottom),
            (40, 40, 40),
            -1,
        )

    def show_frame(self, frame) -> None:
        display_frame = frame
        if self.paused:
            display_frame = frame.copy()
            self.draw_paused_overlay(display_frame)

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
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


class InterpretationPopup(QDialog):
    def __init__(
        self,
        title: str,
        description: str | None = None,
        video_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setMinimumSize(420, 320)
        self.setStyleSheet(
            """
            QDialog {
                background: #101010;
                border: 1px solid #353535;
                border-radius: 12px;
            }
            QLabel {
                color: #f2f2f2;
            }
            """
        )

        self.player: QMediaPlayer | None = None
        self.video_widget: QVideoWidget | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(title_label)

        if video_path is not None and video_path.exists():
            self.video_widget = QVideoWidget(self)
            self.video_widget.setMinimumSize(520, 360)
            layout.addWidget(self.video_widget, stretch=1)

            self.player = QMediaPlayer(self)
            self.player.setVideoOutput(self.video_widget)
            self.player.setSource(QUrl.fromLocalFile(str(video_path.resolve())))
            self.player.play()
        else:
            body_label = QLabel(description or "No details available.")
            body_label.setWordWrap(True)
            body_label.setStyleSheet("font-size: 14px; line-height: 1.4;")
            layout.addWidget(body_label)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.player is not None:
            self.player.stop()
        super().closeEvent(event)


class MudraCard(QFrame):
    def __init__(self, entry: MudraEntry, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.entry = entry
        self.popup: InterpretationPopup | None = None
        self.setObjectName("mudraCard")
        self.setStyleSheet(
            """
            QFrame#mudraCard {
                background: #f7f2e8;
                border: 1px solid #d8c6aa;
                border-radius: 16px;
            }
            QLabel {
                color: #2d2116;
            }
            QPushButton {
                background: #2d2116;
                color: #fffaf0;
                border: none;
                border-radius: 10px;
                padding: 8px 14px;
            }
            QPushButton:hover {
                background: #473424;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        name_label = QLabel(entry.name)
        name_label.setStyleSheet("font-size: 22px; font-weight: 700;")
        layout.addWidget(name_label)

        sketch_label = QLabel()
        sketch_label.setAlignment(Qt.AlignCenter)
        sketch_label.setMinimumHeight(240)
        sketch_label.setStyleSheet(
            "background: #fffaf2; border: 1px solid #dcc8aa; border-radius: 12px;"
        )
        pixmap = QPixmap(str(entry.sketch_path))
        if pixmap.isNull():
            sketch_label.setText("Sketch unavailable")
        else:
            sketch_label.setPixmap(
                pixmap.scaled(
                    420,
                    320,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        layout.addWidget(sketch_label)

        instructions = QLabel(
            "Choose an interpretation to see a note or demonstration.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-size: 14px;")
        layout.addWidget(instructions)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        for interpretation in entry.interpretations:
            button = QPushButton(interpretation.label)
            button.clicked.connect(
                lambda _checked=False, mudra=entry, item=interpretation: self.show_interpretation(
                    mudra,
                    item,
                )
            )
            buttons_layout.addWidget(button)
        buttons_layout.addStretch(1)
        layout.addLayout(buttons_layout)

    def show_interpretation(
        self,
        mudra: MudraEntry,
        interpretation: Interpretation,
    ) -> None:
        self.popup = InterpretationPopup(
            title=f"{mudra.name}: {interpretation.label}",
            description=interpretation.description,
            video_path=interpretation.video_path,
            parent=self,
        )
        self.popup.move(self.mapToGlobal(self.rect().center()) -
                        self.popup.rect().center())
        self.popup.show()


class MudraArchiveTab(QWidget):
    def __init__(self, entries: tuple[MudraEntry, ...]) -> None:
        super().__init__()
        self.entries = {entry.name.lower(): entry for entry in entries}
        self.cards: dict[str, MudraCard] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        heading = QLabel("Mudra archive")
        heading.setStyleSheet("font-size: 28px; font-weight: 700;")
        layout.addWidget(heading)

        subheading = QLabel(
            "Browse mudra sketches and open interpretation popups for notes or mirrored demonstrations."
        )
        subheading.setWordWrap(True)
        subheading.setStyleSheet("color: #555; font-size: 14px;")
        layout.addWidget(subheading)

        self.status_label = QLabel("Waiting for an automatic hold pause.")
        self.status_label.setStyleSheet("font-size: 14px; color: #7a4d16;")
        layout.addWidget(self.status_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        layout.addWidget(self.scroll_area, stretch=1)

        content = QWidget()
        self.grid = QGridLayout(content)
        self.grid.setContentsMargins(0, 4, 0, 4)
        self.grid.setHorizontalSpacing(16)
        self.grid.setVerticalSpacing(16)

        for index, entry in enumerate(entries):
            card = MudraCard(entry, self)
            self.cards[entry.name.lower()] = card
            self.grid.addWidget(card, index, 0)

        self.grid.setRowStretch(len(entries), 1)
        self.scroll_area.setWidget(content)

    def set_hold_label(self, label: str | None) -> None:
        if not label:
            self.status_label.setText("Hold completed.")
            return

        normalized = label.lower()
        if normalized in self.cards:
            self.status_label.setText(
                f"Held mudra: {label}. Matching entry is available below.")
        else:
            self.status_label.setText(f"Held mudra: {label}. No archive entry yet.")

    def focus_entry(self, label: str | None) -> None:
        if not label:
            return

        card = self.cards.get(label.lower())
        if card is None:
            return

        self.scroll_area.ensureWidgetVisible(card, 0, 24)


class AppWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hasta Recognition")
        self.resize(960, 720)

        self.tabs = QTabWidget()
        self.workspace_tab = MudraArchiveTab(MUDRA_ARCHIVE)
        self.viewer_tab = WebcamViewerTab(on_hold_pause=self.handle_hold_pause)

        self.tabs.addTab(self.viewer_tab, "Viewer")
        self.tabs.addTab(self.workspace_tab, "Mudra archive")
        self.tabs.currentChanged.connect(self.handle_tab_changed)
        self.setCentralWidget(self.tabs)

    def handle_hold_pause(self, label: str | None) -> None:
        self.workspace_tab.set_hold_label(label)
        self.tabs.setCurrentWidget(self.workspace_tab)
        self.workspace_tab.focus_entry(label)

    def handle_tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is not self.viewer_tab:
            self.viewer_tab.set_paused(True)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if (
            event.key() == Qt.Key_Space
            and self.tabs.currentWidget() is self.viewer_tab
            and not event.isAutoRepeat()
        ):
            self.viewer_tab.toggle_pause()
            event.accept()
            return
        super().keyPressEvent(event)


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
