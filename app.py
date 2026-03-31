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
from PySide6.QtGui import QFont, QImage, QKeyEvent, QPixmap
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedLayout,
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
PATHAAKAM_DIR = ASSETS_DIR / "PATHAAKAM"
HAMSASYAM_DIR = ASSETS_DIR / "hamsasyam"

INK = "#171411"
PAPER = "#f2ead9"
PAPER_SOFT = "#e7d7b6"
SEPIA = "#c5a46d"
SEPIA_BRIGHT = "#ddb36d"
EARTH = "#7a5b31"
UMBER = "#49321c"
OLIVE_INK = "#6d6435"
PANEL_DARK = "#211810"
OVERLAY_LINE = (106, 88, 48)
OVERLAY_ACCENT = (205, 170, 107)
OVERLAY_TEXT = (246, 232, 205)
APP_FONT_FAMILY = '"Aglatia Regular", "Aglatia", serif'


APP_STYLESHEET = f"""
QMainWindow {{
    background: {PAPER};
}}
QWidget {{
    color: {INK};
    font-family: {APP_FONT_FAMILY};
}}
QTabWidget::pane {{
    border: 1px solid {SEPIA};
    border-radius: 20px;
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 {PAPER},
        stop: 1 {PAPER_SOFT}
    );
    top: -1px;
}}
QTabBar::tab {{
    background: #d9c39f;
    color: {UMBER};
    border: 1px solid #bea06f;
    padding: 10px 18px;
    margin-right: 8px;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
}}
QTabBar::tab:selected {{
    background: {PANEL_DARK};
    color: {PAPER};
    border-color: {PANEL_DARK};
}}
QScrollArea {{
    background: {PAPER};
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: {PAPER};
}}
QScrollBar:vertical {{
    background: #d8c5a6;
    width: 12px;
    border-radius: 6px;
}}
QScrollBar::handle:vertical {{
    background: #8d7346;
    border-radius: 6px;
    min-height: 32px;
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
}}
"""


def _display_hasta_name(label: str) -> str:
    if label == "Pataka":
        return "Pathaakam"
    if label == "Hamsasya":
        return "Hamsasyam"
    return label.replace("_", " ").title()


def _archive_lookup_key(label: str | None) -> str | None:
    if not label:
        return None
    return _display_hasta_name(label).lower()


def _format_detection_name(label: str | None) -> str:
    if not label:
        return "No hand detected"
    return _display_hasta_name(label)

def _build_detection_note(label: str | None, confidence: float | None) -> str:
    del label, confidence
    return ""


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
    performer_description: str | None = None


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


def _format_interpretation_label(asset_path: Path) -> str:
    return asset_path.stem.replace("_", " ").title()


MUDRA_PERFORMER_DESCRIPTIONS: dict[str, str] = {
    "Pathaakam": "Performed by Nikita, Singapore Adavu.",
    "Hamsasyam": "Performed by Swathi, Singapore Adavu.",
}


def _load_directory_interpretations(
    directory: Path,
    *,
    performer_description: str | None = None,
) -> tuple[Interpretation, ...]:
    video_paths = sorted(directory.glob("*.MOV"))
    return tuple(
        Interpretation(
            label=_format_interpretation_label(video_path),
            description=performer_description,
            video_path=video_path,
        )
        for video_path in video_paths
    )


MUDRA_ARCHIVE: tuple[MudraEntry, ...] = (
    MudraEntry(
        name="Pathaakam",
        sketch_path=PATHAAKAM_DIR / "sketch.png",
        performer_description=MUDRA_PERFORMER_DESCRIPTIONS.get("Pathaakam"),
        interpretations=_load_directory_interpretations(
            PATHAAKAM_DIR,
            performer_description=MUDRA_PERFORMER_DESCRIPTIONS.get("Pathaakam"),
        ),
    ),
    MudraEntry(
        name="Hamsasyam",
        sketch_path=HAMSASYAM_DIR / "hamsasyam sketch.png",
        performer_description=MUDRA_PERFORMER_DESCRIPTIONS.get("Hamsasyam"),
        interpretations=_load_directory_interpretations(
            HAMSASYAM_DIR,
            performer_description=MUDRA_PERFORMER_DESCRIPTIONS.get("Hamsasyam"),
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        content_row = QHBoxLayout()
        content_row.setSpacing(18)
        layout.addLayout(content_row, stretch=1)

        video_column = QVBoxLayout()
        video_column.setSpacing(12)
        content_row.addLayout(video_column, stretch=5)

        self.video_frame = QFrame(self)
        self.video_frame.setObjectName("viewerStage")
        self.video_frame.setStyleSheet(
            f"""
            QFrame#viewerStage {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #1c150f,
                    stop: 1 #362619
                );
                border: 1px solid #6e5634;
                border-radius: 28px;
            }}
            """
        )
        video_frame_layout = QVBoxLayout(self.video_frame)
        video_frame_layout.setContentsMargins(18, 18, 18, 18)
        video_frame_layout.setSpacing(10)

        self.video_label = ClickableVideoLabel("Waiting for webcam...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            """
            background: #0f0d0a;
            color: #e6d9c0;
            border-radius: 18px;
            """
        )
        self.video_label.setCursor(Qt.PointingHandCursor)
        self.video_label.clicked.connect(self.toggle_pause)
        video_frame_layout.addWidget(self.video_label, stretch=1)

        video_column.addWidget(self.video_frame, stretch=1)

        info_column = QVBoxLayout()
        info_column.setSpacing(14)
        content_row.addLayout(info_column, stretch=2)

        self.signal_card = QFrame(self)
        self.signal_card.setObjectName("signalCard")
        self.signal_card.setStyleSheet(
            f"""
            QFrame#signalCard {{
                background: {PANEL_DARK};
                border: 1px solid #5a452c;
                border-radius: 24px;
            }}
            QLabel {{
                color: {PAPER};
            }}
            """
        )
        signal_layout = QVBoxLayout(self.signal_card)
        signal_layout.setContentsMargins(18, 18, 18, 18)
        signal_layout.setSpacing(10)

        signal_heading = QLabel("Detection plate")
        signal_heading.setStyleSheet(
            f"font-size: 12px; text-transform: uppercase; letter-spacing: 0.2em; color: #cbb48d;"
        )
        signal_layout.addWidget(signal_heading)

        self.detected_label = QLabel("No hand detected")
        self.detected_label.setWordWrap(True)
        self.detected_label.setStyleSheet("font-size: 28px; font-weight: 700;")
        signal_layout.addWidget(self.detected_label)

        self.confidence_label = QLabel("Confidence / --")
        self.confidence_label.setStyleSheet(
            "font-size: 13px; color: #dac7a4;"
        )
        signal_layout.addWidget(self.confidence_label)

        self.hold_label = QLabel("Hold progress / 0%")
        self.hold_label.setStyleSheet(
            "font-size: 13px; color: #dac7a4;"
        )
        signal_layout.addWidget(self.hold_label)

        self.note_label = QLabel(
            ""
        )
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet("font-size: 14px; color: #f0e5cf;")
        signal_layout.addWidget(self.note_label)
        info_column.addWidget(self.signal_card)

        info_column.addStretch(1)

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
            self._update_signal_labels("no hand", None, 0.0, False)
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
        detected_label = None

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
            self._update_signal_labels(
                prediction.label,
                prediction.confidence,
                progress,
                should_pause,
            )
        else:
            self.hold_sm.reset()
            self.last_prediction_label = None
            self.draw_overlay(frame, "no hand", None, progress)
            self._update_signal_labels("no hand", None, progress, False)

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
        cv2.rectangle(frame, (18, 18), (frame.shape[1] - 18, frame.shape[0] - 18), OVERLAY_LINE, 1)
        del label, confidence, progress

    def draw_paused_overlay(self, frame) -> None:
        height, width = frame.shape[:2]
        overlay = frame.copy()
        cv2.circle(
            overlay,
            (width // 2, height // 2),
            min(width, height) // 8,
            (241, 225, 195),
            -1,
        )
        frame[:] = cv2.addWeighted(overlay, 0.28, frame, 0.72, 0)
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
            (56, 43, 28),
            -1,
        )
        cv2.rectangle(
            frame,
            (center_x + gap // 2, top),
            (center_x + gap // 2 + bar_width, bottom),
            (56, 43, 28),
            -1,
        )

    def _update_signal_labels(
        self,
        label: str | None,
        confidence: float | None,
        progress: float,
        paused: bool,
    ) -> None:
        formatted_label = _format_detection_name(label)
        self.detected_label.setText(formatted_label)
        self.confidence_label.setText(
            "Confidence / --" if confidence is None else f"Confidence / {confidence:.0%}"
        )
        self.hold_label.setText(f"Hold progress / {progress:.0%}")
        self.note_label.setText(_build_detection_note(label, confidence))

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


class InterpretationSidePanel(QFrame):
    def __init__(
        self, parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("interpretationPanel")
        self.setMinimumWidth(420)
        self.setStyleSheet(
            f"""
            QFrame#interpretationPanel {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #1e1610,
                    stop: 1 #2b1f15
                );
                border: 1px solid #6e5634;
                border-radius: 22px;
            }}
            QLabel {{
                color: #f5ead5;
            }}
            QPushButton {{
                background: #d4b07a;
                color: #26180e;
                border: 1px solid #b89562;
                border-radius: 14px;
                padding: 9px 14px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background: #e2bf87;
            }}
            """
        )

        self.current_title = "Select an interpretation"
        self.player: QMediaPlayer | None = None
        self.video_widget: QVideoWidget | None = None
        self.poster_label: ClickableVideoLabel | None = None
        self.poster_pixmap: QPixmap | None = None
        self.media_layout: QStackedLayout | None = None
        self.current_video_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        layout.addLayout(header)

        self.title_label = QLabel(self.current_title)
        self.title_label.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {PAPER};"
        )
        self.title_label.setWordWrap(True)
        header.addWidget(self.title_label, stretch=1)

        self.collapse_button = QPushButton("Collapse")
        header.addWidget(self.collapse_button)

        media_container = QWidget(self)
        self.media_layout = QStackedLayout(media_container)
        self.media_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(media_container, stretch=1)

        self.video_widget = QVideoWidget(self)
        self.video_widget.setMinimumSize(320, 220)
        self.media_layout.addWidget(self.video_widget)

        self.poster_label = ClickableVideoLabel()
        self.poster_label.setAlignment(Qt.AlignCenter)
        self.poster_label.setMinimumSize(320, 220)
        self.poster_label.setStyleSheet("background: #000; border-radius: 8px;")
        self.poster_label.clicked.connect(self.restart_video)
        self.media_layout.addWidget(self.poster_label)

        self.body_label = QLabel(
            "Select a hasta interpretation to play the mirrored demonstration or read its note.")
        self.body_label.setWordWrap(True)
        self.body_label.setStyleSheet(
            f"font-size: 14px; line-height: 1.4; color: #f0dfc0;"
        )
        layout.addWidget(self.body_label)

        self.player = QMediaPlayer(self)
        self.player.setVideoOutput(self.video_widget)
        self.player.mediaStatusChanged.connect(self._handle_media_status_changed)

        self.media_layout.setCurrentWidget(self.poster_label)

    def set_interpretation(
        self,
        mudra: MudraEntry,
        interpretation: Interpretation,
    ) -> None:
        self.current_title = f"{mudra.name}: {interpretation.label}"
        self.title_label.setText(self.current_title)
        self.body_label.setText(
            interpretation.description
            or mudra.performer_description
            or "No details available."
        )

        video_path = interpretation.video_path
        self.current_video_path = video_path
        self.poster_pixmap = None

        if video_path is not None and video_path.exists():
            self.poster_pixmap = self._load_video_preview(video_path)
            self._update_poster_pixmap()
            self.player.setSource(QUrl.fromLocalFile(str(video_path.resolve())))
            self.media_layout.setCurrentWidget(self.video_widget)
            self.player.setPosition(0)
            self.player.play()
            return

        self.player.stop()
        self.player.setSource(QUrl())
        if self.poster_label is not None:
            self.poster_label.setText("No video available")
            self.poster_label.setPixmap(QPixmap())
        self.media_layout.setCurrentWidget(self.poster_label)

    def _load_video_preview(self, video_path: Path) -> QPixmap | None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return None

        preview_frame = None
        success, first_frame = capture.read()
        if success:
            preview_frame = first_frame

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count - 1, 0))
            success, last_frame = capture.read()
            if success:
                preview_frame = last_frame

        capture.release()
        if preview_frame is None:
            return None

        rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(image)

    def _update_poster_pixmap(self) -> None:
        if self.poster_label is None or self.poster_pixmap is None:
            return

        self.poster_label.setText("")
        self.poster_label.setPixmap(
            self.poster_pixmap.scaled(
                self.poster_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _show_poster(self) -> None:
        if (
            self.poster_label is None
            or self.poster_pixmap is None
            or self.media_layout is None
        ):
            return

        if self.video_widget is not None:
            self.poster_label.resize(self.video_widget.size())

        self._update_poster_pixmap()
        self.media_layout.setCurrentWidget(self.poster_label)

    def restart_video(self) -> None:
        if (
            self.player is None
            or self.video_widget is None
            or self.media_layout is None
            or self.current_video_path is None
        ):
            return

        self.media_layout.setCurrentWidget(self.video_widget)
        self.player.setPosition(0)
        self.player.play()

    def _handle_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.EndOfMedia:
            self._show_poster()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_poster_pixmap()

    def stop(self) -> None:
        if self.player is not None:
            self.player.stop()

    def set_target_width(self, width: int) -> None:
        clamped_width = max(self.minimumWidth(), width)
        self.setFixedWidth(clamped_width)


class InterpretationStage(QFrame):
    def __init__(
        self,
        entry: MudraEntry,
        on_interpretation_click: Callable[[MudraEntry, Interpretation], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.entry = entry
        self.on_interpretation_click = on_interpretation_click
        self.source_pixmap = QPixmap(str(entry.sketch_path))
        self.buttons: list[QPushButton] = []
        self._button_columns = 2

        self.setMinimumHeight(360)
        self.setStyleSheet(
            f"""
            QFrame {{
                background: {PAPER};
                border: none;
                border-radius: 24px;
            }}
            QLabel {{
                background: transparent;
                color: {UMBER};
            }}
            QPushButton {{
                background: #f0dfbc;
                color: #51331b;
                border: 2px solid #c8ab78;
                border-radius: 14px;
                padding: 10px 18px;
                font-size: 13px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background: #f7e8c9;
                border-color: #b9935b;
            }}
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(24)

        sketch_frame = QFrame(self)
        sketch_frame.setStyleSheet("background: transparent;")
        sketch_layout = QVBoxLayout(sketch_frame)
        sketch_layout.setContentsMargins(0, 0, 0, 0)
        sketch_layout.setSpacing(0)

        self.sketch_label = QLabel(sketch_frame)
        self.sketch_label.setAlignment(Qt.AlignCenter)
        self.sketch_label.setMinimumSize(220, 300)
        sketch_layout.addWidget(self.sketch_label, stretch=1)
        layout.addWidget(sketch_frame, stretch=3)

        button_frame = QFrame(self)
        button_frame.setObjectName("interpretationButtonsPanel")
        button_frame.setStyleSheet(
            """
            QFrame#interpretationButtonsPanel {
                background: transparent;
                border: none;
            }
            """
        )
        button_layout = QVBoxLayout(button_frame)
        button_layout.setContentsMargins(18, 18, 18, 18)
        button_layout.setSpacing(14)

        button_heading = QLabel("Expanded meanings", button_frame)
        button_heading.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {INK};"
        )
        button_layout.addWidget(button_heading)

        self.button_grid = QGridLayout()
        self.button_grid.setContentsMargins(0, 0, 0, 0)
        self.button_grid.setHorizontalSpacing(12)
        self.button_grid.setVerticalSpacing(12)
        button_layout.addLayout(self.button_grid)
        button_layout.addStretch(1)
        layout.addWidget(button_frame, stretch=2)

        if self.source_pixmap.isNull():
            self.sketch_label.setText("Sketch unavailable")
        else:
            self._update_sketch_pixmap()

        for index, interpretation in enumerate(entry.interpretations):
            button = QPushButton(interpretation.label.lower(), self)
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(44)
            button.clicked.connect(
                lambda _checked=False, mudra=entry, item=interpretation: self.on_interpretation_click(
                    mudra,
                    item,
                )
            )
            row = index // self._button_columns
            column = index % self._button_columns
            self.button_grid.addWidget(button, row, column)
            self.buttons.append(button)

        for column in range(self._button_columns):
            self.button_grid.setColumnStretch(column, 1)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_sketch_pixmap()

    def _update_sketch_pixmap(self) -> None:
        if self.source_pixmap.isNull():
            return

        scaled = self.source_pixmap.scaled(
            self.sketch_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.sketch_label.setPixmap(scaled)


class MudraCard(QFrame):
    def __init__(
        self,
        entry: MudraEntry,
        on_interpretation_click: Callable[[MudraEntry, Interpretation], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.entry = entry
        self.on_interpretation_click = on_interpretation_click
        self.setObjectName("mudraCard")
        self.setStyleSheet(
            f"""
            QFrame#mudraCard {{
                background: {PAPER};
                border: none;
                border-radius: 26px;
            }}
            QLabel {{
                color: #25180f;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        name_label = QLabel(entry.name)
        name_label.setStyleSheet(
            f"font-size: 28px; font-weight: 700; color: {INK};"
        )
        layout.addWidget(name_label)

        instructions = QLabel(
            "Choose an interpretation to open its note or mirrored demonstration."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(
            f"font-size: 14px; color: {EARTH};"
        )
        layout.addWidget(instructions)

        stage = InterpretationStage(entry, self.on_interpretation_click, self)
        layout.addWidget(stage)


class MudraArchiveTab(QWidget):
    def __init__(self, entries: tuple[MudraEntry, ...]) -> None:
        super().__init__()
        self.entries = {entry.name.lower(): entry for entry in entries}
        self.cards: dict[str, MudraCard] = {}
        self.panel_open = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        heading = QLabel("Hasta archive")
        heading.setStyleSheet(
            f"font-size: 30px; font-weight: 700; color: {INK};"
        )
        layout.addWidget(heading)

        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(16)
        layout.addLayout(content_row, stretch=1)

        self.side_panel = InterpretationSidePanel(self)
        self.side_panel.collapse_button.clicked.connect(self.collapse_side_panel)
        self.side_panel.hide()
        content_row.addWidget(self.side_panel)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet(f"background: {PAPER}; border: none;")
        content_row.addWidget(self.scroll_area, stretch=1)

        content = QWidget()
        content.setStyleSheet(f"background: {PAPER};")
        self.grid = QGridLayout(content)
        self.grid.setContentsMargins(0, 4, 0, 4)
        self.grid.setHorizontalSpacing(20)
        self.grid.setVerticalSpacing(20)

        for index, entry in enumerate(entries):
            card = MudraCard(entry, self.show_interpretation, self)
            self.cards[entry.name.lower()] = card
            self.grid.addWidget(card, index, 0)

        self.grid.setRowStretch(len(entries), 1)
        self.scroll_area.setWidget(content)
        self._update_side_panel_width()

    def collapse_side_panel(self) -> None:
        self.panel_open = False
        self.side_panel.stop()
        self.side_panel.hide()

    def show_interpretation(
        self,
        mudra: MudraEntry,
        interpretation: Interpretation,
    ) -> None:
        self._update_side_panel_width()
        self.side_panel.set_interpretation(mudra, interpretation)
        if not self.panel_open:
            self.panel_open = True
            self.side_panel.show()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_side_panel_width()

    def _update_side_panel_width(self) -> None:
        if self.windowHandle() is not None and self.windowHandle().screen() is not None:
            screen_width = self.windowHandle().screen().availableGeometry().width()
        elif self.window() is not None:
            screen_width = self.window().width()
        else:
            screen_width = self.width()

        self.side_panel.set_target_width(int(screen_width * 0.4))

    def set_hold_label(self, label: str | None) -> None:
        del label

    def has_entry(self, label: str | None) -> bool:
        lookup_key = _archive_lookup_key(label)
        if lookup_key is None:
            return False
        return lookup_key in self.entries

    def focus_entry(self, label: str | None) -> None:
        lookup_key = _archive_lookup_key(label)
        if lookup_key is None:
            return

        card = self.cards.get(lookup_key)
        if card is None:
            return

        self.scroll_area.ensureWidgetVisible(card, 0, 24)


class AppWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hasta Recognition")
        self.resize(1280, 860)
        self.setStyleSheet(APP_STYLESHEET)

        self.tabs = QTabWidget()
        self.workspace_tab = MudraArchiveTab(MUDRA_ARCHIVE)
        self.viewer_tab = WebcamViewerTab(on_hold_pause=self.handle_hold_pause)

        self.tabs.addTab(self.viewer_tab, "Viewer")
        self.tabs.addTab(self.workspace_tab, "Hasta archive")
        self.tabs.currentChanged.connect(self.handle_tab_changed)
        self.setCentralWidget(self.tabs)

    def handle_hold_pause(self, label: str | None) -> None:
        self.workspace_tab.set_hold_label(label)
        if not self.workspace_tab.has_entry(label):
            return

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
    app.setFont(QFont("Aglatia Regular"))
    window = AppWindow()
    # try:
    #     window = AppWindow()
    # except Exception as exc:
    #     print(exc)
    #     QMessageBox.critical(None, "Webcam Error", str(exc))
    #     return 1

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
