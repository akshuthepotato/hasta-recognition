from __future__ import annotations

import argparse
import atexit
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import mediapipe as mp
from flask import Flask, Response, abort, jsonify, render_template_string, send_from_directory

from main import (
    BaseOptions,
    DEFAULT_CLASSIFIER_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_MODEL_PATH,
    HandLandmarker,
    HandLandmarkerOptions,
    HoldStateMachine,
    LiveMudraClassifier,
    VisionRunningMode,
    draw_hand_landmarks,
    draw_progress_circle,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
OVERLAY_LINE = (106, 88, 48)

PAGE_CONTENT = {
    "hero": {
        "title": "Hasta Lab",
        "subtitle": "A beginner friendly learning portal of single hand gestures and their expanded meanings.",
        "body": (
            "Bharatanatyam's single-hand gestures form a codified visual language through "
            "which meaning is constructed, communicated, and embodied. This browser version "
            "keeps the live recognition view while presenting the archive with the same warm, "
            "museum-like tone from the reference designs."
        ),
    },
    "about": [
        {
            "heading": "Background & Context",
            "body": (
                "Bharatanatyam learning today is often static, hierarchical, and inaccessible "
                "outside the classroom. Hastas are usually taught through 2D references or "
                "correction-based pedagogy, with little room for self-paced interpretation."
            ),
        },
        {
            "heading": "Hasta Lab",
            "body": (
                "This interface detects a user's hand gesture, helps them correct it in real "
                "time, and then invites them to explore its expanded meanings through sketches, "
                "reference media, and an interpretation archive."
            ),
        },
    ],
    "steps": [
        "Approach the interface and perform a hasta.",
        "Wait for the system to detect the gesture and show visual feedback.",
        "Explore expanded meanings of the recognised hand gesture.",
        "Review the mirrored demonstration and reference material.",
        "Return to the archive and continue through the collection.",
    ],
    "archive": [
        {
            "name": "Pathaakam",
            "sketch": "pathaka_sketch.png",
            "interpretations": [
                "up & down",
                "waving",
                "wind",
                "mirror",
                "slapping",
                "cutting",
                "applying makeup",
                "calling",
                "slicing",
            ],
            "video": "pathaka_mirror.MOV",
            "cta": "Explore Pathaakam",
        },
        {
            "name": "Hamsasyam",
            "sketch": "pathaka_sketch.png",
            "interpretations": [
                "up & down",
                "waving",
                "wind",
                "mirror",
                "slapping",
                "cutting",
                "applying makeup",
                "calling",
                "slicing",
            ],
            "video": "pathaka_mirror.MOV",
            "cta": "Explore Pathaakam",
        }
    ],
}


def _archive_slug(name: str) -> str:
    return name.strip().lower().replace(" ", "-")


def _display_hasta_name(label: str) -> str:
    if label == "Pataka":
        return "Pathaakam"
    if label == "Hamsasya":
        return "Hamsasyam"
    return label.replace("_", " ").title()


def _browser_visible_label(label: str | None) -> str | None:
    if label in (None, "", "uncertain"):
        return None
    return label


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Hasta Detection</title>
    <style>
      :root {
        --paper: #f4e7c3;
        --paper-soft: #efe0b6;
        --paper-deep: #e5cf94;
        --ink: #4d2918;
        --ink-soft: #71513b;
        --panel: rgba(253, 244, 216, 0.84);
        --panel-strong: rgba(248, 231, 188, 0.92);
        --border: rgba(129, 86, 46, 0.24);
        --button: #3b1e0b;
        --button-text: #f8edcf;
        --shadow: 0 24px 80px rgba(80, 48, 18, 0.18);
      }

      * {
        box-sizing: border-box;
      }

      html {
        scroll-behavior: smooth;
      }

      body {
        margin: 0;
        color: var(--ink);
        background:
          radial-gradient(circle at top, rgba(255, 244, 204, 0.75), transparent 28%),
          radial-gradient(circle at bottom left, rgba(221, 182, 98, 0.18), transparent 26%),
          linear-gradient(180deg, #f8ebc8 0%, #f3e4bb 38%, #f1ddb0 100%);
        font-family: Georgia, "Times New Roman", serif;
      }

      body::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.2;
        background:
          radial-gradient(circle at 20% 20%, rgba(120, 86, 39, 0.08), transparent 18%),
          radial-gradient(circle at 80% 0%, rgba(120, 86, 39, 0.07), transparent 20%),
          radial-gradient(circle at 50% 100%, rgba(120, 86, 39, 0.08), transparent 22%);
        mix-blend-mode: multiply;
      }

      a {
        color: inherit;
        text-decoration: none;
      }

      .shell {
        width: min(1120px, calc(100vw - 32px));
        margin: 0 auto;
        padding: 24px 0 80px;
        position: relative;
      }

      .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 8px 0 20px;
      }

      .brand {
        font-size: 1rem;
        letter-spacing: 0.24em;
        text-transform: uppercase;
        color: var(--ink-soft);
      }

      .nav {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        font-size: 0.95rem;
      }

      .nav a {
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(255, 247, 223, 0.6);
        border: 1px solid var(--border);
      }

      .hero {
        display: grid;
        grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr);
        gap: 28px;
        align-items: stretch;
        padding: 28px 0 44px;
      }

      .hero-copy,
      .hero-feed,
      .section-panel,
      .archive-card,
      .interpretation-frame {
        backdrop-filter: blur(6px);
        background: var(--panel);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
      }

      .hero-copy {
        border-radius: 36px;
        padding: 36px;
        background:
          linear-gradient(180deg, rgba(255, 248, 227, 0.94), rgba(243, 225, 177, 0.84)),
          var(--panel);
      }

      .eyebrow,
      .section-mark {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        color: var(--ink-soft);
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.78rem;
      }

      .section-mark::before,
      .eyebrow::before {
        content: "";
        width: 18px;
        height: 2px;
        background: currentColor;
      }

      h1,
      h2,
      h3 {
        margin: 0;
        font-weight: 600;
        line-height: 0.92;
      }

      h1 {
        margin-top: 18px;
        font-size: clamp(4.2rem, 9vw, 7.4rem);
        text-transform: uppercase;
      }

      .hero-copy p {
        max-width: 34rem;
        margin: 20px 0 0;
        font-size: 1.08rem;
        line-height: 1.75;
        color: var(--ink-soft);
      }

      .hero-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        margin-top: 28px;
      }

      .button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 56px;
        padding: 0 26px;
        border-radius: 18px;
        border: 1px solid rgba(53, 26, 8, 0.22);
        background: var(--button);
        color: var(--button-text);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 1rem;
        box-shadow: 0 14px 30px rgba(59, 30, 11, 0.18);
      }

      .button.secondary {
        background: rgba(255, 247, 223, 0.72);
        color: var(--ink);
      }

      .hero-feed {
        border-radius: 36px;
        padding: 18px;
        background:
          linear-gradient(180deg, rgba(90, 59, 32, 0.18), rgba(246, 232, 195, 0.88)),
          var(--panel-strong);
      }

      .hero-feed-head {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding: 8px 8px 18px;
        color: var(--ink-soft);
        font-size: 0.9rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }

      .feed-stage {
        position: relative;
        overflow: hidden;
        border-radius: 28px;
        border: 1px solid rgba(63, 36, 14, 0.16);
        background: #1c150f;
      }

      .feed-stage img {
        display: block;
        width: 100%;
        aspect-ratio: 4 / 3;
        object-fit: cover;
      }

      .feed-note {
        margin: 16px 8px 4px;
        font-size: 0.98rem;
        line-height: 1.65;
        color: var(--ink-soft);
      }

      .feed-status {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 18px 4px 2px;
      }

      .status-card {
        padding: 14px 16px;
        border-radius: 18px;
        border: 1px solid rgba(82, 51, 24, 0.12);
        background: rgba(255, 247, 223, 0.7);
      }

      .status-label {
        display: block;
        margin-bottom: 6px;
        color: var(--ink-soft);
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .status-value {
        font-size: 1.1rem;
        line-height: 1.3;
      }

      .section {
        padding: 28px 0;
      }

      .section-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 20px;
      }

      .section-panel {
        border-radius: 28px;
        padding: 28px;
      }

      .section-panel h3 {
        margin-top: 14px;
        font-size: clamp(1.8rem, 4vw, 2.8rem);
      }

      .section-panel p {
        margin: 16px 0 0;
        color: var(--ink-soft);
        line-height: 1.8;
        font-size: 1rem;
      }

      .steps {
        display: grid;
        gap: 14px;
      }

      .step {
        display: grid;
        grid-template-columns: 86px 1fr;
        gap: 18px;
        align-items: center;
        padding: 18px 22px;
        border-radius: 24px;
        border: 1px solid var(--border);
        background: linear-gradient(90deg, rgba(255, 233, 196, 0.86), rgba(252, 246, 227, 0.64));
        box-shadow: var(--shadow);
      }

      .step:nth-child(2n) {
        background: linear-gradient(90deg, rgba(241, 211, 165, 0.78), rgba(252, 240, 216, 0.64));
      }

      .step-index {
        width: 64px;
        height: 64px;
        border-radius: 18px;
        display: grid;
        place-items: center;
        background: rgba(60, 30, 11, 0.08);
        border: 1px solid rgba(60, 30, 11, 0.12);
        font-size: 1.6rem;
      }

      .step p {
        margin: 0;
        font-size: 1rem;
        line-height: 1.7;
        color: var(--ink-soft);
      }

      .archive {
        display: grid;
        gap: 24px;
      }

      .archive-card {
        border-radius: 32px;
        padding: 28px;
        display: grid;
        grid-template-columns: minmax(260px, 360px) 1fr;
        gap: 28px;
        align-items: start;
      }

      .archive-sketch {
        display: grid;
        gap: 14px;
      }

      .archive-sketch-box {
        padding: 20px;
        border-radius: 24px;
        background: rgba(255, 250, 237, 0.72);
        border: 1px dashed rgba(82, 51, 24, 0.18);
        min-height: 320px;
        display: grid;
        place-items: center;
      }

      .archive-sketch-box img {
        width: min(100%, 260px);
        max-height: 320px;
        object-fit: contain;
      }

      .archive-copy h2 {
        margin-top: 14px;
        font-size: clamp(2.4rem, 5vw, 4rem);
      }

      .chip-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 20px;
      }

      .chip {
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(255, 245, 216, 0.86);
        border: 1px solid rgba(82, 51, 24, 0.12);
        color: var(--ink-soft);
        font-size: 0.95rem;
      }

      .interpretation-frame {
        margin-top: 24px;
        border-radius: 32px;
        padding: 18px;
        background:
          linear-gradient(180deg, rgba(201, 148, 84, 0.22), rgba(255, 245, 220, 0.88)),
          var(--panel-strong);
      }

      .interpretation-media {
        border-radius: 22px;
        overflow: hidden;
        background: #704024;
      }

      .interpretation-media video {
        display: block;
        width: 100%;
        aspect-ratio: 4 / 5;
        object-fit: cover;
      }

      .archive-cta {
        margin-top: 18px;
      }

      @media (max-width: 900px) {
        .hero,
        .section-grid,
        .archive-card {
          grid-template-columns: 1fr;
        }

        .feed-status {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .hero-copy,
        .hero-feed,
        .section-panel,
        .archive-card {
          border-radius: 28px;
        }

        .shell {
          width: min(100vw - 20px, 1120px);
          padding-top: 14px;
        }

        .topbar {
          flex-direction: column;
          align-items: flex-start;
        }
      }

      @media (max-width: 560px) {
        h1 {
          font-size: 3.4rem;
        }

        .hero-copy,
        .section-panel,
        .archive-card {
          padding: 22px;
        }

        .step {
          grid-template-columns: 1fr;
        }

        .feed-status {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <header class="topbar">
        <div class="brand">Hasta Lab</div>
        <nav class="nav">
          <a href="#live">Live Feed</a>
          <a href="#about">About</a>
          <a href="#how">How To Use</a>
          <a href="#archive">Archive</a>
        </nav>
      </header>

      <section class="hero" id="live">
        <div class="hero-copy">
          <div class="eyebrow">Live Recognition Portal</div>
          <h1>{{ content.hero.title }}</h1>
          <p><strong>{{ content.hero.subtitle }}</strong></p>
          <p>{{ content.hero.body }}</p>
          <div class="hero-actions">
            <a class="button" href="#archive">Explore Hastas</a>
            <a class="button secondary" href="#how">How To Use</a>
          </div>
        </div>

        <div class="hero-feed">
          <div class="hero-feed-head">
            <span>Detection Stage</span>
            <span>Browser View</span>
          </div>
          <div class="feed-stage">
            <img id="live-feed" src="{{ video_url }}" alt="Hasta detection video feed">
          </div>
          <p class="feed-note">
            The camera feed mirrors the live viewer styling from the Python application while the surrounding page follows the archive-inspired visual language from the reference boards.
          </p>
          <div class="feed-status">
            <div class="status-card">
              <span class="status-label">Detected Hasta</span>
              <div id="status-label" class="status-value">No hand detected</div>
            </div>
            <div class="status-card">
              <span class="status-label">Confidence</span>
              <div id="status-confidence" class="status-value">--</div>
            </div>
            <div class="status-card">
              <span class="status-label">Hold Progress</span>
              <div id="status-progress" class="status-value">0%</div>
            </div>
            <div class="status-card">
              <span class="status-label">Viewer State</span>
              <div id="status-viewer" class="status-value">Live</div>
            </div>
          </div>
        </div>
      </section>

      <section class="section" id="about">
        <div class="section-mark">About Hasta Lab</div>
        <div class="section-grid" style="margin-top: 18px;">
          {% for panel in content.about %}
          <article class="section-panel">
            <div class="eyebrow">Context</div>
            <h3>{{ panel.heading }}</h3>
            <p>{{ panel.body }}</p>
          </article>
          {% endfor %}
        </div>
      </section>

      <section class="section" id="how">
        <div class="section-mark">How To Use</div>
        <div class="steps" style="margin-top: 18px;">
          {% for step in content.steps %}
          <article class="step">
            <div class="step-index">{{ loop.index }}</div>
            <p>{{ step }}</p>
          </article>
          {% endfor %}
        </div>
      </section>

      <section class="section" id="archive">
        <div class="section-mark">Interpretations</div>
        <div class="archive" style="margin-top: 18px;">
          {% for item in content.archive %}
          <article id="archive-{{ item.slug }}" class="archive-card">
            <div class="archive-sketch">
              <div class="archive-sketch-box">
                {% if item.sketch %}
                <img src="/media/{{ item.sketch }}" alt="{{ item.name }} sketch">
                {% else %}
                <div>No sketch available</div>
                {% endif %}
              </div>
              <a class="button archive-cta" href="#live">{{ item.cta }}</a>
            </div>

            <div class="archive-copy">
              <div class="eyebrow">Hasta Archive</div>
              <h2>{{ item.name }}</h2>
              <div class="chip-grid">
                {% for label in item.interpretations %}
                <span class="chip">{{ label }}</span>
                {% endfor %}
              </div>

              {% if item.video %}
              <div class="interpretation-frame">
                <div class="interpretation-media">
                  <video controls playsinline preload="metadata">
                    <source src="/media/{{ item.video }}">
                  </video>
                </div>
              </div>
              {% endif %}
            </div>
          </article>
          {% endfor %}
        </div>
      </section>
    </div>
    <script>
      const liveFeed = document.getElementById("live-feed");
      const statusLabel = document.getElementById("status-label");
      const statusConfidence = document.getElementById("status-confidence");
      const statusProgress = document.getElementById("status-progress");
      const statusViewer = document.getElementById("status-viewer");
      let lastScrolledArchive = null;

      async function resumeFeed() {
        try {
          const response = await fetch("/resume", { method: "POST" });
          if (!response.ok) {
            throw new Error("resume failed");
          }
          lastScrolledArchive = null;
          const nextUrl = "{{ video_url }}?t=" + Date.now();
          liveFeed.src = "";
          requestAnimationFrame(() => {
            liveFeed.src = nextUrl;
          });
        } catch (_error) {
        }
      }

      async function syncDetectionState() {
        try {
          const response = await fetch("/state", { cache: "no-store" });
          if (!response.ok) {
            return;
          }
          const state = await response.json();
          statusLabel.textContent = state.current_label || "No hand detected";
          statusConfidence.textContent = state.current_confidence || "--";
          statusProgress.textContent = state.hold_progress || "0%";
          statusViewer.textContent = state.paused ? "Paused" : "Live";
          if (!state.archive_slug) {
            return;
          }
          if (state.archive_slug === lastScrolledArchive) {
            return;
          }
          const target = document.getElementById("archive-" + state.archive_slug);
          if (!target) {
            return;
          }
          lastScrolledArchive = state.archive_slug;
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        } catch (_error) {
        }
      }

      liveFeed.addEventListener("click", resumeFeed);
      window.setInterval(syncDetectionState, 400);
    </script>
  </body>
</html>
"""


class BrowserVideoFeed:
    def __init__(
        self,
        camera_id: int,
        model_path: Path,
        classifier_path: Path,
        labels_path: Path,
        classification_threshold: float,
        hold_duration: float,
        num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        self.capture = cv2.VideoCapture(camera_id)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open webcam {camera_id}.")

        self.classifier = LiveMudraClassifier(
            classifier_path,
            labels_path,
            classification_threshold,
        )
        self.lock = threading.Lock()
        self.state_changed = threading.Condition(self.lock)
        self.hold_sm = HoldStateMachine(hold_duration)
        self.paused = False
        self.pause_pending = False
        self.last_frame_bytes: bytes | None = None
        self.last_frame = None
        self.completed_label: str | None = None
        self.current_label: str | None = None
        self.current_confidence: float | None = None
        self.current_progress = 0.0

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def close(self) -> None:
        with self.state_changed:
            if self.capture.isOpened():
                self.capture.release()
            self.landmarker.close()
            self.state_changed.notify_all()

    def resume(self) -> None:
        with self.state_changed:
            self.paused = False
            self.pause_pending = False
            self.last_frame_bytes = None
            self.last_frame = None
            self.completed_label = None
            self.current_label = None
            self.current_confidence = None
            self.current_progress = 0.0
            self.hold_sm.reset()
            self.state_changed.notify_all()

    def is_paused(self) -> bool:
        with self.lock:
            return self.paused

    def wait_until_resumed(self, timeout: float = 0.5) -> bool:
        with self.state_changed:
            if not self.paused:
                return True
            self.state_changed.wait(timeout=timeout)
            return not self.paused

    def get_state(self) -> dict[str, str | bool | None]:
        with self.lock:
            display_label = (
                _display_hasta_name(_browser_visible_label(self.completed_label))
                if _browser_visible_label(self.completed_label) is not None
                else None
            )
            visible_current_label = _browser_visible_label(self.current_label)
            current_label = (
                _display_hasta_name(visible_current_label)
                if visible_current_label is not None
                else None
            )
            archive_slug = (
                _archive_slug(display_label)
                if display_label is not None
                else None
            )
            return {
                "paused": self.paused,
                "completed_label": display_label,
                "current_label": current_label,
                "current_confidence": (
                    f"{self.current_confidence:.0%}"
                    if self.current_confidence is not None
                    else None
                ),
                "hold_progress": f"{self.current_progress:.0%}",
                "archive_slug": archive_slug,
            }

    def next_frame_bytes(self) -> bytes:
        with self.lock:
            if self.paused and self.last_frame_bytes is not None:
                return self.last_frame_bytes

            if self.pause_pending and self.last_frame is not None:
                paused_frame = self.last_frame.copy()
                self._draw_paused_overlay(paused_frame)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    paused_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                )
                if not ok:
                    raise RuntimeError("Failed to encode paused webcam frame.")
                self.last_frame_bytes = encoded.tobytes()
                self.paused = True
                self.pause_pending = False
                return self.last_frame_bytes

            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError("Failed to read from webcam.")

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            prediction = self.classifier.predict(result)

            progress = 0.0
            paused = False

            if prediction is not None:
                progress, paused, detected_label = self.hold_sm.update(prediction.label)
                draw_hand_landmarks(frame, result)
                draw_progress_circle(frame, result, progress, paused)
                self._draw_overlay(frame, prediction.label,
                                   prediction.confidence, progress)
                visible_label = _browser_visible_label(prediction.label)
                if visible_label is not None:
                    self.current_label = visible_label
                    self.current_confidence = prediction.confidence
                self.current_progress = (
                    progress if self.current_label is not None else 0.0
                )
                if paused and _browser_visible_label(detected_label) is not None:
                    self.completed_label = detected_label
            else:
                self.hold_sm.reset()
                self.completed_label = None
                self.current_label = None
                self.current_confidence = None
                self.current_progress = 0.0
                self._draw_overlay(frame, "no hand", None, progress)

            self.last_frame = frame.copy()

            ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise RuntimeError("Failed to encode webcam frame.")
            frame_bytes = encoded.tobytes()
            if paused:
                self.pause_pending = True
            elif not self.paused:
                self.pause_pending = False
            return frame_bytes

    @staticmethod
    def _draw_overlay(
        frame,
        label: str,
        confidence: float | None,
        progress: float,
    ) -> None:
        cv2.rectangle(frame, (18, 18),
                      (frame.shape[1] - 18, frame.shape[0] - 18), OVERLAY_LINE, 1)
        del label, confidence, progress

    @staticmethod
    def _draw_paused_overlay(frame) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start a local browser server for the live hasta detection feed."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
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
    parser.add_argument(
        "--hold-duration",
        type=float,
        default=5.0,
        help="Seconds the same hasta must be held to complete the progress border.",
    )
    return parser


def create_app(args: argparse.Namespace) -> Flask:
    app = Flask(__name__)
    feed = BrowserVideoFeed(
        camera_id=args.camera_id,
        model_path=args.model_path,
        classifier_path=args.classifier_path,
        labels_path=args.labels_path,
        classification_threshold=args.classification_threshold,
        hold_duration=args.hold_duration,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    atexit.register(feed.close)

    @app.get("/")
    def index() -> str:
        archive_items = []
        for item in PAGE_CONTENT["archive"]:
            sketch_name = item["sketch"] if (
                ASSETS_DIR / item["sketch"]).exists() else None
            video_name = item["video"] if (
                ASSETS_DIR / item["video"]).exists() else None
            archive_items.append(
                {
                    **item,
                    "slug": _archive_slug(item["name"]),
                    "sketch": sketch_name,
                    "video": video_name,
                }
            )

        content = {**PAGE_CONTENT, "archive": archive_items}
        return render_template_string(
            HTML_TEMPLATE,
            video_url="/video_feed",
            content=content,
        )

    @app.get("/media/<path:filename>")
    def media(filename: str):
        target = ASSETS_DIR / filename
        if not target.is_file():
            abort(404)
        return send_from_directory(ASSETS_DIR, filename)

    @app.get("/video_feed")
    def video_feed() -> Response:
        def generate() -> Iterator[bytes]:
            while True:
                frame_bytes = feed.next_frame_bytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
                while feed.is_paused():
                    feed.wait_until_resumed()

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/resume")
    def resume() -> Response:
        feed.resume()
        return Response(status=204)

    @app.get("/state")
    def state():
        return jsonify(feed.get_state())

    return app


def main() -> int:
    args = build_parser().parse_args()
    app = create_app(args)
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
