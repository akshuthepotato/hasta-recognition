from __future__ import annotations

import argparse
import atexit
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import mediapipe as mp
from flask import Flask, Response, abort, render_template_string, send_from_directory
from flask_sock import Sock

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
        "intro": (
            "A beginner friendly learning portal of single hand gestures and "
            "their expanded meanings."
        ),
        "body": (
            "Bharatanatyam's single-hand gestures (Asamyuta Hastas) form a "
            "codified visual language through which meaning is constructed, "
            "communicated, and embodied. Traditionally, these gestures are "
            "defined through Vinyogams, canonical descriptions that map each "
            "hand form to a set of spiritual, mythological, or symbolic "
            "associations."
        ),
    },
    "how_to": [
        "Approach the interface and perform a hasta.",
        "Wait for the system to detect the gesture and give you visual feedback.",
        "Explore performing all the expanded meanings of the hand gesture.",
        "Explore the opportunity to add your own interpretation of the hand gesture.",
        "The interface then unfolds into a living archive of interpretations.",
    ],
    "archive": [
        {
            "name": "Pathaakam",
            "asset_dir": "PATHAAKAM",
            "sketch": "sketch.png",
            "video": "MIRROR.MOV",
            "cta": "Back to Home Page",
        },
        {
            "name": "Hamsasyam",
            "asset_dir": "hamsasyam",
            "sketch": "hamsasyam sketch.png",
            "video": "saying_ok.MOV",
            "cta": "Back to Home Page",
        },
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


def _asset_path(*parts: str) -> str:
    return str(Path(*parts).as_posix())


def _interpretation_label(filename: str) -> str:
    stem = Path(filename).stem
    return stem.replace("_", " ").strip().title()


def _archive_item_with_assets(item: dict[str, object]) -> dict[str, object]:
    asset_dir = Path(str(item["asset_dir"]))
    asset_root = ASSETS_DIR / asset_dir

    sketch_value = item.get("sketch")
    sketch_name = None
    if sketch_value:
        candidate = asset_root / str(sketch_value)
        if candidate.is_file():
            sketch_name = _asset_path(str(asset_dir), str(sketch_value))

    interpretation_files = sorted(
        path.name
        for path in asset_root.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".mov", ".mp4", ".m4v", ".webm"}
    )
    video_value = item.get("video")
    video_name = None
    if video_value:
        candidate = asset_root / str(video_value)
        if candidate.is_file():
            video_name = _asset_path(str(asset_dir), str(video_value))
    if video_name is None and interpretation_files:
        video_name = _asset_path(str(asset_dir), interpretation_files[0])

    media_items = [
        {
            "label": _interpretation_label(name),
            "path": _asset_path(str(asset_dir), name),
            "selected": _asset_path(str(asset_dir), name) == video_name,
        }
        for name in interpretation_files
    ]

    return {
        **item,
        "slug": _archive_slug(str(item["name"])),
        "sketch": sketch_name,
        "video": video_name,
        "interpretations": [media["label"] for media in media_items],
        "media_items": media_items,
    }


HOW_TO_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>How To Use | Hasta Lab</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Jura:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      :root {
        --ink: #4d2f1e;
        --ink-soft: #755845;
        --panel: rgba(250, 243, 233, 0.72);
        --border: rgba(123, 86, 56, 0.18);
        --button: #d8a07a;
        --button-text: #4d2f1e;
        --cta-height: 56px;
        --cta-radius: 999px;
        --cta-font-size: 1rem;
        --cta-letter-spacing: 0.04em;
        --shadow: 0 24px 80px rgba(93, 60, 34, 0.14);
        --pastel-yellow: rgba(247, 228, 168, 0.4);
        --pastel-green: rgba(194, 214, 187, 0.32);
        --pastel-orange: rgba(239, 196, 151, 0.32);
        --pastel-pink: rgba(232, 194, 193, 0.28);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        color: var(--ink);
        background:
          radial-gradient(circle at 12% 14%, var(--pastel-yellow), transparent 24%),
          radial-gradient(circle at 84% 18%, var(--pastel-pink), transparent 26%),
          radial-gradient(circle at 18% 78%, var(--pastel-green), transparent 24%),
          radial-gradient(circle at 88% 82%, var(--pastel-orange), transparent 28%),
          linear-gradient(180deg, #f7eddf 0%, #f0e1cd 52%, #ebd9c4 100%);
        font-family: "Jura", sans-serif;
      }

      .shell {
        width: min(980px, calc(100vw - 32px));
        margin: 0 auto;
        padding: 24px 0 64px;
      }

      .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding-bottom: 20px;
      }

      .brand {
        font-size: 1rem;
        letter-spacing: 0.24em;
        text-transform: uppercase;
        color: var(--ink-soft);
      }

      .nav {
        display: grid;
        grid-template-columns: repeat(3, minmax(170px, 1fr));
        gap: 12px;
      }

      .nav a,
      .button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: var(--cta-height);
        padding: 0 24px;
        border-radius: var(--cta-radius);
        border: 1px solid rgba(53, 26, 8, 0.18);
        background: rgba(255, 247, 223, 0.7);
        color: var(--ink);
        text-transform: uppercase;
        letter-spacing: var(--cta-letter-spacing);
        font-size: var(--cta-font-size);
        text-align: center;
        text-decoration: none;
      }

      .button {
        background: var(--button);
        color: var(--button-text);
        box-shadow: 0 14px 30px rgba(59, 30, 11, 0.18);
      }

      .page-card {
        padding: 28px;
        border-radius: 34px;
        border: 1px solid var(--border);
        background:
          linear-gradient(180deg, rgba(255, 246, 230, 0.9), rgba(241, 227, 204, 0.76)),
          var(--panel);
        box-shadow: var(--shadow);
        backdrop-filter: blur(6px);
      }

      .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        color: var(--ink-soft);
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.78rem;
      }

      .eyebrow::before {
        content: "";
        width: 18px;
        height: 2px;
        background: currentColor;
      }

      h1 {
        margin: 16px 0 0;
        font-size: clamp(2.5rem, 7vw, 4.6rem);
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }

      .lede {
        margin: 18px 0 0;
        max-width: 46rem;
        color: var(--ink-soft);
        line-height: 1.7;
        font-size: 1rem;
      }

      .steps {
        display: grid;
        gap: 14px;
        margin-top: 28px;
      }

      .step {
        display: grid;
        grid-template-columns: 56px 1fr;
        gap: 16px;
        align-items: start;
        padding: 20px;
        border-radius: 24px;
        border: 1px solid var(--border);
        background: rgba(255, 248, 237, 0.72);
      }

      .step-index {
        width: 56px;
        height: 56px;
        border-radius: 18px;
        display: grid;
        place-items: center;
        background: rgba(60, 30, 11, 0.08);
        border: 1px solid rgba(60, 30, 11, 0.12);
        font-size: 1.2rem;
      }

      .step p {
        margin: 0;
        color: var(--ink-soft);
        line-height: 1.7;
        font-size: 1rem;
      }

      .page-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        margin-top: 28px;
      }

      @media (max-width: 640px) {
        .shell {
          width: min(100vw - 20px, 980px);
          padding-top: 16px;
        }

        .topbar {
          flex-direction: column;
          align-items: flex-start;
        }

        .nav {
          width: 100%;
          grid-template-columns: 1fr;
        }

        .page-card {
          padding: 22px;
          border-radius: 28px;
        }

        .step {
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
          <a href="/">Live Feed</a>
          <a href="/how-to">How To Use</a>
          <a href="/#archive">Archive</a>
        </nav>
      </header>

      <main class="page-card">
        <div class="eyebrow">How To Use</div>
        <h1>Learn The Flow</h1>
        <p class="lede">
          Hasta Lab invites you to move from gesture to meaning, then from meaning to your own interpretation.
        </p>

        <section class="steps">
          {% for step in content.how_to %}
          <article class="step">
            <div class="step-index">{{ loop.index }}</div>
            <p>{{ step }}</p>
          </article>
          {% endfor %}
        </section>

        <div class="page-actions">
          <a class="button" href="/">Back to Home Page</a>
        </div>
      </main>
    </div>
  </body>
</html>
"""


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Hasta Detection</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Jura:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      :root {
        --paper: #f3e6d2;
        --paper-soft: rgba(244, 230, 208, 0.9);
        --paper-deep: #dcc2a0;
        --ink: #4d2f1e;
        --ink-soft: #755845;
        --panel: rgba(250, 243, 233, 0.68);
        --panel-strong: rgba(247, 236, 221, 0.82);
        --border: rgba(123, 86, 56, 0.18);
        --button: #d8a07a;
        --button-text: #4d2f1e;
        --cta-height: 56px;
        --cta-radius: 999px;
        --cta-font-size: 1rem;
        --cta-letter-spacing: 0.04em;
        --shadow: 0 24px 80px rgba(93, 60, 34, 0.14);
        --pastel-yellow: rgba(247, 228, 168, 0.4);
        --pastel-green: rgba(194, 214, 187, 0.32);
        --pastel-orange: rgba(239, 196, 151, 0.32);
        --pastel-pink: rgba(232, 194, 193, 0.28);
        --pastel-brown: rgba(188, 157, 130, 0.24);
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
          radial-gradient(circle at 12% 14%, var(--pastel-yellow), transparent 24%),
          radial-gradient(circle at 84% 18%, var(--pastel-pink), transparent 26%),
          radial-gradient(circle at 18% 78%, var(--pastel-green), transparent 24%),
          radial-gradient(circle at 88% 82%, var(--pastel-orange), transparent 28%),
          linear-gradient(180deg, #f7eddf 0%, #f0e1cd 52%, #ebd9c4 100%);
        font-family: "Jura", sans-serif;
      }

      body::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.15;
        background:
          radial-gradient(circle at 20% 20%, rgba(120, 86, 39, 0.08), transparent 18%),
          radial-gradient(circle at 80% 0%, rgba(120, 86, 39, 0.06), transparent 20%),
          radial-gradient(circle at 50% 100%, rgba(120, 86, 39, 0.08), transparent 22%),
          repeating-linear-gradient(
            135deg,
            rgba(116, 85, 58, 0.035) 0,
            rgba(116, 85, 58, 0.035) 2px,
            transparent 2px,
            transparent 9px
          );
        mix-blend-mode: multiply;
      }

      a {
        color: inherit;
        text-decoration: none;
      }

      .shell {
        width: min(1280px, calc(100vw - 32px));
        margin: 0 auto;
        padding: 20px 0 80px;
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
        display: grid;
        grid-template-columns: repeat(3, minmax(170px, 1fr));
        gap: 12px;
      }

      .nav a {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: var(--cta-height);
        padding: 0 24px;
        border-radius: var(--cta-radius);
        background: rgba(255, 247, 223, 0.6);
        border: 1px solid var(--border);
        font-size: var(--cta-font-size);
        letter-spacing: var(--cta-letter-spacing);
        text-transform: uppercase;
        text-align: center;
      }

      .hero {
        display: grid;
        grid-template-columns: 1fr;
        gap: 20px;
        align-items: stretch;
        min-height: auto;
        padding: 8px 0 32px;
      }

      .hero-copy,
      .hero-feed,
      .archive-card,
      .interpretation-frame {
        backdrop-filter: blur(6px);
        background: var(--panel);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
      }

      .hero-copy {
        border-radius: 34px;
        padding: 24px 28px;
        background:
          linear-gradient(180deg, rgba(255, 246, 230, 0.9), rgba(241, 227, 204, 0.76)),
          var(--panel);
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        gap: 10px 18px;
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
        line-height: 0.98;
      }

      h1 {
        margin-top: 0;
        font-size: clamp(2.8rem, 6vw, 5.6rem);
        text-transform: uppercase;
        letter-spacing: 0.035em;
        overflow-wrap: anywhere;
      }

      .hero-subtitle {
        margin: 0;
        max-width: none;
        font-size: 1rem;
        line-height: 1.65;
        color: var(--ink-soft);
      }

      .hero-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        margin-top: 0;
      }

      .button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: var(--cta-height);
        padding: 0 26px;
        border-radius: var(--cta-radius);
        border: 1px solid rgba(53, 26, 8, 0.22);
        background: var(--button);
        color: var(--button-text);
        text-transform: uppercase;
        letter-spacing: var(--cta-letter-spacing);
        font-size: var(--cta-font-size);
        text-align: center;
        box-shadow: 0 14px 30px rgba(59, 30, 11, 0.18);
      }

      .button.secondary {
        background: rgba(216, 160, 122, 0.58);
        color: var(--ink);
      }

      .hero-feed {
        border-radius: 36px;
        padding: 16px;
        background:
          linear-gradient(180deg, rgba(141, 101, 67, 0.16), rgba(247, 238, 227, 0.84)),
          var(--panel-strong);
        display: flex;
        flex-direction: column;
      }

      .feed-stage {
        position: relative;
        overflow: hidden;
        border-radius: 30px;
        border: 1px solid rgba(63, 36, 14, 0.16);
        background: #1c150f;
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 1;
        min-height: 72vh;
      }

      .feed-stage img {
        display: block;
        width: 100%;
        height: 100%;
        object-fit: cover;
      }

      .feed-status {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin: 14px 4px 2px;
      }

      .status-card {
        padding: 13px 14px;
        border-radius: 18px;
        border: 1px solid rgba(82, 51, 24, 0.12);
        background:
          linear-gradient(180deg, rgba(255, 248, 239, 0.72), rgba(246, 233, 214, 0.7)),
          rgba(255, 247, 223, 0.66);
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
        padding: 18px 0 28px;
      }

      .archive {
        display: grid;
        gap: 24px;
      }

      .archive-card {
        border-radius: 32px;
        padding: 24px;
        display: grid;
        grid-template-columns: minmax(260px, 360px) 1fr;
        gap: 24px;
        align-items: start;
      }

      .archive-heading {
        grid-column: 1 / -1;
      }

      .archive-heading h2 {
        margin-top: 12px;
        font-size: clamp(2.2rem, 4.4vw, 3.6rem);
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

      .archive-copy p {
        margin: 14px 0 0;
        color: var(--ink-soft);
        line-height: 1.65;
        font-size: 0.96rem;
      }

      .chip-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }

      .chip {
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(255, 245, 216, 0.86);
        border: 1px solid rgba(82, 51, 24, 0.12);
        color: var(--ink-soft);
        font-size: 0.95rem;
        cursor: pointer;
        transition: background-color 160ms ease, color 160ms ease, border-color 160ms ease, transform 160ms ease;
      }

      button.chip {
        font: inherit;
      }

      .chip:hover {
        transform: translateY(-1px);
        border-color: rgba(82, 51, 24, 0.28);
      }

      .chip.is-active {
        background: var(--button);
        color: var(--button-text);
        border-color: rgba(53, 26, 8, 0.3);
      }

      .interpretation-frame {
        margin-top: 16px;
        border-radius: 32px;
        padding: 18px;
        background:
          linear-gradient(180deg, rgba(221, 183, 146, 0.2), rgba(255, 247, 234, 0.84)),
          var(--panel-strong);
      }

      .interpretation-media {
        border-radius: 22px;
        overflow: hidden;
        background: #704024;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .interpretation-media video {
        display: block;
        width: 100%;
        height: auto;
        max-height: 70vh;
        object-fit: contain;
      }

      .archive-cta {
        margin-top: 18px;
      }

      @media (max-width: 900px) {
        .archive-card {
          grid-template-columns: 1fr;
        }

        .feed-status {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .hero-copy,
        .hero-feed,
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

        .nav {
          width: 100%;
          grid-template-columns: 1fr;
        }
      }

      @media (max-width: 560px) {
        h1 {
          font-size: 2.25rem;
          letter-spacing: 0.025em;
        }

        .hero-copy,
        .archive-card {
          padding: 20px;
        }

        .hero-copy {
          align-items: flex-start;
        }

        .feed-status {
          grid-template-columns: 1fr;
        }

        .feed-stage {
          min-height: 56vh;
        }

        .hero-subtitle {
          max-width: 100%;
        }

        .button {
          width: 100%;
          min-height: 54px;
          padding: 12px 20px;
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
          <a href="/how-to">How To Use</a>
          <a href="#archive">Archive</a>
        </nav>
      </header>

      <section class="hero" id="live">
        <div class="hero-copy">
          <h1>{{ content.hero.title }}</h1>
          <p class="hero-subtitle">{{ content.hero.intro }}</p>
        </div>

        <div class="hero-feed">
          <div class="feed-stage">
            <img id="live-feed" src="{{ video_url }}" alt="Hasta detection video feed">
          </div>
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
          </div>
        </div>

        <div class="hero-actions">
          <a class="button" href="#archive">Explore Hastas</a>
          <a class="button secondary" href="/how-to">Learn More About Hasta Lab</a>
        </div>
      </section>

      <section class="section" id="archive">
        <div class="section-mark">Hasta Archive</div>
        <div class="archive" style="margin-top: 18px;">
          {% for item in content.archive %}
          <article id="archive-{{ item.slug }}" class="archive-card">
            <div class="archive-heading">
              <h2>{{ item.name }}</h2>
              <p>Switch clips from the sketch side to view different interpretations.</p>
            </div>

            <div class="archive-sketch">
              <div class="archive-sketch-box">
                {% if item.sketch %}
                <img src="/media/{{ item.sketch }}" alt="{{ item.name }} sketch">
                {% else %}
                <div>No sketch available</div>
                {% endif %}
              </div>
              <div class="chip-grid">
                {% for media in item.media_items %}
                <button
                  type="button"
                  class="chip{% if media.selected %} is-active{% endif %}"
                  data-video-target="video-{{ item.slug }}"
                  data-video-src="/media/{{ media.path }}"
                >
                  {{ media.label }}
                </button>
                {% endfor %}
              </div>
              <a class="button archive-cta" href="/">{{ item.cta }}</a>
            </div>

            <div class="archive-copy">
              <div class="eyebrow">Interpretations</div>
              {% if item.video %}
              <div class="interpretation-frame">
                <div class="interpretation-media">
                  <video id="video-{{ item.slug }}" controls muted playsinline preload="metadata">
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
      let lastScrolledArchive = null;

      function applyDetectionState(state) {
        statusLabel.textContent = state.current_label || "No hand detected";
        statusConfidence.textContent = state.current_confidence || "--";
        statusProgress.textContent = state.hold_progress || "0%";
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
      }

      function connectDetectionSocket() {
        const protocol = window.location.protocol === "https:" ? "wss" : "ws";
        const socket = new WebSocket(protocol + "://" + window.location.host + "/ws/state");

        socket.addEventListener("message", (event) => {
          try {
            applyDetectionState(JSON.parse(event.data));
          } catch (_error) {
          }
        });

        socket.addEventListener("close", () => {
          window.setTimeout(connectDetectionSocket, 1000);
        });

        socket.addEventListener("error", () => {
          socket.close();
        });
      }

      function setupInterpretationChips() {
        const chips = document.querySelectorAll("[data-video-target][data-video-src]");
        for (const chip of chips) {
          chip.addEventListener("click", () => {
            const video = document.getElementById(chip.dataset.videoTarget);
            if (!video) {
              return;
            }

            const source = video.querySelector("source");
            const nextSrc = chip.dataset.videoSrc;
            if (!source || !nextSrc || source.getAttribute("src") === nextSrc) {
                return;
            }

            source.setAttribute("src", nextSrc);
            video.load();
            const playPromise = video.play();
            if (playPromise && typeof playPromise.catch === "function") {
              playPromise.catch(() => {});
            }

            const chipGroup = chip.closest(".chip-grid");
            if (!chipGroup) {
              return;
            }
            for (const groupChip of chipGroup.querySelectorAll(".chip")) {
              groupChip.classList.remove("is-active");
            }
            chip.classList.add("is-active");
          });
        }
      }

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

      liveFeed.addEventListener("click", resumeFeed);
      connectDetectionSocket();
      setupInterpretationChips();
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
        self.state_revision = 0

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
            self.state_revision += 1
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
            self.state_revision += 1
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

    def _state_payload_unlocked(self) -> dict[str, str | bool | None]:
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

    def get_state(self) -> dict[str, str | bool | None]:
        with self.lock:
            return self._state_payload_unlocked()

    def get_state_with_revision(self) -> tuple[int, dict[str, str | bool | None]]:
        with self.lock:
            return self.state_revision, self._state_payload_unlocked()

    def wait_for_state_change(
        self,
        last_revision: int,
        timeout: float = 15.0,
    ) -> tuple[int, dict[str, str | bool | None]]:
        with self.state_changed:
            if self.state_revision == last_revision:
                self.state_changed.wait(timeout=timeout)
            return self.state_revision, self._state_payload_unlocked()

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
                self.state_revision += 1
                self.state_changed.notify_all()
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
            self.state_revision += 1
            self.state_changed.notify_all()
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
    sock = Sock(app)
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
        archive_items = [
            _archive_item_with_assets(item) for item in PAGE_CONTENT["archive"]
        ]
        content = {**PAGE_CONTENT, "archive": archive_items}
        return render_template_string(
            HTML_TEMPLATE,
            video_url="/video_feed",
            content=content,
        )

    @app.get("/how-to")
    def how_to() -> str:
        return render_template_string(
            HOW_TO_TEMPLATE,
            content=PAGE_CONTENT,
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

    @sock.route("/ws/state")
    def state_socket(ws) -> None:
        revision, state = feed.get_state_with_revision()
        ws.send(json.dumps(state))
        while True:
            revision, state = feed.wait_for_state_change(revision)
            ws.send(json.dumps(state))

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
