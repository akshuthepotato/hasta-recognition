import {
  DrawingUtils,
  FilesetResolver,
  GestureRecognizer,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const statusElement = document.getElementById("status");
const gestureNameElement = document.getElementById("gestureName");
const gestureScoreElement = document.getElementById("gestureScore");
const holdProgressElement = document.getElementById("holdProgress");
const archiveListElement = document.getElementById("archiveList");
const videoElement = document.getElementById("webcam");
const pausedFrameElement = document.getElementById("pausedFrame");
const canvasElement = document.getElementById("overlay");
const feedOverlayElement = document.getElementById("feedOverlay");
const canvasContext = canvasElement.getContext("2d");
const pausedFrameContext = pausedFrameElement.getContext("2d");

const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm";
const MODEL_ASSET_PATH =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";
// const WEBSOCKET_URL = `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname || "127.0.0.1"}:8765`;
const WEBSOCKET_URL = `https://hasta-recognition-holy-darkness-1170.fly.dev`;
const HOLD_DURATION_MS = 5000;
const MAX_GAP_MS = 600;
const MAX_MISMATCH_MS = 600;
const ARCHIVE_DEFINITIONS = [
  {
    slug: "pathakam",
    name: "Pathakam",
    performer: "Maya",
    sketch: "./assets/hasta_illustrations/1 illustration.png",
    mediaBasePath: "./assets/Maya/pathakam",
    mediaFiles: [
      "asking.mp4",
      "blessing.mp4",
      "calling.mp4",
      "combing hair.mp4",
      "directions.mp4",
      "embarrased.mp4",
      "fed up.mp4",
      "giggling.mp4",
      "go.mp4",
      "high five.mp4",
      "horse.mp4",
      "killing.mp4",
      "makeup.mp4",
      "me_mine.mp4",
      "mirror.mp4",
      "open and close.mp4",
      "partition.mp4",
      "patting.mp4",
      "petting.mp4",
      "push.mp4",
    ],
    selectedFile: "mirror.mp4",
  },
  {
    slug: "thripathakam",
    name: "Thripathakam",
    performer: "Maya",
    sketch: "./assets/hasta_illustrations/2 illustration.png",
    mediaBasePath: "./assets/Maya/thripathakam",
    mediaFiles: [
      "crocodile.mp4",
      "crown.mp4",
      "eyebrows.mp4",
      "fire pit.mp4",
      "fire.mp4",
      "kajal.mp4",
      "king.mp4",
      "lamp.mp4",
      "lightning.mp4",
      "moon rays.mp4",
    ],
  },
  {
    slug: "ardhapathakam",
    name: "Ardhapathakam",
    performer: "Priya",
    sketch: "./assets/hasta_illustrations/3 illustration.png",
    mediaBasePath: "./assets/Priya/ardhapathakam",
    mediaFiles: [
      "applying butter.mp4",
      "axe.mp4",
      "bumpy road.mp4",
      "flag.mp4",
      "flipping pages.mp4",
      "gun.mp4",
      "horns.mp4",
      "mountain.mp4",
      "sections.mp4",
      "wind.mp4",
    ],
  },
  {
    slug: "kartharimukham",
    name: "Kartharimukham",
    performer: "Priya",
    sketch: "./assets/hasta_illustrations/4 illustration.png",
    mediaBasePath: "./assets/Priya/kartharimukham",
    mediaFiles: [
      "after.mp4",
      "braiding.mp4",
      "break.mp4",
      "creeper.mp4",
      "die.mp4",
      "fury.mp4",
      "hair.mp4",
      "next.mp4",
      "rolling.mp4",
      "scissors.mp4",
      "sneaky.mp4",
    ],
    selectedFile: "scissors.mp4",
  },
  {
    slug: "ardhachandram",
    name: "Ardhachandram",
    performer: "Maya",
    sketch: "./assets/hasta_illustrations/5 illustration.png",
    mediaBasePath: "./assets/Maya/ardhachandram",
    mediaFiles: [
      "after you.mp4",
      "demanding.mp4",
      "feeling hot.mp4",
      "parts.mp4",
      "path.mp4",
      "road.mp4",
      "salute.mp4",
      "shade.mp4",
      "shy.mp4",
      "slapping.mp4",
      "sleeping.mp4",
      "slicing.mp4",
      "smelly.mp4",
      "stop.mp4",
      "sweeping.mp4",
      "taking a picture.mp4",
      "thilakam.mp4",
      "tired.mp4",
      "up and down.mp4",
      "using the phone.mp4",
      "waving.mp4",
      "wind.mp4",
      "wiping sweat.mp4",
      "wiping tears.mp4",
    ],
  },
  {
    slug: "mushti",
    name: "Mushti",
    performer: "Pujitha",
    sketch: "./assets/hasta_illustrations/6 illustration.png",
    mediaBasePath: "./assets/Pujitha/mushti",
    mediaFiles: [
      " grab handle.mp4",
      "PXL_20260414_122034266.TS.mp4",
      "armour.mp4",
      "brave.mp4",
      "carrying.mp4",
      "cheering.mp4",
      "courage.mp4",
      "defeated.mp4",
      "donating food.mp4",
      "hair.mp4",
      "holding hands.mp4",
      "holding someone.mp4",
      "knocking.mp4",
      "mixing.mp4",
      "ponytail.mp4",
      "punching.mp4",
      "squeezing.mp4",
      "strength.mp4",
      "suitcase.mp4",
      "washing clothes.mp4",
    ],
    labelOverrides: {
      " grab handle.mp4": "Grab Handle",
      "PXL_20260414_122034266.TS.mp4": "Demonstration",
    },
  },
  {
    slug: "shikaram",
    name: "Shikaram",
    performer: "Pujitha",
    sketch: "./assets/hasta_illustrations/7 illustration.png",
    mediaBasePath: "./assets/Pujitha/shikaram",
    mediaFiles: [
      "admiring.mp4",
      "applying tilak.mp4",
      "bow.mp4",
      "churning.mp4",
      "determined.mp4",
      "drinking.mp4",
      "god.mp4",
      "hugging.mp4",
      "lips.mp4",
      "man.mp4",
      "pouring.mp4",
      "questioning.mp4",
      "ringing bell.mp4",
      "teasing.mp4",
      "teeth.mp4",
      "yes.mp4",
    ],
  },
  {
    slug: "kapitham",
    name: "Kapitham",
    performer: "Pujitha",
    sketch: "./assets/hasta_illustrations/ 8 illustration.png",
    mediaBasePath: "./assets/Pujitha/kapitham",
    mediaFiles: [
      "eating spoon.mp4",
      "goddess.mp4",
      "kajal.mp4",
      "milking cow.mp4",
      "nattuvangam.mp4",
      "nudge.mp4",
      "picking up.mp4",
      "plucking fruits.mp4",
      "plucking.mp4",
      "prayer.mp4",
      "pull.mp4",
      "serving food.mp4",
      "small.mp4",
      "taking.mp4",
      "tearing paper.mp4",
      "thattukazhi.mp4",
      "veena instrument.mp4",
      "wearing veil.mp4",
      "writing.mp4",
    ],
  },
  {
    slug: "katakaamukham",
    name: "Katakaamukham",
    performer: "Priya",
    sketch: "./assets/hasta_illustrations/9 illustration.png",
    mediaBasePath: "./assets/Priya/katakaamukham",
    mediaFiles: [
      "clip.mp4",
      "decoration.mp4",
      "drawing.mp4",
      "eating.mp4",
      "fragrance.mp4",
      "garland.mp4",
      "giving.mp4",
      "kiss.mp4",
      "ornaments.mp4",
      "peeping.mp4",
      "pick up.mp4",
      "playing chess.mp4",
      "plucking flowers.mp4",
      "recieving.mp4",
      "tucking hair.mp4",
      "tying a sari.mp4",
      "washing drying clothes.mp4",
      "window.mp4",
    ],
  },
];

function toAssetUrl(path) {
  return encodeURI(path);
}

function mediaLabelFromFileName(fileName) {
  return fileName
    .replace(/\.[^.]+$/, "")
    .trim()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function buildMediaItems({
  mediaBasePath,
  mediaFiles,
  selectedFile,
  labelOverrides = {},
}) {
  return mediaFiles.map((fileName, index) => ({
    label: labelOverrides[fileName] ?? mediaLabelFromFileName(fileName),
    path: toAssetUrl(`${mediaBasePath}/${fileName}`),
    selected: selectedFile ? fileName === selectedFile : index === 0,
  }));
}

const ARCHIVE_ITEMS = ARCHIVE_DEFINITIONS.map((item) => ({
  ...item,
  sketch: toAssetUrl(item.sketch),
  cta: "Back to home page",
  mediaItems: buildMediaItems(item),
}));

let gestureRecognizer;
let running = false;
let animationFrameId = null;
let lastVideoTime = -1;
let stream = null;
let recognizerReadyPromise = null;
let landmarkSocket = null;
let holdState = createHoldState();
let lastStablePrediction = null;
let lastScrolledArchive = null;
let latestRecognitionResult = null;
let currentHoldVisual = { progress: 0, paused: false };

function createHoldState() {
  return {
    holdLabel: null,
    holdStartTime: null,
    paused: false,
    lastSeenTime: null,
    mismatchStartTime: null,
    completedLabel: null,
  };
}

function isSecureCameraContext() {
  return (
    window.isSecureContext ||
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
  );
}

function getUserMediaSupportError() {
  if (!isSecureCameraContext()) {
    return "Camera access requires HTTPS on phones, or localhost/127.0.0.1 during local development.";
  }

  return "This browser does not expose navigator.mediaDevices.getUserMedia.";
}

async function requestCameraStream(constraints) {
  if (navigator.mediaDevices?.getUserMedia) {
    return navigator.mediaDevices.getUserMedia(constraints);
  }

  const legacyGetUserMedia =
    navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia;

  if (legacyGetUserMedia) {
    return new Promise((resolve, reject) => {
      legacyGetUserMedia.call(navigator, constraints, resolve, reject);
    });
  }

  throw new Error(getUserMediaSupportError());
}

function setStatus(message) {
  statusElement.textContent = message;
}

function resetResults() {
  gestureNameElement.textContent = "No hand detected";
  gestureScoreElement.textContent = "Waiting";
  holdProgressElement.textContent = "0%";
  feedOverlayElement.classList.remove("is-visible");
  currentHoldVisual = { progress: 0, paused: false };
  clearPausedFrame();
  renderOverlay();
}

function resetHoldState() {
  holdState = createHoldState();
  lastStablePrediction = null;
  lastScrolledArchive = null;
}

function isValidLabel(label) {
  return Boolean(label) && label !== "uncertain";
}

function updateHoldState(label) {
  const now = performance.now();
  const valid = isValidLabel(label);

  if (holdState.holdLabel === null) {
    if (valid) {
      holdState.holdLabel = label;
      holdState.holdStartTime = now;
      holdState.lastSeenTime = now;
    }
    return { progress: 0, paused: false, detectedLabel: null };
  }

  if (valid && label === holdState.holdLabel) {
    holdState.lastSeenTime = now;
    holdState.mismatchStartTime = null;
  } else if (valid && label !== holdState.holdLabel) {
    if (holdState.mismatchStartTime === null) {
      holdState.mismatchStartTime = now;
    }

    if (now - holdState.mismatchStartTime > MAX_MISMATCH_MS) {
      holdState.holdLabel = label;
      holdState.holdStartTime = now;
      holdState.lastSeenTime = now;
      holdState.mismatchStartTime = null;
      holdState.paused = false;
      holdState.completedLabel = null;
      return { progress: 0, paused: false, detectedLabel: null };
    }
  } else if (
    holdState.lastSeenTime !== null &&
    now - holdState.lastSeenTime > MAX_GAP_MS
  ) {
    resetHoldState();
    return { progress: 0, paused: false, detectedLabel: null };
  }

  let progress = 0;
  if (holdState.holdStartTime !== null) {
    const elapsed = now - holdState.holdStartTime;
    progress = Math.min(elapsed / HOLD_DURATION_MS, 1);
    if (elapsed >= HOLD_DURATION_MS) {
      holdState.paused = true;
      holdState.completedLabel = holdState.holdLabel;
    }
  }

  return {
    progress,
    paused: holdState.paused,
    detectedLabel: holdState.holdLabel,
  };
}

function showHoldState({ progress, paused }) {
  currentHoldVisual = { progress, paused };
  holdProgressElement.textContent = `${Math.round(progress * 100)}%`;
  feedOverlayElement.classList.toggle("is-visible", Boolean(paused));
  renderOverlay();
}

function syncPausedFrameSize() {
  const width = videoElement.videoWidth;
  const height = videoElement.videoHeight;

  if (!width || !height) {
    return false;
  }

  if (
    pausedFrameElement.width !== width ||
    pausedFrameElement.height !== height
  ) {
    pausedFrameElement.width = width;
    pausedFrameElement.height = height;
  }

  return true;
}

function capturePausedFrame() {
  if (!syncPausedFrameSize()) {
    return;
  }

  pausedFrameContext.clearRect(
    0,
    0,
    pausedFrameElement.width,
    pausedFrameElement.height,
  );
  pausedFrameContext.drawImage(
    videoElement,
    0,
    0,
    pausedFrameElement.width,
    pausedFrameElement.height,
  );
  pausedFrameElement.classList.add("is-visible");
}

function clearPausedFrame() {
  pausedFrameContext.clearRect(
    0,
    0,
    pausedFrameElement.width,
    pausedFrameElement.height,
  );
  pausedFrameElement.classList.remove("is-visible");
}

function normalizeArchiveKey(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function archiveSlugForPrediction(prediction) {
  const candidates = [
    prediction?.displayLabel,
    prediction?.label,
    holdState.completedLabel,
    holdState.holdLabel,
  ];

  for (const candidate of candidates) {
    const slug = normalizeArchiveKey(candidate);
    if (!slug) {
      continue;
    }

    const target = document.getElementById(`archive-${slug}`);
    if (target) {
      return slug;
    }
  }

  return null;
}

function scrollToArchiveForPrediction(prediction) {
  const slug = archiveSlugForPrediction(prediction);
  if (!slug || slug === lastScrolledArchive) {
    return;
  }

  const target = document.getElementById(`archive-${slug}`);
  if (!target) {
    return;
  }

  lastScrolledArchive = slug;
  target.scrollIntoView({ behavior: "smooth", block: "start" });
}

function applyPrediction(prediction) {
  const label = prediction?.label ?? null;
  const holdResult = updateHoldState(label);
  showHoldState(holdResult);

  if (isValidLabel(label)) {
    lastStablePrediction = prediction;
  }

  if (holdResult.paused) {
    capturePausedFrame();
    scrollToArchiveForPrediction(prediction);
    gestureNameElement.textContent =
      prediction?.displayLabel ||
      lastStablePrediction?.displayLabel ||
      holdState.completedLabel ||
      "No hand detected";
    gestureScoreElement.textContent =
      typeof prediction?.confidence === "number"
        ? `${(prediction.confidence * 100).toFixed(1)}%`
        : typeof lastStablePrediction?.confidence === "number"
          ? `${(lastStablePrediction.confidence * 100).toFixed(1)}%`
          : "Waiting";
    return;
  }

  if (!isValidLabel(label)) {
    if (holdState.holdLabel !== null && lastStablePrediction !== null) {
      gestureNameElement.textContent =
        lastStablePrediction.displayLabel ||
        lastStablePrediction.label ||
        "No hand detected";
      gestureScoreElement.textContent =
        typeof lastStablePrediction.confidence === "number"
          ? `${(lastStablePrediction.confidence * 100).toFixed(1)}%`
          : "Waiting";
      return;
    }

    gestureNameElement.textContent = "No hand detected";
    gestureScoreElement.textContent = "Waiting";
    return;
  }

  gestureNameElement.textContent =
    prediction?.displayLabel || prediction?.label || "No hand detected";
  gestureScoreElement.textContent =
    typeof prediction?.confidence === "number"
      ? `${(prediction.confidence * 100).toFixed(1)}%`
      : "-";
}

function resumeHoldState() {
  resetHoldState();
  showHoldState({ progress: 0, paused: false });
  clearPausedFrame();
  gestureNameElement.textContent = "No hand detected";
  gestureScoreElement.textContent = "Waiting";
}

function resizeCanvasToVideo() {
  const width = videoElement.videoWidth;
  const height = videoElement.videoHeight;

  if (!width || !height) {
    return false;
  }

  if (canvasElement.width !== width || canvasElement.height !== height) {
    canvasElement.width = width;
    canvasElement.height = height;
  }

  return true;
}

function clearOverlay() {
  canvasContext.clearRect(0, 0, canvasElement.width, canvasElement.height);
}

function drawProgressBorder(progress, paused) {
  if (progress <= 0 && !paused) {
    return;
  }

  const width = canvasElement.width;
  const height = canvasElement.height;
  if (!width || !height) {
    return;
  }

  const margin = 10;
  const topLeft = { x: margin, y: margin };
  const topRight = { x: width - margin, y: margin };
  const bottomRight = { x: width - margin, y: height - margin };
  const bottomLeft = { x: margin, y: height - margin };

  const topLen = topRight.x - topLeft.x;
  const rightLen = bottomRight.y - topRight.y;
  const bottomLen = bottomRight.x - bottomLeft.x;
  const leftLen = bottomLeft.y - topLeft.y;
  const perimeter = topLen + rightLen + bottomLen + leftLen;
  let remaining = paused ? perimeter : Math.round(progress * perimeter);

  canvasContext.strokeStyle = paused ? "#22c55e" : "#facc15";
  canvasContext.lineWidth = 6;
  canvasContext.lineCap = "round";

  function drawSegment(start, end, segmentLength) {
    if (remaining <= 0) {
      return;
    }

    canvasContext.beginPath();
    canvasContext.moveTo(start.x, start.y);

    if (remaining >= segmentLength) {
      canvasContext.lineTo(end.x, end.y);
      canvasContext.stroke();
      remaining -= segmentLength;
      return;
    }

    const ratio = remaining / segmentLength;
    const x = start.x + (end.x - start.x) * ratio;
    const y = start.y + (end.y - start.y) * ratio;
    canvasContext.lineTo(x, y);
    canvasContext.stroke();
    remaining = 0;
  }

  drawSegment(topRight, topLeft, topLen);
  drawSegment(topLeft, bottomLeft, leftLen);
  drawSegment(bottomLeft, bottomRight, bottomLen);
  drawSegment(bottomRight, topRight, rightLen);
}

function renderOverlay() {
  clearOverlay();

  if (latestRecognitionResult?.landmarks?.length) {
    const drawingUtils = new DrawingUtils(canvasContext);

    for (const landmarks of latestRecognitionResult.landmarks) {
      drawingUtils.drawConnectors(
        landmarks,
        GestureRecognizer.HAND_CONNECTIONS,
        {
          color: "#60a5fa",
          lineWidth: 4,
        },
      );
      drawingUtils.drawLandmarks(landmarks, {
        color: "#f97316",
        lineWidth: 2,
        radius: 4,
      });
    }
  }

  drawProgressBorder(currentHoldVisual.progress, currentHoldVisual.paused);
}

function renderArchive() {
  if (!archiveListElement) {
    return;
  }

  archiveListElement.innerHTML = ARCHIVE_ITEMS.map((item) => {
    const selectedMedia =
      item.mediaItems.find((media) => media.selected) ?? item.mediaItems[0] ?? null;
    const chipsMarkup = item.mediaItems
      .map(
        (media) => `
          <button
            type="button"
            class="chip${media.selected ? " is-active" : ""}"
            data-video-target="video-${item.slug}"
            data-video-src="${media.path}"
          >
            ${media.label}
          </button>
        `,
      )
      .join("");

    return `
      <article id="archive-${item.slug}" class="archive-card">
        <div class="archive-heading">
          <h2>${item.name}</h2>
        </div>

        <div class="archive-sketch">
          <div class="archive-sketch-box">
            ${
              item.sketch
                ? `<img src="${item.sketch}" alt="${item.name} sketch">`
                : "<div>No sketch available</div>"
            }
          </div>
          <div class="chip-grid">${chipsMarkup}</div>
          <a class="button archive-cta" href="#live">${item.cta}</a>
        </div>

        <div class="archive-copy">
          <div class="eyebrow">Interpretations</div>
          ${
            selectedMedia
              ? `
                <div class="interpretation-frame">
                  <div class="interpretation-media">
                    <video
                      id="video-${item.slug}"
                      controls
                      muted
                      playsinline
                      preload="metadata"
                    >
                      <source src="${selectedMedia.path}">
                    </video>
                  </div>
                </div>
              `
              : ""
          }
          ${
            item.performer
              ? `<p class="performer-credit">Performed by ${item.performer}, Apsaras Dance Academy</p>`
              : ""
          }
        </div>
      </article>
    `;
  }).join("");

  setupInterpretationChips();
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

function drawResults(result) {
  latestRecognitionResult = result;
  renderOverlay();
}

function closeLandmarkSocket() {
  if (!landmarkSocket) {
    return;
  }

  landmarkSocket.onopen = null;
  landmarkSocket.onclose = null;
  landmarkSocket.onerror = null;
  landmarkSocket.close();
  landmarkSocket = null;
}

function connectLandmarkSocket() {
  return new Promise((resolve, reject) => {
    closeLandmarkSocket();

    const socket = new WebSocket(WEBSOCKET_URL);

    socket.onopen = () => {
      landmarkSocket = socket;
      setStatus("Camera running. Connected to landmark server.");
      resolve(socket);
    };

    socket.onmessage = (event) => {
      try {
        const prediction = JSON.parse(event.data);
        applyPrediction(prediction);
      } catch (_error) {
      }
    };

    socket.onerror = () => {
      reject(
        new Error(
          `Unable to connect to landmark server at ${WEBSOCKET_URL}. Start the Python server first.`,
        ),
      );
    };

    socket.onclose = () => {
      if (landmarkSocket === socket) {
        landmarkSocket = null;
      }

      if (running) {
        setStatus("Landmark server disconnected.");
      }
    };
  });
}

function sendLandmarks(result) {
  if (landmarkSocket?.readyState !== WebSocket.OPEN) {
    return;
  }

  const hands = (result.landmarks || []).map((landmarks, index) => ({
    handedness: result.handedness?.[index]?.[0]?.categoryName || "Unknown",
    landmarks: landmarks.map(({ x, y, z }, landmarkIndex) => ({
      index: landmarkIndex,
      x,
      y,
      z,
    })),
  }));

  landmarkSocket.send(
    JSON.stringify({
      timestamp: new Date().toISOString(),
      hands,
    }),
  );
}

async function ensureRecognizer() {
  if (gestureRecognizer) {
    return gestureRecognizer;
  }

  if (!recognizerReadyPromise) {
    recognizerReadyPromise = (async () => {
      setStatus("Loading MediaPipe gesture recognizer.");
      const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
      gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_ASSET_PATH,
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      setStatus("Recognizer ready. Starting camera.");
      return gestureRecognizer;
    })().catch((error) => {
      recognizerReadyPromise = null;
      throw error;
    });
  }

  return recognizerReadyPromise;
}

function stopStreamTracks() {
  if (!stream) {
    return;
  }

  for (const track of stream.getTracks()) {
    track.stop();
  }

  stream = null;
}

function stopCamera() {
  running = false;
  lastVideoTime = -1;

  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  stopStreamTracks();
  closeLandmarkSocket();
  videoElement.srcObject = null;
  clearOverlay();
  resetHoldState();
  resetResults();
  setStatus("Camera stopped.");
}

function predictWebcam() {
  if (!running || !gestureRecognizer) {
    return;
  }

  if (holdState.paused) {
    animationFrameId = window.requestAnimationFrame(predictWebcam);
    return;
  }

  if (resizeCanvasToVideo() && videoElement.currentTime !== lastVideoTime) {
    const nowInMs = performance.now();
    const result = gestureRecognizer.recognizeForVideo(videoElement, nowInMs);
    drawResults(result);
    sendLandmarks(result);
    lastVideoTime = videoElement.currentTime;
  }

  animationFrameId = window.requestAnimationFrame(predictWebcam);
}

async function startCamera() {
  if (running) {
    return;
  }

  try {
    await ensureRecognizer();
    setStatus("Connecting to landmark server.");
    await connectLandmarkSocket();
    setStatus("Requesting camera access.");

    stream = await requestCameraStream({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    videoElement.srcObject = stream;

    await new Promise((resolve) => {
      videoElement.onloadeddata = () => resolve();
    });

    running = true;
    lastVideoTime = -1;
    resetHoldState();
    resetResults();
    setStatus("Camera running. Show a hand to the webcam.");
    predictWebcam();
  } catch (error) {
    stopStreamTracks();
    closeLandmarkSocket();
    setStatus(
      `Unable to start gesture recognition: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

videoElement.addEventListener("click", () => {
  if (!holdState.paused) {
    return;
  }

  resumeHoldState();
  setStatus("Camera running. Show a hand to the webcam.");
});

window.addEventListener("beforeunload", () => {
  stopCamera();
});

renderArchive();
resetResults();

void startCamera().catch((error) => {
  setStatus(
    `Unable to start gesture recognition: ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
});
