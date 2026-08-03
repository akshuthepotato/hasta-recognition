import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm";
const MODEL_ASSET_PATH = "./assets/rasa/face_landmarker.task";
const CSV_PATH = "./assets/rasa/landmarks.csv";
const HOLD_DURATION_MS = 3000;
const MAX_GAP_MS = 600;
const MAX_MISMATCH_MS = 600;
const MATCH_THRESHOLD = 0.5;

const video = document.getElementById("webcam");
const pausedFrame = document.getElementById("pausedFrame");
const overlay = document.getElementById("overlay");
const feedOverlay = document.getElementById("feedOverlay");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const rasaName = document.getElementById("rasaName");
const rasaMeaning = document.getElementById("rasaMeaning");
const rasaIllustration = document.getElementById("rasaIllustration");
const expressionProgress = document.getElementById("expressionProgress");
const rasaScore = document.getElementById("rasaScore");
const status = document.getElementById("status");
const pausedContext = pausedFrame.getContext("2d");
const overlayContext = overlay.getContext("2d");

let landmarker;
let matcher;
let stream;
let running = false;
let frameId;
let lastVideoTime = -1;
let latestResult;
let hold = newHold();

const RASA_ILLUSTRATIONS = {
  Adbutham: "./assets/rasa/illustrations/adbutham.png",
  Bhayanakam: "./assets/rasa/illustrations/bhayanakam.png",
  Hasyam: "./assets/rasa/illustrations/hasyam.png",
  Shantham: "./assets/rasa/illustrations/shantham.png",
};
const RASA_MEANINGS = {
  Adbutham: "Surprise",
  Bhayanakam: "Fear",
  Hasyam: "Laughter",
  Shantham: "Peace",
};
const preloadedRasaImages = new Map(Object.entries(RASA_ILLUSTRATIONS).map(
  ([label, src]) => {
    const image = new Image();
    image.src = src;
    return [label, image];
  },
));
let currentRasaIllustration = "Hasyam";

function newHold() {
  return {
    label: null,
    startedAt: null,
    lastSeenAt: null,
    mismatchAt: null,
    paused: false,
  };
}

function updateHold(state, label, now) {
  if (state.label === null) {
    if (label) {
      state.label = label;
      state.startedAt = now;
      state.lastSeenAt = now;
    }
    return 0;
  }

  if (label === state.label) {
    state.lastSeenAt = now;
    state.mismatchAt = null;
  } else if (label) {
    state.mismatchAt ??= now;
    if (now - state.mismatchAt > MAX_MISMATCH_MS) {
      Object.assign(state, newHold(), {
        label,
        startedAt: now,
        lastSeenAt: now,
      });
      return 0;
    }
  } else if (now - state.lastSeenAt > MAX_GAP_MS) {
    Object.assign(state, newHold());
    return 0;
  }

  const progress = Math.min((now - state.startedAt) / HOLD_DURATION_MS, 1);
  state.paused = progress === 1;
  return progress;
}

function cosine(a, b) {
  let total = 0;
  for (let index = 0; index < a.length; index += 1) total += a[index] * b[index];
  return total;
}

function normalize(vector) {
  const norm = Math.hypot(...vector);
  if (norm === 0) throw new Error("Blendshape vector has zero magnitude.");
  return vector.map((value) => value / norm);
}

function parseCsv(text) {
  return text.trim().split(/\r?\n/).map((line) => line.split(","));
}

function buildMatcher(rows) {
  const [headers, ...data] = rows;
  const labelIndex = headers.indexOf("label");
  const featureIndexes = headers
    .map((name, index) => ({ name, index }))
    .filter(({ name }) => name !== "image_path" && name !== "label");

  if (labelIndex === -1 || featureIndexes.length === 0) {
    throw new Error("Rasa CSV is missing labels or blendshape columns.");
  }

  const examples = new Map();
  for (const row of data) {
    const label = row[labelIndex]?.trim();
    if (!label) continue;
    const vector = featureIndexes.map(({ index }) => Number(row[index]));
    if (vector.some((value) => !Number.isFinite(value))) {
      throw new Error(`Invalid blendshape value for ${label}.`);
    }
    examples.set(label, [...(examples.get(label) || []), vector]);
  }

  const labels = [...examples.keys()];
  const centroids = labels.map((label) => {
    const vectors = examples.get(label);
    const centroid = featureIndexes.map((_, featureIndex) =>
      vectors.reduce((sum, vector) => sum + vector[featureIndex], 0) /
        vectors.length,
    );
    return normalize(centroid);
  });
  const features = featureIndexes.map(({ name }) => name);

  return {
    match(scores) {
      const vector = normalize(features.map((name) => scores[name] ?? 0));
      let best = { label: null, similarity: -Infinity };
      for (let index = 0; index < labels.length; index += 1) {
        const similarity = cosine(centroids[index], vector);
        if (similarity > best.similarity) {
          best = { label: labels[index], similarity };
        }
      }
      return best;
    },
  };
}

// ponytail: one browser-runnable check covers centroid cosine matching.
{
  const check = buildMatcher([
    ["image_path", "label", "a", "b"],
    ["one", "A", "1", "0"],
    ["two", "B", "0", "1"],
  ]);
  console.assert(check.match({ a: 0.9, b: 0.1 }).label === "A");
}

function setStatus(message) {
  status.textContent = message;
}

function resizeCanvases() {
  if (!video.videoWidth || !video.videoHeight) return false;
  for (const canvas of [pausedFrame, overlay]) {
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
  }
  return true;
}

function captureFrame() {
  if (!resizeCanvases()) return;
  pausedContext.drawImage(video, 0, 0, pausedFrame.width, pausedFrame.height);
  pausedFrame.classList.add("is-visible");
}

function showProgress(progress) {
  const yellow = [239, 195, 63];
  const green = [34, 197, 94];
  const greenRamp = Math.max(0, (progress - 0.8) / 0.2);
  const color = yellow.map((channel, index) =>
    Math.round(channel + (green[index] - channel) * greenRamp),
  );
  expressionProgress.textContent = `${Math.round(progress * 100)}%`;
  expressionProgress.style.color = `rgb(${color.join(", ")})`;
}

function renderOverlay() {
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  const landmarks = latestResult?.faceLandmarks?.[0];
  if (!landmarks) return;

  const bounds = landmarks.reduce((box, point) => ({
    minX: Math.min(box.minX, point.x),
    minY: Math.min(box.minY, point.y),
    maxX: Math.max(box.maxX, point.x),
    maxY: Math.max(box.maxY, point.y),
  }), { minX: 1, minY: 1, maxX: 0, maxY: 0 });
  const padding = 0.035;
  const x = (bounds.minX - padding) * overlay.width;
  const y = (bounds.minY - padding) * overlay.height;
  const width = (bounds.maxX - bounds.minX + padding * 2) * overlay.width;
  const height = (bounds.maxY - bounds.minY + padding * 2) * overlay.height;

  overlayContext.strokeStyle = "rgba(239, 195, 63, 0.72)";
  overlayContext.fillStyle = "rgba(239, 195, 63, 0.08)";
  overlayContext.lineWidth = 3;
  overlayContext.fillRect(x, y, width, height);
  overlayContext.strokeRect(x, y, width, height);
}

function applyPrediction(prediction) {
  const wasPaused = hold.paused;
  const known = prediction && prediction.similarity >= MATCH_THRESHOLD;
  const progress = updateHold(hold, known ? prediction.label : null, performance.now());
  const shownLabel = prediction?.label;

  showProgress(progress);
  rasaName.textContent = known
    ? prediction.label
    : "Unknown";
  rasaMeaning.textContent = known
    ? RASA_MEANINGS[prediction.label] || ""
    : prediction ? "Low confidence" : "No face detected";
  rasaScore.textContent = typeof prediction?.similarity === "number"
    ? `${(prediction.similarity * 100).toFixed(1)}%`
    : "Waiting";
  if (RASA_ILLUSTRATIONS[shownLabel] && shownLabel !== currentRasaIllustration) {
    currentRasaIllustration = shownLabel;
    rasaIllustration.src =
      preloadedRasaImages.get(shownLabel)?.src || RASA_ILLUSTRATIONS[shownLabel];
    rasaIllustration.alt = `${shownLabel} facial expression illustration`;
  }
  feedOverlay.classList.toggle("is-visible", hold.paused);
  renderOverlay();

  if (hold.paused && !wasPaused) captureFrame();
}

function resetHold() {
  hold = newHold();
  showProgress(0);
  feedOverlay.classList.remove("is-visible");
  pausedFrame.classList.remove("is-visible");
  pausedContext.clearRect(0, 0, pausedFrame.width, pausedFrame.height);
}

async function loadMatcher() {
  const response = await fetch(CSV_PATH);
  if (!response.ok) throw new Error(`Cannot load ${CSV_PATH}`);
  matcher = buildMatcher(parseCsv(await response.text()));
}

async function loadLandmarker() {
  setStatus("Loading facial expression recognition.");
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_ASSET_PATH },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
  });
}

async function enableAssetCache() {
  if (!("serviceWorker" in navigator)) return;
  await navigator.serviceWorker.register("./service-worker.js");
  await navigator.serviceWorker.ready;
}

function scoresFrom(result) {
  const blendshapes = result.faceBlendshapes?.[0]?.categories;
  if (!blendshapes?.length) return null;
  return Object.fromEntries(blendshapes.map(({ categoryName, score }) => [
    categoryName,
    score,
  ]));
}

function predict() {
  if (!running) return;
  if (!hold.paused && resizeCanvases() && video.currentTime !== lastVideoTime) {
    latestResult = landmarker.detectForVideo(video, performance.now());
    const scores = scoresFrom(latestResult);
    applyPrediction(scores ? matcher.match(scores) : null);
    lastVideoTime = video.currentTime;
  }
  frameId = requestAnimationFrame(predict);
}

async function start() {
  try {
    setStatus("Requesting camera access.");
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    const videoReady = new Promise((resolve) =>
      video.addEventListener("loadeddata", resolve, { once: true }),
    );
    video.srcObject = stream;
    await videoReady;
    cameraPlaceholder.hidden = true;

    await enableAssetCache();
    await loadMatcher();
    await loadLandmarker();
    running = true;
    predict();
    setStatus("Camera running. Show an expression to the webcam.");
  } catch (error) {
    setStatus(`Unable to start rasa recognition: ${error.message || error}`);
  }
}

feedOverlay.addEventListener("click", () => {
  resetHold();
  rasaName.textContent = "Unknown";
  rasaMeaning.textContent = "No face detected";
  currentRasaIllustration = "Hasyam";
  rasaIllustration.src = RASA_ILLUSTRATIONS.Hasyam;
  rasaIllustration.alt = "Hasyam facial expression illustration";
  rasaScore.textContent = "Waiting";
  renderOverlay();
  setStatus("Camera running. Show an expression to the webcam.");
});

window.addEventListener("beforeunload", () => {
  running = false;
  cancelAnimationFrame(frameId);
  stream?.getTracks().forEach((track) => track.stop());
  landmarker?.close();
});

showProgress(0);
void start();
