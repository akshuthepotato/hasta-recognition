import {
  DrawingUtils,
  FilesetResolver,
  GestureRecognizer,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm";
const MODEL_ASSET_PATH =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";
const WEBSOCKET_URL = "wss://hasta-recognition-holy-darkness-1170.fly.dev";
const HOLD_DURATION_MS = 3000;
const MAX_GAP_MS = 600;
const MAX_MISMATCH_MS = 600;

const video = document.getElementById("webcam");
const pausedFrame = document.getElementById("pausedFrame");
const overlay = document.getElementById("overlay");
const feedOverlay = document.getElementById("feedOverlay");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const gestureName = document.getElementById("gestureName");
const gestureScore = document.getElementById("gestureScore");
const holdProgress = document.getElementById("holdProgress");
const status = document.getElementById("status");
const pausedContext = pausedFrame.getContext("2d");
const overlayContext = overlay.getContext("2d");

let recognizer;
let socket;
let stream;
let running = false;
let frameId;
let lastVideoTime = -1;
let latestResult;
let lastPrediction;
let hold = newHold();

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
  const valid = Boolean(label) && label !== "uncertain";

  if (state.label === null) {
    if (valid) {
      state.label = label;
      state.startedAt = now;
      state.lastSeenAt = now;
    }
    return 0;
  }

  if (valid && label === state.label) {
    state.lastSeenAt = now;
    state.mismatchAt = null;
  } else if (valid) {
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

// ponytail: one browser-runnable check covers the hold state machine.
{
  const check = newHold();
  console.assert(updateHold(check, "pathakam", 0) === 0);
  console.assert(updateHold(check, "pathakam", 3000) === 1 && check.paused);
}

function setStatus(message) {
  status.textContent = message;
}

function resetHold() {
  hold = newHold();
  lastPrediction = null;
  showProgress(0);
  feedOverlay.classList.remove("is-visible");
  pausedFrame.classList.remove("is-visible");
  pausedContext.clearRect(0, 0, pausedFrame.width, pausedFrame.height);
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
  holdProgress.textContent = `${Math.round(progress * 100)}%`;
  holdProgress.style.color = `rgb(${color.join(", ")})`;
}

function renderOverlay() {
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  if (latestResult?.landmarks?.length) {
    const drawing = new DrawingUtils(overlayContext);
    for (const landmarks of latestResult.landmarks) {
      drawing.drawConnectors(landmarks, GestureRecognizer.HAND_CONNECTIONS, {
        color: "#e89ba4",
        lineWidth: 4,
      });
      drawing.drawLandmarks(landmarks, {
        color: "#efc33f",
        lineWidth: 2,
        radius: 4,
      });
    }
  }
}

function applyPrediction(prediction) {
  const progress = updateHold(hold, prediction?.label ?? null, performance.now());
  const valid = prediction?.label && prediction.label !== "uncertain";
  if (valid) lastPrediction = prediction;
  const shown = valid ? prediction : lastPrediction;

  showProgress(progress);
  gestureName.textContent =
    shown?.displayLabel || shown?.label || "No hand detected";
  gestureScore.textContent =
    typeof shown?.confidence === "number"
      ? `${(shown.confidence * 100).toFixed(1)}%`
      : "Waiting";
  feedOverlay.classList.toggle("is-visible", hold.paused);
  renderOverlay();

  if (hold.paused) captureFrame();
}

function connectSocket() {
  return new Promise((resolve, reject) => {
    socket = new WebSocket(WEBSOCKET_URL);
    socket.onopen = () => resolve();
    socket.onmessage = ({ data }) => {
      try {
        applyPrediction(JSON.parse(data));
      } catch (_error) {
        setStatus("The landmark server returned an invalid response.");
      }
    };
    socket.onerror = () => reject(new Error(`Cannot connect to ${WEBSOCKET_URL}`));
    socket.onclose = () => {
      if (running) setStatus("Landmark server disconnected.");
    };
  });
}

function sendLandmarks(result) {
  if (socket?.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({
    timestamp: new Date().toISOString(),
    hands: (result.landmarks || []).map((landmarks, index) => ({
      handedness: result.handedness?.[index]?.[0]?.categoryName || "Unknown",
      landmarks: landmarks.map(({ x, y, z }, pointIndex) => ({
        index: pointIndex,
        x,
        y,
        z,
      })),
    })),
  }));
}

async function loadRecognizer() {
  setStatus("Loading hand recognition.");
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
  recognizer = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_ASSET_PATH },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

async function enableAssetCache() {
  if (!("serviceWorker" in navigator)) return;
  await navigator.serviceWorker.register("./service-worker.js");
  await navigator.serviceWorker.ready;
}

function predict() {
  if (!running) return;
  if (!hold.paused && resizeCanvases() && video.currentTime !== lastVideoTime) {
    latestResult = recognizer.recognizeForVideo(video, performance.now());
    renderOverlay();
    sendLandmarks(latestResult);
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
    await loadRecognizer();
    running = true;
    predict();

    setStatus("Camera running. Connecting to landmark server.");
    try {
      await connectSocket();
      setStatus("Camera running. Show a hand to the webcam.");
    } catch (error) {
      setStatus(`Camera running without recognition: ${error.message || error}`);
    }
  } catch (error) {
    setStatus(`Unable to start gesture recognition: ${error.message || error}`);
  }
}

feedOverlay.addEventListener("click", () => {
  resetHold();
  gestureName.textContent = "No hand detected";
  gestureScore.textContent = "Waiting";
  renderOverlay();
  setStatus("Camera running. Show a hand to the webcam.");
});

window.addEventListener("beforeunload", () => {
  running = false;
  cancelAnimationFrame(frameId);
  stream?.getTracks().forEach((track) => track.stop());
  socket?.close();
});

showProgress(0);
void start();
