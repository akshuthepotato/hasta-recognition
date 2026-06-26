const CACHE_NAME = "natya-model-v2";
const CACHEABLE_ASSETS = [
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm/",
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
  new URL("./assets/rasa/", self.location.href).href,
];

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  if (!CACHEABLE_ASSETS.some((asset) => event.request.url.startsWith(asset))) return;

  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(event.request);
    if (cached) return cached;

    const response = await fetch(event.request);
    if (response.ok || response.type === "opaque") {
      await cache.put(event.request, response.clone());
    }
    return response;
  })());
});
