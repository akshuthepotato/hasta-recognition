# Web Interface Demo

This directory contains a minimal browser demo for the MediaPipe Gesture
Recognizer Web API.

## Run

Start the websocket server from the repository root:

```bash
uv run python server/landmark_server.py
```

In a second terminal, start the static file server:

```bash
uv run python -m http.server 8000 --directory web-interface
```

Then open `http://127.0.0.1:8000`.

When you click `Start camera`, the browser connects to `ws://127.0.0.1:8765`
and streams detected hand landmarks. The Python server prints the normalized
`x`, `y`, and `z` coordinates for each landmark it receives.

To test on a phone, do not use your computer's LAN IP over plain HTTP. Mobile
browsers generally require HTTPS for camera access, so use HTTPS or a secure
tunnel when opening the page from another device.

## Notes

- The webcam stream stays in the browser unless you modify the app to upload it.
- The page loads the MediaPipe JavaScript bundle and gesture model from official
  CDN and Google storage URLs at runtime.
- Camera access generally requires `http://localhost`, `http://127.0.0.1`, or
  HTTPS.
