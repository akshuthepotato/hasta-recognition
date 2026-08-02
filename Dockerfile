FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN uv pip install --system \
    "aiohttp>=3.13.5" \
    "numpy>=2.0.2" \
    "pymongo" \
    "websockets>=15.0.1" \
    "xgboost>=2.1.4" \
    "scikit-learn>=1.6.1"

COPY web-interface/server ./server

EXPOSE 8765

CMD ["python", "server/landmark_server.py", "--host", "0.0.0.0", "--port", "8765"]
