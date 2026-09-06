# SkyGuard AI console -- single container, ML pipeline and web UI together.
# Works on Hugging Face Spaces (Docker SDK), Render, Railway, Fly.io, or any
# host that runs a container and sets $PORT.
FROM python:3.12-slim

# scikit-learn / numpy wheels are self-contained; no system build tools needed.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SKYGUARD_SPEED=8

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[api]"

# Hugging Face Spaces runs as uid 1000 and expects port 7860; other hosts pass
# their own $PORT. Default covers both.
ENV PORT=7860
EXPOSE 7860
USER 1000

CMD ["sh", "-c", "uvicorn skyguard.api.asgi:app --host 0.0.0.0 --port ${PORT:-7860} --log-level warning"]
