# syntax=docker/dockerfile:1

# Pinned base image: exact digest, so every build uses the same image
ARG PYTHON_IMAGE=python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
# ---------- Stage 1: builder (installs packages) ----------
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install -r /tmp/requirements.txt

# ---------- Stage 2: runtime (small, clean, non-root) ----------
FROM ${PYTHON_IMAGE} AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODEL_DIR=/app/models

# Normal user with no admin rights
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY models/ ./models/

USER appuser
EXPOSE 8000

# Docker calls /health to check the app is alive
# (slim has no curl, so we use Python)
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]