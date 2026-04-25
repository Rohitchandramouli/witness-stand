# The Witness Stand — HuggingFace Spaces Dockerfile
# Port 7860 is required by HuggingFace Spaces.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./

RUN uv python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

RUN uv pip install -e . \
    && mkdir -p data/personas logs/health logs/episodes logs/eval logs/training

# Do not build the dossier at Docker build time because HF secrets are not
# available during image build. Build it after deployment/startup when secrets
# are injected, or run scripts/00_build_dossier.py manually.
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
