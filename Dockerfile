# ── The Witness Stand — HuggingFace Spaces Dockerfile ──────────────────────
# Builds and serves the OpenEnv environment + dashboard.
# Port 7860 is required by HuggingFace Spaces.

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache — rebuild only when deps change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Install the package in editable mode so all imports resolve
RUN pip install --no-cache-dir -e .

# Create runtime directories (data/ and logs/ are gitignored but needed at runtime)
RUN mkdir -p data/personas logs/episodes logs/eval logs/training

# Build dossier artifacts at container startup.
# GROQ_API_KEY is not available at Docker build time — this runs once on cold start.
# If the build fails (no key), the container still starts; the /build endpoint
# in server/app.py can trigger a rebuild after secrets are injected via HF Space settings.
RUN python scripts/build_dossier.py || echo "[dossier] Skipped — GROQ_API_KEY not set at build time. Run /build after deployment."

# Expose HF Spaces port
EXPOSE 7860

# Health check — confirms the FastAPI app is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
