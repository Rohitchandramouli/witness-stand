FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build dossier artifacts at container startup (data/ is gitignored)
RUN python scripts/build_dossier.py || echo "Dossier build skipped (no API key at build time)"

EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
