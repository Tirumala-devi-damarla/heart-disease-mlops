# Serving image for the heart disease risk API
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src/ src/
COPY api/ api/
COPY models/model.joblib models/metadata.json models/

# run as a non-root user
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER 1000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
