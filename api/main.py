"""FastAPI service that serves the heart disease model.

Endpoints
  GET  /              basic info
  GET  /health        liveness / readiness probe
  GET  /model-info    metadata of the loaded model
  POST /predict       predict for one patient
  POST /predict/batch predict for up to 100 patients
  GET  /metrics       Prometheus metrics
"""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)

from api.schemas import (BatchRequest, BatchResponse, PatientData,
                         PredictionResponse)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so the pickled pipeline can import heart.*

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
METADATA_PATH = MODEL_PATH.with_name("metadata.json")
FEATURE_ORDER = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "sex",
                 "fbs", "exang", "cp", "restecg", "slope", "thal"]


# ---------- logging: one JSON line per event so it is easy to grep/ship ----------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("heart_api")
logger.handlers = [handler]
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
logger.propagate = False

# ---------- Prometheus metrics ----------
REQUEST_COUNT = Counter("api_requests_total", "Total HTTP requests",
                        ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds",
                            "Request latency in seconds", ["endpoint"],
                            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
                                     0.5, 1, 2.5))
PREDICTIONS = Counter("model_predictions_total", "Predictions by class",
                      ["label"])
DISEASE_PROBABILITY = Histogram("model_probability_disease",
                                "Predicted probability of disease",
                                buckets=[i / 10 for i in range(1, 11)])
MODEL_LOADED = Gauge("model_loaded", "1 if the model is loaded")

state = {"model": None, "metadata": {}}


def load_model():
    if not MODEL_PATH.exists():
        logger.error("Model file not found at %s", MODEL_PATH)
        MODEL_LOADED.set(0)
        return
    state["model"] = joblib.load(MODEL_PATH)
    if METADATA_PATH.exists():
        state["metadata"] = json.loads(METADATA_PATH.read_text())
    MODEL_LOADED.set(1)
    logger.info("Model loaded from %s (%s)", MODEL_PATH,
                state["metadata"].get("model_name", "unknown"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Heart Disease Risk API",
              description="Predicts heart disease risk from patient data "
                          "(UCI Cleveland dataset).",
              version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        endpoint = request.url.path
        if endpoint != "/metrics":
            REQUEST_COUNT.labels(request.method, endpoint, status).inc()
            REQUEST_LATENCY.labels(endpoint).observe(elapsed)
            logger.info("request", extra={"extra_fields": {
                "request_id": request_id,
                "method": request.method,
                "path": endpoint,
                "status": status,
                "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None,
            }})


def model_version() -> str:
    meta = state["metadata"]
    run = meta.get("mlflow_run_id", "")[:8]
    return f"{meta.get('model_name', 'model')}-{run}" if run else "unknown"


def risk_level(prob: float) -> str:
    if prob < 0.3:
        return "low"
    if prob < 0.6:
        return "moderate"
    return "high"


def predict_frame(df: pd.DataFrame) -> list:
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    probs = model.predict_proba(df[FEATURE_ORDER])[:, 1]
    results = []
    for p in probs:
        pred = int(p >= 0.5)
        label = "heart disease" if pred else "no heart disease"
        PREDICTIONS.labels(label).inc()
        DISEASE_PROBABILITY.observe(float(p))
        results.append(PredictionResponse(
            prediction=pred,
            label=label,
            confidence=round(float(p if pred else 1 - p), 4),
            probability_disease=round(float(p), 4),
            risk_level=risk_level(float(p)),
            model_version=model_version(),
        ))
    return results


@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "docs": "/docs",
            "predict": "POST /predict"}


@app.get("/health")
def health():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_version": model_version()}


@app.get("/model-info")
def model_info():
    return state["metadata"] or {"detail": "no metadata"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    df = pd.DataFrame([patient.model_dump()])
    result = predict_frame(df)[0]
    logger.info("prediction", extra={"extra_fields": {
        "input": patient.model_dump(), "output": result.model_dump()}})
    return result


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    df = pd.DataFrame([p.model_dump() for p in batch.patients])
    return BatchResponse(predictions=predict_frame(df))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
