import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from heart import config
from heart.features import build_pipeline

SAMPLE = {"age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260,
          "fbs": 0, "restecg": 2, "thalach": 130, "exang": 1,
          "oldpeak": 2.4, "slope": 2, "ca": 2, "thal": 7}


@pytest.fixture
def client(synthetic_clean, tmp_path, monkeypatch):
    from api import main

    X, y = synthetic_clean[config.FEATURES], synthetic_clean[config.TARGET]
    model_path = tmp_path / "model.joblib"
    joblib.dump(build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y),
                model_path)
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    monkeypatch.setattr(main, "METADATA_PATH", tmp_path / "metadata.json")
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert 0 <= body["probability_disease"] <= 1
    assert body["risk_level"] in ("low", "moderate", "high")
    assert "x-request-id" in r.headers


def test_predict_without_optional_fields(client):
    data = {k: v for k, v in SAMPLE.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=data).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("age", -1), ("cp", 9), ("chol", "high"), ("thal", 5)])
def test_predict_rejects_invalid_input(client, field, value):
    r = client.post("/predict", json={**SAMPLE, field: value})
    assert r.status_code == 422


def test_predict_missing_required_field(client):
    data = dict(SAMPLE)
    data.pop("age")
    assert client.post("/predict", json=data).status_code == 422


def test_batch_predict(client):
    r = client.post("/predict/batch", json={"patients": [SAMPLE, SAMPLE]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_metrics_endpoint(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text


def test_service_unavailable_without_model(tmp_path, monkeypatch):
    from api import main
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setitem(main.state, "model", None)
    with TestClient(main.app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=SAMPLE).status_code == 503
