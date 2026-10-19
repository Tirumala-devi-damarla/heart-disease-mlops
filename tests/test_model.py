import json

import joblib
import mlflow
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from heart import config, train
from heart.features import build_pipeline


@pytest.mark.parametrize("model", [
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(n_estimators=30, random_state=0),
])
def test_pipeline_trains_and_predicts(synthetic_clean, model):
    X, y = synthetic_clean[config.FEATURES], synthetic_clean[config.TARGET]
    pipe = build_pipeline(model).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)
    # synthetic target is learnable, so training accuracy should be decent
    assert (pipe.predict(X) == y).mean() > 0.7


def test_evaluate_returns_all_metrics(synthetic_clean):
    X, y = synthetic_clean[config.FEATURES], synthetic_clean[config.TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    metrics = train.evaluate(pipe, X, y)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= v <= 1 for v in metrics.values())


def test_model_roundtrip_with_joblib(synthetic_clean, tmp_path):
    X, y = synthetic_clean[config.FEATURES], synthetic_clean[config.TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    path = tmp_path / "m.joblib"
    joblib.dump(pipe, path)
    loaded = joblib.load(path)
    assert np.allclose(loaded.predict_proba(X), pipe.predict_proba(X))


def test_training_main_end_to_end(synthetic_clean, tmp_path, monkeypatch):
    """Runs the real training entry point (one model) into a temp folder."""
    data_path = tmp_path / "clean.csv"
    synthetic_clean.to_csv(data_path, index=False)
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models" / "model.joblib")
    monkeypatch.setattr(train, "load_clean", lambda: synthetic_clean)

    table = train.main(["--models", "logistic_regression",
                        "--tracking-uri", (tmp_path / "mlruns").as_uri(),
                        "--experiment", "pytest"])
    assert "logistic_regression" in table.index
    assert (tmp_path / "models" / "model.joblib").exists()
    meta = json.loads((tmp_path / "models" / "metadata.json").read_text())
    assert meta["model_name"] == "logistic_regression"

    runs = mlflow.search_runs(experiment_names=["pytest"])
    assert len(runs) >= 2  # parent + nested candidate runs
    assert "metrics.cv_roc_auc_mean" in runs.columns
