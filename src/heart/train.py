"""Train, tune and compare classifiers with MLflow experiment tracking.

For every model family a GridSearchCV is run (5-fold stratified CV, ROC-AUC).
Each grid candidate is logged as a nested MLflow run, and the parent run logs
the best parameters, cross-validated metrics, hold-out test metrics, plots and
the fitted pipeline. The model with the best CV ROC-AUC is exported to
models/ for serving.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,  # noqa: E402
                             RocCurveDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_validate, train_test_split)

from heart import config  # noqa: E402
from heart.data import load_clean, split_xy  # noqa: E402
from heart.features import build_pipeline, get_feature_names  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Search spaces kept small on purpose: only 303 rows, so big grids overfit CV
MODEL_SPECS = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, solver="liblinear",
                                        random_state=config.RANDOM_STATE),
        "grid": {
            "model__C": [0.01, 0.1, 0.5, 1.0, 5.0],
            "model__penalty": ["l1", "l2"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=config.RANDOM_STATE,
                                            n_jobs=-1),
        "grid": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [3, 5, None],
            "model__min_samples_leaf": [1, 4],
        },
    },
    "gradient_boosting": {
        "estimator": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
        "grid": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
    },
}


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def save_plots(model, X_test, y_test, name, out_dir) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["No disease", "Disease"],
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {name}")
    paths.append(_save(fig, out_dir / f"confusion_matrix_{name}.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(f"ROC curve - {name}")
    paths.append(_save(fig, out_dir / f"roc_curve_{name}.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax,
                                          name=name)
    ax.set_title(f"Precision-recall - {name}")
    paths.append(_save(fig, out_dir / f"pr_curve_{name}.png"))

    importance = feature_importance(model)
    if importance is not None:
        top = importance.head(15)[::-1]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top.index, top.values, color="#457b9d")
        ax.set_title(f"Top features - {name}")
        paths.append(_save(fig, out_dir / f"feature_importance_{name}.png"))
    return paths


def feature_importance(model):
    est = model.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        values = np.abs(est.coef_[0])
    else:
        return None
    return pd.Series(values, index=get_feature_names(model)).sort_values(
        ascending=False)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def train_one(name, spec, X_train, y_train, X_test, y_test, cv, plot_dir):
    """Tune one model family and log everything to MLflow."""
    with mlflow.start_run(run_name=name) as parent:
        mlflow.set_tags({"model_family": name, "stage": "tuning"})
        search = GridSearchCV(build_pipeline(spec["estimator"]), spec["grid"],
                              scoring="roc_auc", cv=cv, n_jobs=-1,
                              return_train_score=True)
        start = time.time()
        search.fit(X_train, y_train)
        train_time = time.time() - start

        # one child run per grid candidate -> visible in the MLflow UI
        results = pd.DataFrame(search.cv_results_)
        for i, row in results.iterrows():
            with mlflow.start_run(run_name=f"{name}_cand{i:02d}", nested=True):
                mlflow.log_params({k.replace("model__", ""): v
                                   for k, v in row["params"].items()})
                mlflow.log_metrics({
                    "cv_roc_auc_mean": row["mean_test_score"],
                    "cv_roc_auc_std": row["std_test_score"],
                    "train_roc_auc_mean": row["mean_train_score"],
                })

        best = search.best_estimator_
        mlflow.log_params({k.replace("model__", "best_"): v
                           for k, v in search.best_params_.items()})
        mlflow.log_params({"cv_folds": cv.get_n_splits(),
                           "train_rows": len(X_train),
                           "test_rows": len(X_test),
                           "n_candidates": len(results)})

        cv_scores = cross_validate(best, X_train, y_train, cv=cv,
                                   scoring=SCORING)
        cv_metrics = {}
        for metric in SCORING:
            scores = cv_scores[f"test_{metric}"]
            cv_metrics[f"cv_{metric}_mean"] = float(scores.mean())
            cv_metrics[f"cv_{metric}_std"] = float(scores.std())
        test_metrics = {f"test_{k}": v
                        for k, v in evaluate(best, X_test, y_test).items()}
        mlflow.log_metrics({**cv_metrics, **test_metrics,
                            "tuning_time_sec": train_time})

        for path in save_plots(best, X_test, y_test, name, plot_dir):
            mlflow.log_artifact(str(path), artifact_path="plots")
        results.to_csv(plot_dir / f"cv_results_{name}.csv", index=False)
        mlflow.log_artifact(str(plot_dir / f"cv_results_{name}.csv"),
                            artifact_path="cv_results")

        signature = infer_signature(X_train.astype(float),
                                    best.predict_proba(X_train))
        mlflow.sklearn.log_model(best, artifact_path="model",
                                 signature=signature,
                                 input_example=X_train.head(3))

        logger.info("%-20s cv_roc_auc=%.4f test_roc_auc=%.4f best=%s", name,
                    cv_metrics["cv_roc_auc_mean"], test_metrics["test_roc_auc"],
                    search.best_params_)
        return {
            "name": name,
            "run_id": parent.info.run_id,
            "model": best,
            "best_params": {k.replace("model__", ""): v
                            for k, v in search.best_params_.items()},
            **cv_metrics,
            **test_metrics,
        }


def export_best(best: dict, X_train) -> None:
    """Save the winning pipeline as joblib + MLflow model + metadata."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], config.MODEL_PATH)

    mlflow_dir = config.MODEL_DIR / "mlflow_model"
    if mlflow_dir.exists():
        import shutil
        shutil.rmtree(mlflow_dir)
    mlflow.sklearn.save_model(
        best["model"], str(mlflow_dir),
        signature=infer_signature(X_train.astype(float),
                                  best["model"].predict_proba(X_train)),
        input_example=X_train.head(3))

    metadata = {
        "model_name": best["name"],
        "mlflow_run_id": best["run_id"],
        "best_params": best["best_params"],
        "metrics": {k: round(v, 4) for k, v in best.items()
                    if k.startswith(("cv_", "test_"))},
        "features": config.FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (config.MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str))
    logger.info("Exported %s to %s", best["name"], config.MODEL_PATH)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="heart-disease-classifier")
    parser.add_argument("--tracking-uri", default=None,
                        help="defaults to ./mlruns (local file store)")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS),
                        choices=list(MODEL_SPECS))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    mlflow.set_tracking_uri(args.tracking_uri
                            or (config.ROOT_DIR / "mlruns").as_uri())
    mlflow.set_experiment(args.experiment)

    df = load_clean()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=config.RANDOM_STATE)
    plot_dir = config.ROOT_DIR / "reports" / "training"
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = [train_one(name, MODEL_SPECS[name], X_train, y_train, X_test,
                         y_test, cv, plot_dir) for name in args.models]

    table = pd.DataFrame([{k: v for k, v in r.items() if k != "model"}
                          for r in results]).set_index("name")
    table.to_csv(plot_dir / "model_comparison.csv")
    logger.info("Model comparison:\n%s", table[[
        "cv_accuracy_mean", "cv_precision_mean", "cv_recall_mean",
        "cv_roc_auc_mean", "test_accuracy", "test_roc_auc"]].round(4))

    best = max(results, key=lambda r: r["cv_roc_auc_mean"])
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("selected_for_serving", "true")
    export_best(best, X_train)
    return table


if __name__ == "__main__":
    main()
