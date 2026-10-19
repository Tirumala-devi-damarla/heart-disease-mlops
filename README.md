# Heart Disease Risk Prediction - MLOps Assignment 1

[![CI/CD](https://github.com/Tirumala-devi-damarla/heart-disease-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Tirumala-devi-damarla/heart-disease-mlops/actions/workflows/ci.yml)

Predicts heart disease from patient data (UCI Heart Disease dataset) and serves the model as an
API on Kubernetes, with MLflow tracking, CI/CD and Prometheus/Grafana monitoring.

- **How to run everything:** [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md)
- **Report:** [reports/MLOps_Assignment1_Report.docx](reports/MLOps_Assignment1_Report.docx) (PDF in the same folder)
- **Screenshots:** [screenshots/](screenshots/)

## Result

Best model: **Logistic Regression**, CV ROC-AUC **0.907**, test ROC-AUC **0.960**.

## Folders

| Folder | Contents |
|---|---|
| `data/` | raw and cleaned dataset |
| `src/heart/` | data cleaning, EDA, features, training code |
| `notebooks/` | EDA, training and inference notebooks |
| `models/` | saved model |
| `api/` | FastAPI app |
| `tests/` | unit tests |
| `k8s/` | Kubernetes files (app + monitoring) |
| `monitoring/` | Prometheus and Grafana config |
| `.github/workflows/` | CI/CD pipeline |
| `reports/`, `screenshots/` | report, figures and screenshots |
