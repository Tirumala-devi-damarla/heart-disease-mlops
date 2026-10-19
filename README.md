# Heart Disease Risk Prediction - End-to-End MLOps

MLOps Assignment 1 (AIMLCZG523). A classifier that predicts heart disease from patient data (UCI
Heart Disease, Cleveland subset), with experiment tracking, CI/CD, Docker, Kubernetes and monitoring.

> Full write-up: [`reports/MLOps_Assignment1_Report.docx`](reports/MLOps_Assignment1_Report.docx)
> (PDF copy in the same folder). Architecture: [`reports/figures/architecture.png`](reports/figures/architecture.png)

## Results at a glance

| Model | CV ROC-AUC (5-fold) | CV accuracy | Test ROC-AUC | Test accuracy |
|---|---|---|---|---|
| **Logistic Regression (selected)** | **0.907 ± 0.020** | **0.851** | 0.960 | 0.869 |
| Random Forest | 0.897 ± 0.028 | 0.818 | 0.958 | 0.869 |
| Gradient Boosting | 0.874 ± 0.025 | 0.818 | 0.959 | 0.902 |

## Repository layout

```
heart-disease-mlops/
├── data/raw/                     raw UCI file (processed.cleveland.data)
├── data/processed/               cleaned dataset heart_clean.csv
├── scripts/download_data.sh      download + clean the data
├── src/heart/                    python package
│   ├── config.py                 paths + column lists
│   ├── data.py                   download, clean, validate
│   ├── eda.py                    EDA plots -> reports/figures
│   ├── features.py               feature engineering + preprocessing Pipeline
│   └── train.py                  GridSearchCV + MLflow tracking + model export
├── notebooks/                    01_eda, 02_training_mlflow, 03_inference (executed)
├── models/                       model.joblib, metadata.json, mlflow_model/
├── api/                          FastAPI app (main.py, schemas.py)
├── tests/                        pytest unit tests (data, features, model, api)
├── Dockerfile                    serving image
├── docker-compose.yml            API + Prometheus + Grafana (local)
├── k8s/                          namespace, configmap, deployment, service, ingress, hpa
├── k8s/monitoring/               Prometheus + Grafana for the cluster
├── monitoring/                   Prometheus config + Grafana dashboard JSON
├── .github/workflows/ci.yml      lint -> test -> train -> docker build/smoke test
├── reports/                      report, figures, training plots
└── screenshots/                  screenshots for the report (+ raw logs in screenshots/logs)
```

## 1. Setup

Requires Python 3.12, Docker, and (for deployment) minikube + kubectl.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data

```bash
bash scripts/download_data.sh          # add --force to re-download
```

Downloads `processed.cleveland.data` from the UCI repository, replaces `?` with NaN, fills the 6
missing values (`ca`, `thal`) with the mode, converts `num` (0-4) into a binary `target`, validates
and writes `data/processed/heart_clean.csv`.

## 3. EDA

```bash
PYTHONPATH=src python -m heart.eda     # writes reports/figures/*.png
jupyter notebook notebooks/01_eda.ipynb
```

## 4. Training + MLflow

```bash
PYTHONPATH=src python -m heart.train
mlflow ui --backend-store-uri ./mlruns --port 5000     # http://localhost:5000
```

Three model families are tuned with `GridSearchCV` (5-fold stratified CV, ROC-AUC). Every grid
candidate is a nested MLflow run; the parent run stores best params, CV and test metrics, plots
(confusion matrix, ROC, PR, feature importance), the CV results table and the model. The model
with the best mean CV ROC-AUC is exported to `models/model.joblib` (full sklearn Pipeline including
preprocessing) and `models/mlflow_model/`.

Inference without the API:

```bash
python scripts/predict.py --json sample_request.json
```

## 5. Tests and lint

```bash
flake8 src api tests scripts
pytest -v
```

## 6. Docker

```bash
docker build -t heart-api:1.0.0 .
docker run -d --name heart-api -p 8080:8000 heart-api:1.0.0
curl -X POST localhost:8080/predict -H "Content-Type: application/json" -d @sample_request.json
# {"prediction":1,"label":"heart disease","confidence":0.9934,"probability_disease":0.9934,
#  "risk_level":"high","model_version":"logistic_regression-c800af5b"}
```

Swagger UI: http://localhost:8080/docs

| Endpoint | Description |
|---|---|
| `POST /predict` | one patient -> prediction, confidence, probability, risk level |
| `POST /predict/batch` | up to 100 patients |
| `GET /health` | liveness / readiness |
| `GET /model-info` | model name, run id, params, metrics |
| `GET /metrics` | Prometheus metrics |

## 7. Kubernetes (minikube)

```bash
minikube start -p heart-mlops --driver=docker --cpus=2 --memory=3072
minikube -p heart-mlops addons enable ingress
minikube -p heart-mlops addons enable metrics-server
minikube -p heart-mlops image load heart-api:1.0.0

kubectl apply -f k8s/
kubectl apply -f k8s/monitoring/
kubectl -n heart-mlops get all,ingress,hpa
```

Access the API:

```bash
# Option A - LoadBalancer (keep this terminal open, asks for sudo password)
minikube -p heart-mlops tunnel
curl http://127.0.0.1/health

# Option B - minikube service tunnel (prints a URL)
minikube -p heart-mlops service heart-api -n heart-mlops --url

# Option C - through the NGINX ingress
kubectl -n ingress-nginx port-forward svc/ingress-nginx-controller 8088:80
curl -H "Host: heart-api.local" http://127.0.0.1:8088/health
```

## 8. Monitoring

```bash
kubectl -n heart-mlops port-forward svc/prometheus 9095:9090   # http://localhost:9095
kubectl -n heart-mlops port-forward svc/grafana 3301:3000      # http://localhost:3301 (admin/admin)
python scripts/generate_traffic.py --url <api-url> -n 300       # generate some load
```

Without Kubernetes: `docker compose up -d --build` starts API (8080), Prometheus (9091) and
Grafana (3001).

The API writes one JSON log line per request (request id, path, status, latency) and per
prediction (input + output): `kubectl -n heart-mlops logs deploy/heart-api`.

## 9. CI/CD

`.github/workflows/ci.yml` runs on every push / PR:

1. **lint** - flake8 (report uploaded as artifact)
2. **test** - pytest with JUnit XML + log artifact
3. **train** - downloads data, trains all models, enforces a quality gate (CV ROC-AUC >= 0.80),
   uploads model, plots and `mlruns/` as artifacts
4. **docker** - builds the image, runs it, calls `/predict`, uploads container logs

Each job depends on the previous one, so any lint/test/training error stops the pipeline.

## Dataset citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI
Machine Learning Repository. https://doi.org/10.24432/C52P4X
