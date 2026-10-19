# Run instructions

Run the steps in order from the project folder. Steps 1-5 only need Python;
steps 6-8 also need Docker Desktop, minikube and kubectl.

## 1. Setup (once)

```bash
git clone https://github.com/Tirumala-devi-damarla/heart-disease-mlops.git
cd heart-disease-mlops
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Get and clean the data

```bash
bash scripts/download_data.sh
```

## 3. EDA plots

```bash
PYTHONPATH=src python -m heart.eda
```

Plots are saved in `reports/figures/`.

## 4. Train models (MLflow)

```bash
PYTHONPATH=src python -m heart.train
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open http://localhost:5000 to see the runs. Press Ctrl+C to stop the UI.

## 5. Tests

```bash
flake8 src api tests scripts
pytest -v
```

## 6. Docker

```bash
docker build -t heart-api:1.0.0 .
docker run -d --name heart-api -p 8080:8000 heart-api:1.0.0
curl -X POST localhost:8080/predict -H "Content-Type: application/json" -d @sample_request.json
```

API docs: http://localhost:8080/docs. Stop it with `docker rm -f heart-api`.

## 7. Kubernetes (minikube)

```bash
minikube start -p heart-mlops --driver=docker --cpus=2 --memory=3072
minikube -p heart-mlops addons enable ingress
minikube -p heart-mlops addons enable metrics-server
minikube -p heart-mlops image load heart-api:1.0.0
kubectl apply -f k8s/
kubectl apply -f k8s/monitoring/
kubectl -n heart-mlops get all
```

Call the API (keep this terminal open; it asks for your Mac password):

```bash
minikube -p heart-mlops tunnel
```

Then in another terminal:

```bash
curl -X POST http://127.0.0.1/predict -H "Content-Type: application/json" -d @sample_request.json
```

## 8. Monitoring

```bash
kubectl -n heart-mlops port-forward svc/grafana 3301:3000
```

Open http://localhost:3301 (login admin / admin). To see data on the dashboard, send some requests
from another terminal:

```bash
python scripts/generate_traffic.py --url http://127.0.0.1 -n 200
```

API logs: `kubectl -n heart-mlops logs deploy/heart-api --tail=20`

## CI/CD

Every push to `main` runs `.github/workflows/ci.yml` on GitHub: lint → tests → training → Docker
build and test. Results are in the repository's **Actions** tab.
