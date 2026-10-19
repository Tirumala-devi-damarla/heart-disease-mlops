# Demo video script (about 5-7 minutes)

Record the screen (macOS: Cmd+Shift+5 -> "Record entire screen") and talk through each step.
Keep the terminal font large. Prepare beforehand: cluster running, MLflow UI open, Grafana open.

| # | Time | Show | Say |
|---|---|---|---|
| 1 | 0:00 | README / architecture.png | Problem, dataset, the three parts of the architecture |
| 2 | 0:40 | `bash scripts/download_data.sh` | Download + cleaning, 6 missing values filled, class balance |
| 3 | 1:00 | `reports/figures/` or notebook 01 | 2-3 EDA findings (cp=4, thal=7, thalach lower for disease) |
| 4 | 1:40 | `PYTHONPATH=src python -m heart.train` | GridSearchCV, 3 models, 5-fold CV, best model chosen by CV ROC-AUC |
| 5 | 2:20 | MLflow UI (`mlflow ui --backend-store-uri ./mlruns`) | Parent + nested runs, compare view, plots in artifacts |
| 6 | 3:00 | `pytest -v` and `flake8 src api tests scripts` | 30 tests, what they cover |
| 7 | 3:30 | GitHub -> Actions tab | 4 jobs (lint -> test -> train -> docker), artifacts of the run |
| 8 | 4:10 | `docker build` + `docker run` + `curl /predict` | Prediction + confidence in JSON |
| 9 | 4:50 | `kubectl -n heart-mlops get all,ingress,hpa` | 2 replicas, LoadBalancer, Ingress, HPA |
| 10 | 5:20 | `minikube -p heart-mlops tunnel` + `curl http://127.0.0.1/predict` | API reachable through the cluster |
| 11 | 5:50 | `python scripts/generate_traffic.py --url ... -n 200` + Grafana | Live dashboard: request rate, latency, predictions by class |
| 12 | 6:30 | `kubectl -n heart-mlops logs deploy/heart-api --tail=5` | JSON request logs, wrap-up |

Useful commands:

```bash
kubectl -n heart-mlops port-forward svc/grafana 3301:3000      # http://localhost:3301
kubectl -n heart-mlops port-forward svc/prometheus 9095:9090   # http://localhost:9095
```
