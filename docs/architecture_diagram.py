"""Draws reports/figures/architecture.png (run: python docs/architecture_diagram.py)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "architecture.png"

fig, ax = plt.subplots(figsize=(16, 9.2))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9.2)
ax.axis("off")


def group(x, y, w, h, title, color, title_bottom=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                                fc=color, ec="#666", lw=1.2, alpha=0.3))
    ty = y + 0.2 if title_bottom else y + h - 0.3
    ax.text(x + 0.2, ty, title, fontsize=12, weight="bold", color="#222")


def box(x, y, text, w=2.2, h=0.9, color="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                fc=color, ec="#333", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.8)
    return x, y, w, h


def L(b):
    return b[0], b[1] + b[3] / 2


def R(b):
    return b[0] + b[2], b[1] + b[3] / 2


def T(b, dx=0.5):
    return b[0] + b[2] * dx, b[1] + b[3]


def B(b, dx=0.5):
    return b[0] + b[2] * dx, b[1]


def arrow(p1, p2, label="", offset=(0, 0.13), rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, color="#333",
                                 lw=1.3, shrinkA=2, shrinkB=2,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + offset[0], (p1[1] + p2[1]) / 2 + offset[1], label,
                fontsize=8.5, ha="center", color="#333", style="italic",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))


# ---------------- 1. data & training ----------------
group(0.2, 4.5, 7.5, 4.5, "1. Data, EDA & training (local or CI)", "#a8dadc")
uci = box(0.45, 7.35, "UCI repository\nprocessed.cleveland.data", w=2.3)
clean = box(3.15, 7.35, "data.py\ndownload + clean", w=1.9)
eda = box(5.4, 7.35, "eda.py + notebooks\nEDA figures", w=2.0)
feat = box(0.45, 5.75, "features.py\nsklearn Pipeline\nhr_pct_max, impute,\nscale, one-hot", w=2.3, h=1.1)
train = box(3.15, 5.75, "train.py\nGridSearchCV\n3 models\n5-fold stratified CV", w=1.9, h=1.1)
mlf = box(5.4, 5.75, "MLflow tracking\nparams, metrics,\nplots, models", w=2.0, h=1.1, color="#fff3b0")
model = box(3.15, 4.65, "models/model.joblib\n+ metadata.json", w=1.9, h=0.75, color="#ffe5ec")
arrow(R(uci), L(clean))
arrow(R(clean), L(eda))
arrow(B(clean), T(train), "heart_clean.csv", offset=(0.75, 0))
arrow(R(feat), L(train))
arrow(R(train), L(mlf))
arrow(B(train), T(model))

# ---------------- 2. CI/CD ----------------
group(8.0, 4.5, 7.8, 4.5, "2. CI/CD - GitHub Actions (.github/workflows/ci.yml)", "#ffd6a5")
gh = box(8.25, 7.35, "git push / PR\nGitHub repo", w=1.6)
lint = box(10.15, 7.35, "lint\nflake8", w=1.5)
test = box(11.95, 7.35, "test\npytest (30 tests)", w=1.6)
trn = box(13.85, 7.35, "train\nmodels + quality\ngate (AUC>=0.80)", w=1.75, h=1.0)
dock = box(13.6, 5.3, "docker build +\nsmoke test /predict", w=2.0, h=1.0)
art = box(10.9, 5.3, "uploaded artifacts\nmodel, mlruns, logs,\ntest + lint reports", w=2.2, h=1.0, color="#fff3b0")
fail = box(8.25, 5.3, "any failure ->\npipeline stops\n(red build + logs)", w=2.0, h=1.0, color="#ffccd5")
arrow(R(gh), L(lint))
arrow(R(lint), L(test))
arrow(R(test), L(trn))
arrow(B(trn), T(dock, 0.6))
arrow(L(dock), R(art))

# ---------------- 3. serving & monitoring ----------------
group(0.2, 0.2, 15.6, 4.0,
      "3. Serving & monitoring - Kubernetes (minikube, namespace heart-mlops)", "#cdb4db",
      title_bottom=True)
img = box(0.45, 1.5, "Docker image\nheart-api:1.0.0\npython:3.12-slim\nFastAPI + uvicorn", w=2.3, h=1.3)
pods = box(3.8, 1.3, "Deployment heart-api\n2 replicas (HPA 2-4)\n/predict  /health  /metrics\nJSON request logs", w=3.2, h=1.6)
svc = box(8.3, 1.65, "Service\ntype LoadBalancer :80", w=2.2)
ing = box(11.0, 1.65, "NGINX Ingress\nheart-api.local", w=2.0)
client = box(13.5, 1.65, "Client\ncurl / Swagger UI /\ntraffic script", w=2.1)
prom = box(8.3, 3.0, "Prometheus\n(k8s pod discovery)", w=2.2, h=0.85, color="#fff3b0")
graf = box(11.0, 3.0, "Grafana\nprovisioned dashboard", w=2.0, h=0.85, color="#fff3b0")
arrow(B(model, 0.2), T(img, 0.7), "COPY into image", offset=(-0.2, 0.25))
arrow(R(img), L(pods), "image load")
arrow(L(client), R(ing))
arrow(L(ing), R(svc))
arrow(L(svc), R(pods))
arrow(T(pods, 0.8), L(prom), "scrape /metrics", offset=(0, 0.2), rad=-0.25)
arrow(R(prom), L(graf), "PromQL")

ax.set_title("Heart Disease Risk Prediction - MLOps architecture", fontsize=16, weight="bold")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved", OUT)
