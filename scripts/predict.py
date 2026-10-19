"""Command-line inference with the saved model (no API needed).

Usage:
  python scripts/predict.py --json sample_request.json
  python scripts/predict.py --csv data/processed/heart_clean.csv --limit 5
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart import config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(config.MODEL_PATH))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", help="json file with one patient record")
    group.add_argument("--csv", help="csv file with patient rows")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model = joblib.load(args.model)
    if args.json:
        df = pd.DataFrame([json.loads(Path(args.json).read_text())])
    else:
        df = pd.read_csv(args.csv).head(args.limit)

    proba = model.predict_proba(df[config.FEATURES])[:, 1]
    out = df[config.FEATURES].copy()
    out["probability_disease"] = proba.round(4)
    out["prediction"] = (proba >= 0.5).astype(int)
    if config.TARGET in df.columns:
        out["actual"] = df[config.TARGET]
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
