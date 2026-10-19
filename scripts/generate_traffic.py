"""Send random (but realistic) patient requests to the API.

Used to produce data for the Prometheus/Grafana dashboard and for the demo video.
Usage: python scripts/generate_traffic.py --url http://127.0.0.1:8080 -n 200
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request


def random_patient(rng: random.Random) -> dict:
    sick = rng.random() < 0.45
    return {
        "age": rng.randint(35, 75),
        "sex": rng.choice([0, 1, 1]),
        "cp": rng.choice([4, 4, 4, 3] if sick else [1, 2, 3, 3]),
        "trestbps": rng.randint(105, 170),
        "chol": rng.randint(170, 330),
        "fbs": int(rng.random() < 0.15),
        "restecg": rng.choice([0, 2]),
        "thalach": rng.randint(95, 150) if sick else rng.randint(140, 195),
        "exang": int(rng.random() < (0.6 if sick else 0.1)),
        "oldpeak": round(rng.uniform(0.8, 4.0) if sick else rng.uniform(0, 1.2), 1),
        "slope": rng.choice([2, 2, 3] if sick else [1, 1, 2]),
        "ca": rng.choice([1, 2, 3] if sick else [0, 0, 1]),
        "thal": rng.choice([7, 7, 6] if sick else [3, 3, 7]),
    }


def post(url: str, payload: dict) -> int:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--invalid-ratio", type=float, default=0.05,
                        help="share of deliberately invalid requests (422s)")
    args = parser.parse_args()
    rng = random.Random(7)
    codes = {}
    for _ in range(args.n):
        patient = random_patient(rng)
        if rng.random() < args.invalid_ratio:
            patient["age"] = -1
        code = post(f"{args.url}/predict", patient)
        codes[code] = codes.get(code, 0) + 1
        time.sleep(args.delay)
    print("status codes:", codes)


if __name__ == "__main__":
    main()
