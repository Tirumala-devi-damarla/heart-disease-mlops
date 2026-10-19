import numpy as np
import pandas as pd
import pytest

from heart import config


@pytest.fixture
def raw_df():
    """Small raw-format frame including '?'-style NaNs and a duplicate row."""
    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],
        [67, 1, 4, 120, 229, 0, 2, 129, 1, 2.6, 2, 2, 7, 1],
        [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, np.nan, 3, 0],
        [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, 0, np.nan, 0],
        [56, 1, 2, 120, 236, 0, 0, 178, 0, 0.8, 1, 0, 3, 4],
        [56, 1, 2, 120, 236, 0, 0, 178, 0, 0.8, 1, 0, 3, 4],  # duplicate
    ]
    return pd.DataFrame(rows, columns=config.COLUMNS)


@pytest.fixture
def synthetic_clean():
    """Random but valid 'clean' dataset for fast model tests."""
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "age": rng.integers(30, 75, n).astype(float),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(100, 180, n).astype(float),
        "chol": rng.integers(150, 350, n).astype(float),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(90, 200, n).astype(float),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 4, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    })
    # make the target learnable: depends on cp, exang and oldpeak
    score = (df["cp"] == 4) * 1.5 + df["exang"] + df["oldpeak"] / 2
    df[config.TARGET] = (score + rng.normal(0, 0.5, n) > 1.8).astype(int)
    return df
