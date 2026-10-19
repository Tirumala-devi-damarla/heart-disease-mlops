import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from heart import config
from heart.features import (HeartRateReserve, build_pipeline,
                            build_preprocessor, get_feature_names)


def test_heart_rate_reserve_value():
    X = pd.DataFrame({"age": [40, 60], "thalach": [180, 80]})
    out = HeartRateReserve().fit_transform(X)
    assert np.isclose(out.loc[0, "hr_pct_max"], 1.0)
    assert np.isclose(out.loc[1, "hr_pct_max"], 0.5)


def test_preprocessor_output_has_no_nans(synthetic_clean):
    X = synthetic_clean[config.FEATURES]
    out = build_preprocessor().fit_transform(X)
    assert out.shape[0] == len(X)
    assert not np.isnan(out).any()


def test_preprocessor_one_hot_and_scaling(synthetic_clean):
    X = synthetic_clean[config.FEATURES]
    pre = build_preprocessor().fit(X)
    out = pre.transform(X)
    # 7 numeric (incl. engineered) + 3 binary + one-hot(cp4 + restecg3 + slope3 + thal3)
    assert out.shape[1] == 7 + 3 + 4 + 3 + 3 + 3
    # scaled numeric columns have ~zero mean
    assert np.allclose(out[:, :7].mean(axis=0), 0, atol=1e-6)


def test_preprocessor_handles_missing_and_unseen(synthetic_clean):
    X = synthetic_clean[config.FEATURES]
    pre = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy().astype(float)
    row.loc[:, ["ca", "thal"]] = np.nan
    row.loc[:, "restecg"] = 9  # unseen category -> ignored by OneHotEncoder
    out = pre.transform(row)
    assert not np.isnan(out).any()


def test_feature_names(synthetic_clean):
    X, y = synthetic_clean[config.FEATURES], synthetic_clean[config.TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    names = get_feature_names(pipe)
    assert "hr_pct_max" in names
    assert any(n.startswith("cp_") for n in names)
