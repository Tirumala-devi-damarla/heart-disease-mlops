"""Feature engineering + preprocessing pipeline.

Everything that transforms raw patient fields into model inputs lives inside a
single sklearn Pipeline, so the exact same steps run during training, in the
tests and inside the API container (no train/serve skew).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart import config


class HeartRateReserve(BaseEstimator, TransformerMixin):
    """Adds `hr_pct_max`: achieved max heart rate as a fraction of the
    age-predicted maximum (220 - age). Patients who cannot reach a high
    percentage during the exercise test are more likely to have disease,
    so this combines `age` and `thalach` into one clinically meaningful value.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        predicted_max = (220 - X["age"]).clip(lower=1)
        X["hr_pct_max"] = X["thalach"] / predicted_max
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(input_features) if input_features is not None else []
        return np.array(names + ["hr_pct_max"], dtype=object)


ENGINEERED_FEATURES = ["hr_pct_max"]


def build_preprocessor() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES + ENGINEERED_FEATURES),
        ("bin", binary, config.BINARY_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("engineer", HeartRateReserve()),
        ("columns", columns),
    ])


def build_pipeline(model) -> Pipeline:
    """Full pipeline: preprocessing + classifier."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])


def get_feature_names(pipeline: Pipeline) -> list:
    columns = pipeline.named_steps["preprocess"].named_steps["columns"]
    return [n.split("__", 1)[1] for n in columns.get_feature_names_out()]
