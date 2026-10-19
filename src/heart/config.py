"""Central place for paths and column definitions used across the project."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "processed.cleveland.data"
CLEAN_DATA_PATH = DATA_DIR / "processed" / "heart_clean.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# Column names as documented in heart-disease.names on the UCI repository
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

# continuous / count features -> imputed + scaled
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# already 0/1 -> only imputed
BINARY_FEATURES = ["sex", "fbs", "exang"]
# multi-class codes -> one-hot encoded
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2
