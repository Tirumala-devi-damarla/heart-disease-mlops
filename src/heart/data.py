"""Download and clean the UCI Heart Disease (Cleveland) dataset."""

import argparse
import logging
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from heart import config

logger = logging.getLogger(__name__)


def download_data(url: str = config.DATA_URL,
                  dest: Path = config.RAW_DATA_PATH,
                  force: bool = False) -> Path:
    """Download the raw csv from UCI (skipped if it already exists)."""
    dest = Path(dest)
    if dest.exists() and not force:
        logger.info("Raw data already present at %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    # use certifi's CA bundle when available (python.org builds on macOS
    # do not ship system certificates)
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, context=ctx, timeout=30) as resp:
        dest.write_bytes(resp.read())
    return dest


def load_raw(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw file. Missing values are encoded as '?' in the source."""
    return pd.read_csv(path, header=None, names=config.COLUMNS,
                       na_values="?")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe.

    Steps:
      1. drop exact duplicate rows
      2. cast every column to numeric
      3. fill the few missing values in `ca` and `thal` with the mode
      4. convert the 0-4 `num` column into a binary target (0 = no disease)
      5. cast code-like columns to int
    """
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors="coerce")

    for col in ["ca", "thal"]:
        if df[col].isna().any():
            mode_value = df[col].mode()[0]
            logger.info("Filling %d missing values in %s with %s",
                        df[col].isna().sum(), col, mode_value)
            df[col] = df[col].fillna(mode_value)

    # remaining rows with missing values (should be none) are dropped
    df = df.dropna().reset_index(drop=True)

    df[config.TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    int_cols = config.BINARY_FEATURES + config.CATEGORICAL_FEATURES + ["ca"]
    df[int_cols] = df[int_cols].astype(int)
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Very small sanity checks so bad data fails loudly."""
    missing = set(config.FEATURES + [config.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[config.FEATURES].isna().any().any():
        raise ValueError("Cleaned data still contains missing values")
    if not set(df[config.TARGET].unique()) <= {0, 1}:
        raise ValueError("Target must be binary")
    if (df["age"] <= 0).any() or (df["chol"] <= 0).any():
        raise ValueError("Found non-positive age/cholesterol values")


def load_clean(path: Path = config.CLEAN_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    return df[config.FEATURES], df[config.TARGET]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file exists")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    raw_path = download_data(force=args.force)
    raw = load_raw(raw_path)
    logger.info("Raw shape: %s, missing values:\n%s", raw.shape,
                raw.isna().sum()[raw.isna().sum() > 0])
    clean = clean_data(raw)
    validate_data(clean)
    config.CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.CLEAN_DATA_PATH, index=False)
    logger.info("Saved cleaned data %s -> %s", clean.shape,
                config.CLEAN_DATA_PATH)
    logger.info("Class balance: %s",
                np.bincount(clean[config.TARGET]).tolist())


if __name__ == "__main__":
    main()
