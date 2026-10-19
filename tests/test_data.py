import pandas as pd
import pytest

from heart import config
from heart.data import clean_data, load_raw, split_xy, validate_data


def test_clean_removes_duplicates(raw_df):
    clean = clean_data(raw_df)
    assert len(clean) == len(raw_df) - 1


def test_clean_fills_missing_values(raw_df):
    assert raw_df[["ca", "thal"]].isna().sum().sum() == 2
    clean = clean_data(raw_df)
    assert clean.isna().sum().sum() == 0


def test_target_is_binary(raw_df):
    clean = clean_data(raw_df)
    assert "num" not in clean.columns
    assert set(clean[config.TARGET]) == {0, 1}
    # num values 1-4 must all map to 1
    assert clean.loc[clean["age"] == 56, config.TARGET].iloc[0] == 1


def test_categorical_columns_are_int(raw_df):
    clean = clean_data(raw_df)
    for col in config.CATEGORICAL_FEATURES + config.BINARY_FEATURES:
        assert pd.api.types.is_integer_dtype(clean[col]), col


def test_validate_passes_on_clean_data(raw_df):
    validate_data(clean_data(raw_df))


def test_validate_rejects_missing_column(raw_df):
    clean = clean_data(raw_df).drop(columns=["chol"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_data(clean)


def test_validate_rejects_bad_values(raw_df):
    clean = clean_data(raw_df)
    clean.loc[0, "age"] = -5
    with pytest.raises(ValueError):
        validate_data(clean)


def test_load_raw_parses_question_marks(tmp_path):
    f = tmp_path / "raw.data"
    f.write_text("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,6.0,0\n")
    df = load_raw(f)
    assert list(df.columns) == config.COLUMNS
    assert pd.isna(df.loc[0, "ca"])


def test_split_xy(synthetic_clean):
    X, y = split_xy(synthetic_clean)
    assert list(X.columns) == config.FEATURES
    assert len(X) == len(y)
