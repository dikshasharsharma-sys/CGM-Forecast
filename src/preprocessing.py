from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    df = df.copy()
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df["hour"] = df[time_col].dt.hour
        df["day_of_week"] = df[time_col].dt.dayofweek
    return df


def compute_time_interval_minutes(
    df: pd.DataFrame,
    time_col: str = "time",
    group_col: str = "subject_id",
) -> float:
    df = df[[time_col, group_col]].dropna().copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    deltas = (
        df.sort_values([group_col, time_col])
        .groupby(group_col)[time_col]
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60
    )
    if deltas.empty:
        return 5.0
    return float(deltas.median())


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: int,
    time_col: str = "time",
    group_col: str = "subject_id",
) -> pd.DataFrame:
    df = df.sort_values([group_col, time_col]).copy()
    for lag in range(1, lags + 1):
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    return df


def add_forecast_target(
    df: pd.DataFrame,
    target_col: str,
    steps_ahead: int,
    horizon_label: str,
    time_col: str = "time",
    group_col: str = "subject_id",
) -> pd.DataFrame:
    df = df.sort_values([group_col, time_col]).copy()
    df[f"{target_col}_t+{horizon_label}"] = df.groupby(group_col)[target_col].shift(-steps_ahead)
    return df


def build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_features)],
        remainder="drop",
    )
