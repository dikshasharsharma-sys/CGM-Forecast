from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR
from src.data_loader import build_harmonization_report, load_azt1d, load_hupa
from src.evaluation import evaluate_predictions, plot_metric, save_metrics, save_model_comparison
from src.modeling import get_models
from src.preprocessing import (
    add_forecast_target,
    add_lag_features,
    add_time_features,
    build_preprocessor,
    compute_time_interval_minutes,
)


TARGET_COLUMN = "glucose"
TIME_COLUMN = "time"
MIN_LAGS = 6
HORIZONS_MINUTES = {"30m": 30, "60m": 60}
MAX_TRAIN_SAMPLES = 100000
MAX_TEST_SAMPLES = 100000
DATASET_CLASSIFIER_PATH = MODELS_DIR / "dataset_classifier.pkl"
DATASET_CLASSIFIER_METRICS_PATH = REPORTS_DIR / "dataset_classifier_metrics.json"


def _prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return add_time_features(df)


def _get_context_features(azt1d_df: pd.DataFrame, hupa_df: pd.DataFrame) -> list[str]:
    candidates = [
        "basal_rate",
        "bolus_volume_delivered",
        "carb_input",
        "hour",
        "day_of_week",
    ]
    return [feature for feature in candidates if feature in azt1d_df.columns and feature in hupa_df.columns]


def _sample_df(df: pd.DataFrame, max_samples: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) > max_samples:
        return df.sample(max_samples, random_state=random_state)
    return df


def _build_dataset_classifier(
    azt1d_df: pd.DataFrame,
    hupa_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    azt1d_features = azt1d_df[feature_columns].copy()
    hupa_features = hupa_df[feature_columns].copy()
    azt1d_features["dataset_type"] = "AZT1D"
    hupa_features["dataset_type"] = "HUPA"
    combined = pd.concat([azt1d_features, hupa_features], ignore_index=True)
    combined = combined.dropna(subset=feature_columns + ["dataset_type"])
    combined = _sample_df(combined, MAX_TRAIN_SAMPLES * 2)

    X = combined[feature_columns]
    y = combined["dataset_type"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(feature_columns)
    classifier = LogisticRegression(max_iter=1000)
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    try:
        y_score = pipeline.predict_proba(X_test)
        positive_index = list(pipeline.classes_).index("AZT1D")
        y_proba = y_score[:, positive_index]
        roc_auc = roc_auc_score((y_test == "AZT1D").astype(int), y_proba)
    except Exception:
        roc_auc = float("nan")

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "roc_auc": float(roc_auc) if pd.notna(roc_auc) else None,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "classes": [str(label) for label in pipeline.classes_],
    }

    DATASET_CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, DATASET_CLASSIFIER_PATH)
    DATASET_CLASSIFIER_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_CLASSIFIER_METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


def run_experiments() -> pd.DataFrame:
    azt1d_artifacts = load_azt1d()
    hupa_artifacts = load_hupa()

    azt1d_df = _prepare_dataset(azt1d_artifacts.data)
    hupa_df = _prepare_dataset(hupa_artifacts.data)

    interval_azt1d = compute_time_interval_minutes(azt1d_df)
    interval_hupa = compute_time_interval_minutes(hupa_df)

    horizon_steps = {
        "AZT1D": {label: max(1, round(minutes / interval_azt1d)) for label, minutes in HORIZONS_MINUTES.items()},
        "HUPA": {label: max(1, round(minutes / interval_hupa)) for label, minutes in HORIZONS_MINUTES.items()},
    }

    lag_count_azt1d = max(horizon_steps["AZT1D"]["60m"], MIN_LAGS)
    lag_count_hupa = max(horizon_steps["HUPA"]["60m"], MIN_LAGS)
    lag_count_common = min(lag_count_azt1d, lag_count_hupa)

    azt1d_df = add_lag_features(azt1d_df, TARGET_COLUMN, lag_count_common)
    hupa_df = add_lag_features(hupa_df, TARGET_COLUMN, lag_count_common)

    for label in HORIZONS_MINUTES:
        azt1d_df = add_forecast_target(
            azt1d_df, TARGET_COLUMN, horizon_steps["AZT1D"][label], label
        )
        hupa_df = add_forecast_target(
            hupa_df, TARGET_COLUMN, horizon_steps["HUPA"][label], label
        )

    common_columns = sorted(
        set(azt1d_artifacts.data.columns).intersection(hupa_artifacts.data.columns)
    )

    build_harmonization_report(
        azt1d_raw=azt1d_artifacts.raw_columns,
        hupa_raw=hupa_artifacts.raw_columns,
        common_columns=common_columns,
        interval_minutes={
            "AZT1D": interval_azt1d,
            "HUPA": interval_hupa,
        },
        horizon_steps=horizon_steps,
        lag_count_common=lag_count_common,
    )

    common_columns_path = REPORTS_DIR / "common_columns.json"
    common_columns_path.parent.mkdir(parents=True, exist_ok=True)
    common_columns_path.write_text(json.dumps(common_columns, indent=2), encoding="utf-8")

    context_features = _get_context_features(azt1d_df, hupa_df)
    lag_features = [f"{TARGET_COLUMN}_lag_{i}" for i in range(1, lag_count_common + 1)]
    feature_columns = lag_features + context_features

    feature_config_path = REPORTS_DIR / "feature_config.json"
    feature_config_path.write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "context_features": context_features,
                "lag_count": lag_count_common,
                "interval_minutes": {
                    "AZT1D": interval_azt1d,
                    "HUPA": interval_hupa,
                },
                "horizon_steps": horizon_steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    numeric_features = feature_columns
    preprocessor = build_preprocessor(numeric_features)

    experiments = [
        {"name": "AZT1D->AZT1D", "type": "within", "train": azt1d_df},
        {"name": "HUPA->HUPA", "type": "within", "train": hupa_df},
        {"name": "AZT1D->HUPA", "type": "cross", "train": azt1d_df, "test": hupa_df},
        {"name": "HUPA->AZT1D", "type": "cross", "train": hupa_df, "test": azt1d_df},
    ]

    metrics: list[dict] = []
    models = get_models()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for horizon_label in HORIZONS_MINUTES:
        target_column = f"{TARGET_COLUMN}_t+{horizon_label}"
        for experiment in experiments:
            if experiment["type"] == "within":
                base_df = experiment["train"].dropna(subset=feature_columns + [target_column])
                train_df, test_df = train_test_split(
                    base_df, test_size=0.2, random_state=42
                )
            else:
                train_df = experiment["train"].dropna(subset=feature_columns + [target_column])
                test_df = experiment["test"].dropna(subset=feature_columns + [target_column])

            train_df = _sample_df(train_df, MAX_TRAIN_SAMPLES)
            test_df = _sample_df(test_df, MAX_TEST_SAMPLES)

            X_train = train_df[feature_columns]
            y_train = train_df[target_column]
            X_test = test_df[feature_columns]
            y_test = test_df[target_column]

            for model_name, model in models.items():
                pipeline = Pipeline(
                    steps=[
                        ("preprocess", preprocessor),
                        ("model", model),
                    ]
                )
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                scores = evaluate_predictions(y_test, preds)

                metrics.append(
                    {
                        "experiment": experiment["name"],
                        "horizon": horizon_label,
                        "model": model_name,
                        **scores,
                        "train_samples": len(train_df),
                        "test_samples": len(test_df),
                    }
                )

                safe_experiment = experiment["name"].replace("->", "_to_")
                model_path = MODELS_DIR / f"{safe_experiment}_{horizon_label}_{model_name}.joblib"
                joblib.dump(pipeline, model_path)

    _build_dataset_classifier(azt1d_df, hupa_df, feature_columns)

    metrics_df = save_metrics(metrics, OUTPUTS_DIR / "metrics.csv")
    metrics_df["experiment_horizon"] = metrics_df["experiment"] + " (" + metrics_df["horizon"] + ")"
    save_model_comparison(metrics_df, OUTPUTS_DIR / "model_comparison.csv")
    plot_metric(metrics_df, "mae", OUTPUTS_DIR / "mae_by_experiment.png", x_col="experiment_horizon")
    plot_metric(metrics_df, "rmse", OUTPUTS_DIR / "rmse_by_experiment.png", x_col="experiment_horizon")
    plot_metric(metrics_df, "r2", OUTPUTS_DIR / "r2_by_experiment.png", x_col="experiment_horizon")
    return metrics_df


if __name__ == "__main__":
    run_experiments()
