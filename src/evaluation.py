from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "r2": r2_score(y_true, y_pred),
    }


def save_metrics(metrics: list[dict], output_path: Path) -> pd.DataFrame:
    metrics_df = pd.DataFrame(metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    return metrics_df


def save_model_comparison(metrics_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    group_cols = [col for col in ["experiment", "horizon", "model"] if col in metrics_df.columns]
    summary = (
        metrics_df.groupby(group_cols, as_index=False)
        .agg({"mae": "mean", "rmse": "mean", "r2": "mean"})
        .sort_values(group_cols)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def plot_metric(
    metrics_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    x_col: str = "experiment",
) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x=x_col, y=metric, hue="model")
    plt.title(f"{metric.upper()} by {x_col.replace('_', ' ').title()} and Model")
    plt.xticks(rotation=15)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
