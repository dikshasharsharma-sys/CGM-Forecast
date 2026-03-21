from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import AZT1D_DIR, HUPA_DIR, REPORTS_DIR


AZT1D_RENAME_MAP = {
    "EventDateTime": "time",
    "Basal": "basal_rate",
    "TotalBolusInsulinDelivered": "bolus_volume_delivered",
    "CarbSize": "carb_input",
    "CGM": "glucose",
}


@dataclass
class DatasetArtifacts:
    data: pd.DataFrame
    raw_columns: list[str]
    standardized_columns: list[str]


def _list_files(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.glob(pattern))


def _standardize_azt1d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns=AZT1D_RENAME_MAP, inplace=True)
    if "glucose" not in df.columns and "Readings (CGM / BGM)" in df.columns:
        df["glucose"] = df["Readings (CGM / BGM)"]
    elif "Readings (CGM / BGM)" in df.columns:
        df["glucose"] = df["glucose"].fillna(df["Readings (CGM / BGM)"])
    return df


def load_azt1d() -> DatasetArtifacts:
    subject_dirs = _list_files(AZT1D_DIR, "Subject *")
    all_frames: list[pd.DataFrame] = []
    raw_columns: set[str] = set()

    for subject_dir in subject_dirs:
        csv_files = _list_files(subject_dir, "Subject *.csv")
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            raw_columns.update(df.columns)
            df = _standardize_azt1d(df)
            df["subject_id"] = subject_dir.name
            all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined["time"] = pd.to_datetime(combined["time"], errors="coerce")
    standardized_columns = sorted(combined.columns)
    return DatasetArtifacts(
        data=combined,
        raw_columns=sorted(raw_columns),
        standardized_columns=standardized_columns,
    )


def load_hupa() -> DatasetArtifacts:
    csv_files = _list_files(HUPA_DIR, "*.csv")
    all_frames: list[pd.DataFrame] = []
    raw_columns: set[str] = set()

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, sep=";")
        raw_columns.update(df.columns)
        df["subject_id"] = csv_file.stem
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined["time"] = pd.to_datetime(combined["time"], errors="coerce")
    standardized_columns = sorted(combined.columns)
    return DatasetArtifacts(
        data=combined,
        raw_columns=sorted(raw_columns),
        standardized_columns=standardized_columns,
    )


def build_harmonization_report(
    azt1d_raw: Iterable[str],
    hupa_raw: Iterable[str],
    common_columns: Iterable[str],
    output_path: Path | None = None,
    interval_minutes: dict[str, float] | None = None,
    horizon_steps: dict[str, dict[str, int]] | None = None,
    lag_count_common: int | None = None,
) -> Path:
    output_path = output_path or (REPORTS_DIR / "harmonization_report.md")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    azt1d_raw_set = sorted(set(azt1d_raw))
    hupa_raw_set = sorted(set(hupa_raw))
    common_columns = sorted(set(common_columns))
    dropped_azt1d = sorted(set(azt1d_raw_set) - set(AZT1D_RENAME_MAP.keys()) - {"Readings (CGM / BGM)"})
    dropped_hupa = sorted(set(hupa_raw_set) - set(common_columns))

    report_lines = [
        "# Feature Harmonization Report",
        "",
        "## AZT1D Raw Columns",
        *[f"- {col}" for col in azt1d_raw_set],
        "",
        "## HUPA Raw Columns",
        *[f"- {col}" for col in hupa_raw_set],
        "",
        "## Common Columns (After Standardization)",
        *[f"- {col}" for col in common_columns],
        "",
        "## Dropped / Non-Common Columns",
        "### AZT1D",
        *[f"- {col}" for col in dropped_azt1d],
        "",
        "### HUPA",
        *[f"- {col}" for col in dropped_hupa],
        "",
        "## Notes",
        "- AZT1D glucose target is derived from the `CGM` column (fallback to `Readings (CGM / BGM)` when present).",
        "- HUPA glucose target is `glucose` from the preprocessed files (semicolon-separated).",
    ]

    if interval_minutes:
        report_lines.extend(
            [
                "",
                "## Estimated CGM Sampling Interval (minutes)",
                *[f"- {dataset}: {minutes:.2f}" for dataset, minutes in interval_minutes.items()],
            ]
        )

    if horizon_steps:
        report_lines.extend(
            [
                "",
                "## Forecasting Horizons (steps ahead)",
            ]
        )
        for dataset, horizons in horizon_steps.items():
            report_lines.append(f"- {dataset}:")
            for label, steps in horizons.items():
                report_lines.append(f"  - {label}: {steps} steps")

    if lag_count_common:
        report_lines.extend(
            [
                "",
                "## Lag Feature Configuration",
                f"- Common lag count used across datasets: {lag_count_common}",
            ]
        )

    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    return output_path
