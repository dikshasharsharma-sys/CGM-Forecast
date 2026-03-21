from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
COMMON_COLUMNS_PATH = REPORTS_DIR / "common_columns.json"
FEATURE_CONFIG_PATH = REPORTS_DIR / "feature_config.json"
HARMONIZATION_REPORT_PATH = REPORTS_DIR / "harmonization_report.md"
METRICS_PATH = OUTPUTS_DIR / "metrics.csv"
DATASET_CLASSIFIER_PATH = MODELS_DIR / "dataset_classifier.pkl"
DATASET_CONFIDENCE_THRESHOLD = 0.65


@dataclass(frozen=True)
class ExperimentModel:
    experiment: str
    horizon: str
    model_name: str
    path: Path


def load_feature_config() -> dict:
    if not FEATURE_CONFIG_PATH.exists():
        return {}
    return json.loads(FEATURE_CONFIG_PATH.read_text(encoding="utf-8"))


def _read_json_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [str(item) for item in payload]
    return []


def _write_json_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(set(values)), indent=2), encoding="utf-8")


def _infer_common_columns_from_report(report_path: Path) -> list[str]:
    if not report_path.exists():
        return []
    lines = report_path.read_text(encoding="utf-8").splitlines()
    common_columns: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## Common Columns"):
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break
        if in_section and stripped.startswith("-"):
            column = stripped.lstrip("- ").strip()
            if column:
                common_columns.append(column)
    return common_columns


def _infer_common_columns_from_datasets() -> list[str]:
    try:
        from src.data_loader import load_azt1d, load_hupa
    except Exception:
        return []

    try:
        azt1d_artifacts = load_azt1d()
        hupa_artifacts = load_hupa()
    except Exception:
        return []

    return sorted(
        set(azt1d_artifacts.standardized_columns).intersection(hupa_artifacts.standardized_columns)
    )


def _infer_common_columns_from_models(feature_config: dict) -> list[str]:
    model_features: list[str] = []
    if MODELS_DIR.exists():
        for model_path in MODELS_DIR.glob("*.joblib"):
            try:
                model = joblib.load(model_path)
            except Exception:
                continue
            if hasattr(model, "feature_names_in_"):
                model_features = list(getattr(model, "feature_names_in_"))
                break
            if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
                preprocessor = model.named_steps.get("preprocess")
                if hasattr(preprocessor, "feature_names_in_"):
                    model_features = list(getattr(preprocessor, "feature_names_in_"))
                    break

    context_features = feature_config.get("context_features", []) or []
    if model_features:
        return sorted(set(model_features) | set(context_features))
    if context_features:
        return sorted(context_features)
    return sorted(feature_config.get("feature_columns", []) or [])


def load_common_columns(feature_config: dict) -> tuple[list[str], str]:
    existing = _read_json_list(COMMON_COLUMNS_PATH)
    if existing:
        return existing, "file"

    inferred = _infer_common_columns_from_report(HARMONIZATION_REPORT_PATH)
    if inferred:
        _write_json_list(COMMON_COLUMNS_PATH, inferred)
        return inferred, "harmonization_report"

    inferred = _infer_common_columns_from_datasets()
    if inferred:
        _write_json_list(COMMON_COLUMNS_PATH, inferred)
        return inferred, "dataset_intersection"

    inferred = _infer_common_columns_from_models(feature_config)
    if inferred:
        _write_json_list(COMMON_COLUMNS_PATH, inferred)
        return inferred, "model_or_config"

    return [], "unavailable"


def load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(METRICS_PATH)


def load_dataset_classifier():
    if not DATASET_CLASSIFIER_PATH.exists():
        return None
    try:
        return joblib.load(DATASET_CLASSIFIER_PATH)
    except Exception:
        return None


def parse_model_metadata(model_path: Path) -> ExperimentModel | None:
    parts = model_path.stem.split("_")
    if len(parts) < 5 or parts[1] != "to":
        return None
    experiment = f"{parts[0]}->{parts[2]}"
    horizon = parts[3]
    model_name = " ".join(parts[4:])
    return ExperimentModel(experiment=experiment, horizon=horizon, model_name=model_name, path=model_path)


def load_model_registry() -> dict[str, list[ExperimentModel]]:
    registry: dict[str, list[ExperimentModel]] = {}
    if MODELS_DIR.exists():
        for model_path in sorted(MODELS_DIR.glob("*.joblib")):
            metadata = parse_model_metadata(model_path)
            if metadata is None:
                continue
            key = f"{metadata.experiment} | {metadata.horizon}"
            registry.setdefault(key, []).append(metadata)
    return registry


def format_experiment_label(experiment: str, horizon: str) -> str:
    train, test = experiment.split("->")
    return f"{train} -> {test} | {horizon}"


def split_experiment_key(experiment_key: str) -> tuple[str, str]:
    experiment, horizon = [part.strip() for part in experiment_key.split("|")]
    return experiment, horizon


def get_available_horizons(model_registry: dict[str, list[ExperimentModel]]) -> list[str]:
    horizons = {split_experiment_key(key)[1] for key in model_registry}
    return sorted(horizons)


def filter_registry_by_horizon(
    model_registry: dict[str, list[ExperimentModel]], horizon: str
) -> dict[str, list[ExperimentModel]]:
    return {
        key: models
        for key, models in model_registry.items()
        if split_experiment_key(key)[1] == horizon
    }


def get_models_for_experiment(
    model_registry: dict[str, list[ExperimentModel]],
    experiment: str,
    horizon: str,
) -> list[ExperimentModel]:
    for key, models in model_registry.items():
        exp, hor = split_experiment_key(key)
        if exp == experiment and hor == horizon:
            return models
    return []


def _section_header(badge: str, title: str) -> None:
    """Render a styled section header with a purple pill badge."""
    st.markdown(
        f'<div class="form-section-header">'
        f'<span class="form-section-badge">{badge}</span>'
        f'<span class="form-section-title">{title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def build_input_form(feature_columns: list[str]) -> dict[str, float]:
    inputs: dict[str, float] = {}
    lag_features = sorted(
        [f for f in feature_columns if f.startswith("glucose_lag_")],
        key=lambda v: int(v.split("_")[-1]) if v.split("_")[-1].isdigit() else 0,
    )
    context_features = [
        f for f in feature_columns if f in {"basal_rate", "bolus_volume_delivered", "carb_input"}
    ]
    time_features = [f for f in feature_columns if f in {"hour", "day_of_week"}]
    remaining_features = [
        f for f in feature_columns
        if f not in set(lag_features + context_features + time_features)
    ]

    # ── Two-column layout: lags left, context/time right ────────────────
    left_col, right_col = st.columns([2.4, 1], gap="large")

    with left_col:
        st.markdown(
            '<div class="form-card form-card-cgm">'
            '<div class="form-card-header form-card-header-cgm">'
            '<span class="form-card-icon">📉</span>'
            '<span class="form-card-title">CGM History &mdash; Glucose Readings</span>'
            '<div class="form-card-subtitle">Enter past CGM values in mg/dL &nbsp;·&nbsp; Lag 1 = most recent &nbsp;·&nbsp; Lag N = oldest</div>'
            '<div class="glucose-range-pills">'
            '<span class="range-pill range-pill-low">🟡 Hypo &lt;70</span>'
            '<span class="range-pill range-pill-norm">🟢 Normal 70–180</span>'
            '<span class="range-pill range-pill-high">🔴 Hyper &gt;180</span>'
            '</div></div>'
            '<div class="form-card-body">',
            unsafe_allow_html=True,
        )
        if lag_features:
            # Split into 3 groups of 6 for recency banding
            _LAG_GROUPS = [
                (lag_features[0:6],   "lag-band-recent", "lag-group-recent", "🟣",  "Most Recent — Lags 1–6 (0–30 min ago)"),
                (lag_features[6:12],  "lag-band-mid",    "lag-group-mid",    "🟡", "Intermediate — Lags 7–12 (30–60 min ago)"),
                (lag_features[12:18], "lag-band-old",    "lag-group-old",    "🟢",  "Oldest — Lags 13–18 (60–90 min ago)"),
            ]
            for group_feats, band_cls, label_cls, icon, label_text in _LAG_GROUPS:
                if not group_feats:
                    continue
                st.markdown(
                    f'<div class="lag-group">'
                    f'<div class="lag-group-label {label_cls}">{icon}&nbsp;{label_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="{band_cls}">', unsafe_allow_html=True)
                grid = st.columns(6)
                for idx, feature in enumerate(group_feats):
                    lag_num = feature.split("_")[-1]
                    with grid[idx]:
                        inputs[feature] = st.number_input(
                            f"Lag {lag_num}",
                            min_value=0.0,
                            value=110.0,
                            key=feature,
                            label_visibility="visible",
                        )
                st.markdown('</div>', unsafe_allow_html=True)
            # any overflow if lags > 18
            overflow = lag_features[18:]
            if overflow:
                grid = st.columns(6)
                for idx, feature in enumerate(overflow):
                    lag_num = feature.split("_")[-1]
                    with grid[idx % 6]:
                        inputs[feature] = st.number_input(
                            f"Lag {lag_num}", min_value=0.0, value=110.0, key=feature
                        )
        st.markdown('</div></div>', unsafe_allow_html=True)

    with right_col:
        _CONTEXT_LABELS = {
            "basal_rate": ("Basal Rate", "U/hr"),
            "bolus_volume_delivered": ("Bolus Volume", "U"),
            "carb_input": ("Carb Input", "g"),
        }
        if context_features:
            st.markdown(
                '<div class="form-card form-card-context">'
                '<div class="form-card-header form-card-header-context">'
                '<span class="form-card-icon">💉</span>'
                '<span class="form-card-title">Insulin &amp; Carbs</span>'
                '<div class="form-card-subtitle">Treatment context inputs</div>'
                '</div><div class="form-card-body">',
                unsafe_allow_html=True,
            )
            for feature in context_features:
                label, unit = _CONTEXT_LABELS.get(feature, (feature.replace("_", " ").title(), ""))
                st.markdown(f'<div class="context-row-label">{label} ({unit})</div>', unsafe_allow_html=True)
                inputs[feature] = st.number_input(
                    label, min_value=0.0, value=0.0, key=feature, label_visibility="collapsed"
                )
            st.markdown('</div></div>', unsafe_allow_html=True)

        if time_features:
            st.markdown(
                '<div class="form-card form-card-clock">'
                '<div class="form-card-header form-card-header-clock">'
                '<span class="form-card-icon">⏰</span>'
                '<span class="form-card-title">Time Context</span>'
                '<div class="form-card-subtitle">Current time of reading</div>'
                '</div><div class="form-card-body">',
                unsafe_allow_html=True,
            )
            for feature in time_features:
                if feature == "hour":
                    st.markdown('<div class="context-row-label">Hour of Day (0–23)</div>', unsafe_allow_html=True)
                    inputs[feature] = st.number_input(
                        "Hour", min_value=0, max_value=23, value=12, key=feature, label_visibility="collapsed"
                    )
                else:
                    st.markdown('<div class="context-row-label">Day of Week (0=Mon — 6=Sun)</div>', unsafe_allow_html=True)
                    inputs[feature] = st.number_input(
                        "Day", min_value=0, max_value=6, value=2, key=feature, label_visibility="collapsed"
                    )
            st.markdown('</div></div>', unsafe_allow_html=True)

        if remaining_features:
            st.markdown(
                '<div class="form-card">'
                '<div class="form-card-header" style="background:#f8fafc;">'
                '<span class="form-card-icon">⚙️</span>'
                '<span class="form-card-title">Other Features</span>'
                '</div><div class="form-card-body">',
                unsafe_allow_html=True,
            )
            for feature in remaining_features:
                label = feature.replace("_", " ").title()
                inputs[feature] = st.number_input(label, min_value=0.0, value=0.0, key=feature)
            st.markdown('</div></div>', unsafe_allow_html=True)

        if not (context_features or time_features or remaining_features):
            st.markdown(
                "<div style='color:#94a3b8;font-size:0.85rem;padding:2rem 0;text-align:center;'>"
                "No additional context features configured."
                "</div>",
                unsafe_allow_html=True,
            )

    return inputs


def inject_css() -> None:
    st.markdown(
        """
        <style>
            /* ── Global ─────────────────────────────────────── */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
            [data-testid="stAppViewContainer"] > .main {
                padding-top: 1rem;
                background:
                    radial-gradient(ellipse at 0% 0%,    rgba(99,102,241,0.18) 0%, transparent 45%),
                    radial-gradient(ellipse at 100% 0%,  rgba(168,85,247,0.15) 0%, transparent 45%),
                    radial-gradient(ellipse at 50% 100%, rgba(20,184,166,0.13) 0%, transparent 55%),
                    radial-gradient(ellipse at 50% 50%,  rgba(236,72,153,0.07) 0%, transparent 60%),
                    linear-gradient(160deg, #eef2ff 0%, #f5f3ff 40%, #fdf4ff 70%, #ecfdf5 100%);
            }
            [data-testid="stHeader"] { background: transparent; }
            #MainMenu, footer { visibility: hidden; }



            /* ── Number inputs ───────────────────────────────── */
            .stNumberInput input {
                border-radius: 10px !important; border: 1.5px solid #e2e8f0 !important;
                background: #ffffff !important; font-size: 0.9rem !important;
                font-weight: 600 !important; color: #0f172a !important;
                padding: 0.3rem 0.5rem !important;
            }
            .stNumberInput input:focus {
                border-color: #6366f1 !important;
                box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
            }
            .stNumberInput label {
                font-size: 0.72rem !important; font-weight: 700 !important;
                color: #64748b !important; text-transform: uppercase !important;
                letter-spacing: 0.05em !important;
            }
            .stNumberInput button {
                border-radius: 6px !important; background: #f1f5f9 !important;
                border: 1px solid #e2e8f0 !important;
            }

            /* ── Selectbox ───────────────────────────────────── */
            .stSelectbox > div > div {
                border-radius: 12px !important; border: 1.5px solid #e2e8f0 !important;
                background: #ffffff !important; font-weight: 600 !important;
                font-size: 0.95rem !important;
            }

            /* ── Primary button ──────────────────────────────── */
            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
                border: none !important; border-radius: 14px !important;
                font-size: 1rem !important; font-weight: 700 !important;
                padding: 0.75rem 1.5rem !important;
                box-shadow: 0 4px 16px rgba(99,102,241,0.4) !important;
                letter-spacing: 0.01em !important;
            }
            .stButton > button[kind="primary"]:hover {
                opacity: 0.9 !important;
                box-shadow: 0 6px 22px rgba(99,102,241,0.5) !important;
            }

            /* ── Form section header ─────────────────────────── */
            .form-section-header {
                display: flex; align-items: center; gap: 0.5rem;
                margin: 1.2rem 0 0.6rem 0; padding-bottom: 0.45rem;
                border-bottom: 1.5px solid #e2e8f0;
            }
            .form-section-badge {
                background: #ede9fe; color: #6366f1; font-size: 0.7rem; font-weight: 700;
                text-transform: uppercase; letter-spacing: 0.07em;
                padding: 0.15rem 0.5rem; border-radius: 20px;
            }
            .form-section-title { font-size: 0.88rem; font-weight: 700; color: #1e293b; }

            /* ── Form card base ────────────────────────────────── */
            .form-card {
                border-radius: 20px; overflow: hidden;
                margin-bottom: 1.1rem; transition: box-shadow 0.25s, transform 0.15s;
            }
            .form-card:hover { transform: translateY(-2px); }

            /* CGM card — royal blue */
            .form-card-cgm {
                background: #dbeafe;
                border: 2.5px solid #2563eb;
                box-shadow: 0 6px 28px rgba(37,99,235,0.20), inset 0 1px 0 rgba(255,255,255,0.6);
            }
            .form-card-cgm .form-card-header {
                background: linear-gradient(90deg, #1e3a8a 0%, #1d4ed8 100%);
                padding: 1rem 1.4rem 0.9rem 1.4rem; border-bottom: none;
            }
            .form-card-cgm .form-card-title   { color: #ffffff !important; font-size: 1.02rem; font-weight: 800; }
            .form-card-cgm .form-card-subtitle { color: rgba(255,255,255,0.85) !important; font-size: 0.76rem; font-weight: 500; margin-top: 0.2rem; }
            .form-card-cgm .range-pill-low  { background: rgba(255,255,255,0.20); color: #fff; border: 1px solid rgba(255,255,255,0.35); }
            .form-card-cgm .range-pill-norm { background: rgba(255,255,255,0.20); color: #fff; border: 1px solid rgba(255,255,255,0.35); }
            .form-card-cgm .range-pill-high { background: rgba(255,255,255,0.20); color: #fff; border: 1px solid rgba(255,255,255,0.35); }

            /* Insulin & Carbs card — pure crimson */
            .form-card-context {
                background: #fee2e2;
                border: 2.5px solid #dc2626;
                box-shadow: 0 6px 28px rgba(220,38,38,0.18), inset 0 1px 0 rgba(255,255,255,0.6);
            }
            .form-card-context .form-card-header {
                background: linear-gradient(90deg, #7f1d1d 0%, #b91c1c 100%);
                padding: 1rem 1.4rem 0.9rem 1.4rem; border-bottom: none;
            }
            .form-card-context .form-card-title   { color: #ffffff !important; font-size: 1.02rem; font-weight: 800; }
            .form-card-context .form-card-subtitle { color: rgba(255,255,255,0.85) !important; font-size: 0.76rem; font-weight: 500; margin-top: 0.2rem; }
            .form-card-context .context-row-label  { color: #7f1d1d; }

            /* Time Context card — pure emerald */
            .form-card-clock {
                background: #d1fae5;
                border: 2.5px solid #059669;
                box-shadow: 0 6px 28px rgba(5,150,105,0.18), inset 0 1px 0 rgba(255,255,255,0.6);
            }
            .form-card-clock .form-card-header {
                background: linear-gradient(90deg, #064e3b 0%, #047857 100%);
                padding: 1rem 1.4rem 0.9rem 1.4rem; border-bottom: none;
            }
            .form-card-clock .form-card-title   { color: #ffffff !important; font-size: 1.02rem; font-weight: 800; }
            .form-card-clock .form-card-subtitle { color: rgba(255,255,255,0.85) !important; font-size: 0.76rem; font-weight: 500; margin-top: 0.2rem; }
            .form-card-clock .context-row-label  { color: #064e3b; }

            /* Shared card parts */
            .form-card-body    { padding: 1.1rem 1.4rem 1.1rem 1.4rem; }
            .form-card-icon    { font-size: 1.3rem; margin-right: 0.4rem; vertical-align: middle; }
            .form-card-title   { vertical-align: middle; }
            .form-card-subtitle { display: block; }
            .glucose-range-pills { display: flex; gap: 0.4rem; margin-top: 0.6rem; flex-wrap: wrap; }
            .range-pill      { font-size: 0.68rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 20px; letter-spacing: 0.04em; }
            .range-pill-low  { background: #fef9c3; color: #854d0e; border: 1px solid #fde68a; }
            .range-pill-norm { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
            .range-pill-high { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
            .context-row-label { font-size: 0.71rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.1rem; }

            /* ── Metric tiles ────────────────────────────────── */
            [data-testid="metric-container"] {
                background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
                padding: 0.9rem 1rem !important;
                box-shadow: 0 2px 8px rgba(15,23,42,0.04);
            }
            [data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: 800 !important; color: #0f172a !important; }
            [data-testid="stMetricLabel"] { font-size: 0.74rem !important; font-weight: 700 !important; color: #64748b !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }

            /* ── Dataframe ───────────────────────────────────── */
            [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

            /* ── Hero header ─────────────────────────────────── */
            @keyframes hero-shimmer {
                0%   { transform: translateX(-100%) skewX(-12deg); }
                100% { transform: translateX(250%) skewX(-12deg); }
            }
            @keyframes hero-float {
                0%, 100% { transform: translateY(0px); }
                50%       { transform: translateY(-8px); }
            }
            @keyframes hero-pulse {
                0%, 100% { opacity: 0.6; transform: scale(1); }
                50%       { opacity: 1;   transform: scale(1.05); }
            }

            .hero-banner {
                background:
                    radial-gradient(ellipse at 20% 50%,  rgba(139,92,246,0.55)  0%, transparent 55%),
                    radial-gradient(ellipse at 80% 20%,  rgba(59,130,246,0.45)  0%, transparent 50%),
                    radial-gradient(ellipse at 60% 90%,  rgba(236,72,153,0.30)  0%, transparent 45%),
                    linear-gradient(130deg, #0f0c29 0%, #1a1048 35%, #24243e 65%, #0f0c29 100%);
                border-radius: 24px;
                padding: 2.4rem 2.8rem 2rem 2.8rem;
                margin-bottom: 1.6rem;
                box-shadow:
                    0 0 0 1px rgba(139,92,246,0.3),
                    0 20px 60px rgba(15,12,41,0.55),
                    0 4px 20px rgba(139,92,246,0.25),
                    inset 0 1px 0 rgba(255,255,255,0.07);
                position: relative; overflow: hidden;
            }

            /* animated shimmer sweep */
            .hero-banner::before {
                content: '';
                position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                background: linear-gradient(105deg,
                    transparent 40%,
                    rgba(255,255,255,0.06) 50%,
                    transparent 60%);
                animation: hero-shimmer 4s ease-in-out infinite;
                pointer-events: none;
            }

            /* top-right decorative orb */
            .hero-banner::after {
                content: '';
                position: absolute; top: -60px; right: -60px;
                width: 260px; height: 260px;
                background: radial-gradient(circle, rgba(139,92,246,0.25) 0%, transparent 70%);
                border-radius: 50%;
                animation: hero-pulse 5s ease-in-out infinite;
                pointer-events: none;
            }

            /* bottom-left accent orb */
            .hero-orb-bl {
                position: absolute; bottom: -50px; left: 60px;
                width: 180px; height: 180px;
                background: radial-gradient(circle, rgba(59,130,246,0.20) 0%, transparent 70%);
                border-radius: 50%;
                animation: hero-pulse 6s ease-in-out 2s infinite;
                pointer-events: none;
            }

            .hero-inner { position: relative; z-index: 2; }

            .hero-eyebrow {
                display: inline-flex; align-items: center; gap: 0.4rem;
                background: rgba(139,92,246,0.25);
                border: 1px solid rgba(139,92,246,0.45);
                border-radius: 20px; padding: 0.22rem 0.75rem;
                font-size: 0.7rem; font-weight: 800; color: #c4b5fd;
                text-transform: uppercase; letter-spacing: 0.1em;
                margin-bottom: 0.9rem;
            }

            .hero-title {
                font-size: 2.6rem; font-weight: 900; color: #ffffff;
                letter-spacing: -0.03em; line-height: 1.1;
                margin-bottom: 0.5rem;
                text-shadow: 0 2px 20px rgba(139,92,246,0.4);
            }
            .hero-title-accent { color: #a78bfa; }

            .hero-subtitle {
                font-size: 0.95rem; color: rgba(255,255,255,0.65);
                font-weight: 400; margin-bottom: 1.5rem;
                max-width: 560px; line-height: 1.55;
            }

            /* stat chips row */
            .hero-stats {
                display: flex; gap: 0.9rem; flex-wrap: wrap;
                margin-bottom: 1.3rem;
            }
            .hero-stat {
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 14px; padding: 0.55rem 1rem;
                display: flex; flex-direction: column; align-items: center;
                min-width: 80px; backdrop-filter: blur(4px);
                transition: background 0.2s;
            }
            .hero-stat:hover { background: rgba(255,255,255,0.11); }
            .hero-stat-value {
                font-size: 1.35rem; font-weight: 900; color: #ffffff;
                line-height: 1; letter-spacing: -0.02em;
            }
            .hero-stat-label {
                font-size: 0.62rem; font-weight: 600; color: rgba(255,255,255,0.5);
                text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem;
            }
            .hero-stat-value-blue   { color: #93c5fd; }
            .hero-stat-value-purple { color: #c4b5fd; }
            .hero-stat-value-pink   { color: #f9a8d4; }
            .hero-stat-value-amber  { color: #fcd34d; }
            .hero-stat-value-green  { color: #6ee7b7; }

            /* tag pills */
            .hero-badges { display: flex; gap: 0.45rem; flex-wrap: wrap; }
            .hero-badge {
                background: rgba(255,255,255,0.08);
                color: rgba(255,255,255,0.80);
                font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.07em; padding: 0.25rem 0.75rem;
                border-radius: 20px;
                border: 1px solid rgba(255,255,255,0.12);
                backdrop-filter: blur(3px);
            }

            /* ── Horizon card ────────────────────────────────── */
            .horizon-card {
                background: #ffffff; border: 1.5px solid #e2e8f0;
                border-radius: 16px; padding: 1.2rem 1.5rem;
                box-shadow: 0 2px 12px rgba(15,23,42,0.05);
                margin-bottom: 1.2rem;
                display: flex; align-items: center; gap: 1.2rem;
            }
            .horizon-label {
                font-size: 0.78rem; font-weight: 700; color: #6366f1;
                text-transform: uppercase; letter-spacing: 0.08em;
                margin-bottom: 0.25rem;
            }
            .horizon-desc { font-size: 0.82rem; color: #64748b; font-weight: 500; }

            /* ── Config notice banner ────────────────────────── */
            .config-notice-wrap {
                background: #fefce8; border: 1px solid #fde68a;
                border-radius: 12px; padding: 0.7rem 1rem;
                margin-bottom: 1rem; font-size: 0.84rem; color: #92400e;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="dashboard-header">
            <div class="dashboard-title">CGM Forecast Analysis Dashboard</div>
            <div class="dashboard-subtitle">
                Short-term glucose forecasting, model comparison, and within-vs-cross dataset interpretation
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-box">
            This dashboard analyzes how different ML models forecast future glucose using recent CGM and
            treatment context features.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(title: str, items: list[tuple[str, str]], note: str) -> None:
    tiles = "".join(
        [
            """
            <div class="result-item">
                <div class="result-label">{label}</div>
                <div class="result-value">{value}</div>
            </div>
            """.format(label=label, value=value)
            for label, value in items
        ]
    )
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">{title}</div>
            <div class="result-grid">{tiles}</div>
            <div class="result-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_card(title: str, lines: list[str]) -> None:
    body = "".join([f"<div>{line}</div>" for line in lines])
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-title">{title}</div>
            <div class="summary-text">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def classify_agreement(spread: float) -> str:
    if spread <= 10:
        return "High agreement across models"
    if spread <= 30:
        return "Moderate disagreement across models"
    return "Large disagreement across models; interpret with caution"


def is_within_dataset(experiment: str) -> bool:
    train, test = experiment.split("->")
    return train == test


def summarize_metrics(metrics_df: pd.DataFrame) -> dict[str, float]:
    return {
        "mae": metrics_df["mae"].mean(),
        "rmse": metrics_df["rmse"].mean(),
        "r2": metrics_df["r2"].mean(),
    }


def summarize_predictions(values: list[float]) -> dict[str, float]:
    series = pd.Series(values)
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "avg": float(series.mean()),
        "median": float(series.median()),
        "spread": float(series.max() - series.min()),
    }


def predict_models(
    models: list[ExperimentModel],
    input_df: pd.DataFrame,
    experiment: str | None = None,
    horizon: str | None = None,
) -> tuple[list[dict[str, float | str]], list[str]]:
    predictions: list[dict[str, float | str]] = []
    errors: list[str] = []
    for model_entry in models:
        try:
            model = joblib.load(model_entry.path)
            prediction = float(model.predict(input_df)[0])
            row: dict[str, float | str] = {
                "Model": model_entry.model_name,
                "Predicted Glucose": prediction,
            }
            if experiment is not None:
                row["Experiment"] = experiment
            if horizon is not None:
                row["Horizon"] = horizon
            predictions.append(row)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{model_entry.model_name}: {exc}")
    return predictions, errors


def predict_dataset_type(
    classifier,
    input_df: pd.DataFrame,
) -> tuple[str, float, float]:
    if classifier is None:
        return "Unknown", 0.0, 0.0
    try:
        predicted = str(classifier.predict(input_df)[0])
    except Exception:
        predicted = "Unknown"

    azt1d_prob = 0.0
    hupa_prob = 0.0
    try:
        probabilities = classifier.predict_proba(input_df)[0]
        class_labels = list(classifier.classes_)
        class_map = {label: float(prob) for label, prob in zip(class_labels, probabilities)}
        azt1d_prob = class_map.get("AZT1D", 0.0)
        hupa_prob = class_map.get("HUPA", 0.0)
    except Exception:
        pass
    return predicted, azt1d_prob, hupa_prob


def select_best_model_from_metrics(
    metrics_df: pd.DataFrame,
    experiment: str,
    horizon: str,
) -> tuple[str | None, str]:
    required_columns = {"experiment", "horizon", "model", "mae", "rmse", "r2"}
    if metrics_df.empty or not required_columns.issubset(metrics_df.columns):
        return None, "Metrics unavailable; default model selection applied."

    selection = metrics_df[
        (metrics_df["experiment"] == experiment) & (metrics_df["horizon"] == horizon)
    ]
    if selection.empty:
        return None, "Metrics unavailable for this experiment; default model selection applied."

    best_row = selection.sort_values(by=["mae", "rmse"], ascending=[True, True]).iloc[0]
    reason = (
        f"Selected because it has the lowest MAE ({best_row['mae']:.2f}) "
        f"for {experiment} at {horizon}."
    )
    return str(best_row["model"]), reason


def predict_best_model(
    models: list[ExperimentModel],
    input_df: pd.DataFrame,
    preferred_model: str | None,
) -> tuple[float | None, str | None, str | None]:
    if not models:
        return None, None, "No models available for this experiment."

    selected_model = None
    if preferred_model:
        selected_model = next(
            (model for model in models if model.model_name == preferred_model), None
        )
    if selected_model is None:
        selected_model = models[0]

    try:
        model = joblib.load(selected_model.path)
        prediction = float(model.predict(input_df)[0])
        return prediction, selected_model.model_name, None
    except Exception as exc:  # noqa: BLE001
        return None, selected_model.model_name, f"{selected_model.model_name}: {exc}"


def collect_all_predictions(
    model_registry: dict[str, list[ExperimentModel]],
    input_df: pd.DataFrame,
    horizon_filter: str | None = None,
) -> tuple[list[dict[str, float | str]], list[str]]:
    all_predictions: list[dict[str, float | str]] = []
    all_errors: list[str] = []
    for experiment_key, models in model_registry.items():
        experiment, horizon = split_experiment_key(experiment_key)
        if horizon_filter and horizon != horizon_filter:
            continue
        predictions, errors = predict_models(models, input_df, experiment=experiment, horizon=horizon)
        all_predictions.extend(predictions)
        all_errors.extend(errors)
    return all_predictions, all_errors


def main() -> None:
    st.set_page_config(
        page_title="CGM Glucose Forecast",
        page_icon="🩸",
        layout="wide",
    )
    inject_css()

    # ── Hero banner ──────────────────────────────────────────────────────────
    st.markdown(
        """
<div class="hero-banner">
<div class="hero-orb-bl"></div>
<div class="hero-inner">
<div class="hero-eyebrow">&#9889;&nbsp; ML-Powered Glucose Forecasting Research Platform</div>
<div class="hero-title">&#129656; CGM <span class="hero-title-accent">Glucose</span> Forecast</div>
<div class="hero-subtitle">
Short-term blood glucose prediction for Type 1 Diabetes &nbsp;&middot;&nbsp;
MAE-weighted ensemble across two clinical datasets &nbsp;&middot;&nbsp;
30 &amp; 60 minute ahead forecasting
</div>
<div class="hero-stats">
<div class="hero-stat"><span class="hero-stat-value hero-stat-value-purple">5</span><span class="hero-stat-label">ML Models</span></div>
<div class="hero-stat"><span class="hero-stat-value hero-stat-value-blue">2</span><span class="hero-stat-label">Datasets</span></div>
<div class="hero-stat"><span class="hero-stat-value hero-stat-value-amber">30&amp;60</span><span class="hero-stat-label">Min Horizon</span></div>
<div class="hero-stat"><span class="hero-stat-value hero-stat-value-pink">48</span><span class="hero-stat-label">Experiments</span></div>
<div class="hero-stat"><span class="hero-stat-value hero-stat-value-green">A+B</span><span class="hero-stat-label">Clarke Zones</span></div>
</div>
<div class="hero-badges">
<span class="hero-badge">&#128452; AZT1D Dataset</span>
<span class="hero-badge">&#127973; HUPA-UCM Dataset</span>
<span class="hero-badge">&#9201; 30 &amp; 60 min Horizons</span>
<span class="hero-badge">&#129302; Ensemble Prediction</span>
<span class="hero-badge">&#128208; Clarke Error Grid</span>
<span class="hero-badge">&#128202; Wilcoxon Analysis</span>
</div>
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    model_registry = load_model_registry()
    if not model_registry:
        st.error("No trained models found in the `models/` folder. Run the training pipeline first.")
        return

    feature_config = load_feature_config()
    feature_columns = feature_config.get("feature_columns", [])
    context_features = feature_config.get("context_features", [])
    lag_count = feature_config.get("lag_count", 0)
    roc_count = feature_config.get("roc_count", 0)
    rolling_window = feature_config.get("rolling_window", 0)
    auto_feature_names: list[str] = feature_config.get("auto_feature_names", [])
    manual_feature_columns = [f for f in feature_columns if f not in auto_feature_names]
    if not feature_columns:
        st.error("Feature metadata not found. Ensure `reports/feature_config.json` exists.")
        return

    common_columns, common_source = load_common_columns(feature_config)

    horizon_options = get_available_horizons(model_registry)
    if not horizon_options:
        st.error("No forecast horizons were found in the saved models.")
        return

    # ── Horizon selector card ────────────────────────────────────────────────
    hz_col, _ = st.columns([1, 3])
    with hz_col:
        st.markdown(
            '<div class="horizon-label">⏱ Forecast Horizon</div>'
            '<div class="horizon-desc">30m = next 30 min &nbsp;·&nbsp; 60m = next 1 hour</div>',
            unsafe_allow_html=True,
        )
        selected_horizon = st.selectbox(
            "",
            horizon_options,
            index=0,
            label_visibility="collapsed",
        )

    inputs = build_input_form(manual_feature_columns)
    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("📊  Run Forecast", type="primary")

    if run_clicked:
        lag_values = [inputs.get(f"glucose_lag_{i}", 0.0) for i in range(1, lag_count + 1)]
        if lag_count > 0 and all(v == 0.0 for v in lag_values):
            st.warning("⚠️ All glucose lag inputs are 0. Please enter real CGM readings (30–600 mg/dL).")

        with st.spinner("Running forecast..."):
            try:
                # ── Build derived features ──────────────────────────────────────
                derived_inputs = dict(inputs)
                for i in range(1, roc_count + 1):
                    derived_inputs[f"glucose_roc_{i}"] = (
                        derived_inputs.get(f"glucose_lag_{i}", 0.0) - derived_inputs.get(f"glucose_lag_{i + 1}", 0.0)
                    )
                if rolling_window > 0:
                    roll_vals = [derived_inputs.get(f"glucose_lag_{j}", 0.0) for j in range(1, rolling_window + 1)]
                    derived_inputs[f"glucose_roll_mean_{rolling_window}"] = sum(roll_vals) / len(roll_vals)
                    derived_inputs[f"glucose_roll_std_{rolling_window}"] = float(
                        pd.Series(roll_vals).std() if len(roll_vals) > 1 else 0.0
                    )

                input_df = pd.DataFrame([{k: derived_inputs.get(k, 0.0) for k in feature_columns}])

                # ── Dataset type detection ──────────────────────────────────────
                classifier = load_dataset_classifier()
                predicted_dataset, azt1d_prob, hupa_prob = predict_dataset_type(classifier, input_df)
                high_confidence = max(azt1d_prob, hupa_prob) >= DATASET_CONFIDENCE_THRESHOLD
                metrics_df = load_metrics()

                # ── Select and run best model(s) ────────────────────────────────
                if high_confidence and predicted_dataset in {"AZT1D", "HUPA"}:
                    exp = f"{predicted_dataset}->{predicted_dataset}"
                    exp_models = get_models_for_experiment(model_registry, exp, selected_horizon)
                    preferred_model, _ = select_best_model_from_metrics(metrics_df, exp, selected_horizon)
                    final_pred, model_name_used, pred_err = predict_best_model(exp_models, input_df, preferred_model)
                    if pred_err:
                        st.warning(f"Model warning: {pred_err}")
                    final_model_info = f"{predicted_dataset} → {model_name_used or 'N/A'}"
                    within_predictions: dict[str, float] = {}
                    within_model_names: dict[str, str] = {}
                    within_mae: dict[str, float] = {}
                    weights: dict[str, float] = {}
                else:
                    within_predictions = {}
                    within_model_names = {}
                    within_mae = {}
                    pred_errors: list[str] = []
                    for dataset_type in ["AZT1D", "HUPA"]:
                        exp = f"{dataset_type}->{dataset_type}"
                        exp_models = get_models_for_experiment(model_registry, exp, selected_horizon)
                        preferred_model, _ = select_best_model_from_metrics(metrics_df, exp, selected_horizon)
                        pv, mn, pe = predict_best_model(exp_models, input_df, preferred_model)
                        if pe:
                            pred_errors.append(f"{dataset_type}: {pe}")
                        if pv is not None:
                            within_predictions[dataset_type] = pv
                        if mn:
                            within_model_names[dataset_type] = mn
                        sel = metrics_df[
                            (metrics_df["experiment"] == exp)
                            & (metrics_df["horizon"] == selected_horizon)
                            & (metrics_df["model"] == preferred_model)
                        ]
                        if not sel.empty:
                            within_mae[dataset_type] = float(sel.iloc[0]["mae"])

                    if pred_errors:
                        st.warning("Some models had errors: " + " | ".join(pred_errors))

                    weights = {
                        dt: 1.0 / within_mae[dt]
                        for dt in within_predictions
                        if dt in within_mae and within_mae[dt] > 0
                    }
                    if within_predictions and len(weights) == len(within_predictions) and sum(weights.values()) > 0:
                        total_weight = sum(weights.values())
                        final_pred = sum(within_predictions[dt] * weights[dt] for dt in within_predictions) / total_weight
                        final_model_info = (
                            f"MAE-weighted ensemble — "
                            f"AZT1D {within_model_names.get('AZT1D','N/A')} ({within_predictions.get('AZT1D', 0):.1f}) + "
                            f"HUPA {within_model_names.get('HUPA','N/A')} ({within_predictions.get('HUPA', 0):.1f})"
                        )
                    elif within_predictions:
                        final_pred = sum(within_predictions.values()) / len(within_predictions)
                        final_model_info = "Mean of available models"
                    else:
                        final_pred = None
                        final_model_info = "No predictions available"

            except Exception as _exc:
                st.error(f"Forecast error: {_exc}")
                st.exception(_exc)
                final_pred = None
                final_model_info = "Error during prediction"
                within_predictions = {}
                within_mae = {}
                weights = {}
                within_model_names = {}
                high_confidence = False

        # ── Show result ──────────────────────────────────────────────────────
        st.divider()
        if final_pred is not None:
            if 70 <= final_pred <= 180:
                zone_icon, zone_text = "🟢", "Normal Range (70–180 mg/dL)"
                st.success(f"**Predicted Glucose ({selected_horizon}):** {final_pred:.1f} mg/dL  —  {zone_icon} {zone_text}")
            elif final_pred < 70:
                zone_icon, zone_text = "🟡", "Hypoglycemia Risk (< 70 mg/dL)"
                st.warning(f"**Predicted Glucose ({selected_horizon}):** {final_pred:.1f} mg/dL  —  {zone_icon} {zone_text}")
            else:
                zone_icon, zone_text = "🔴", "Hyperglycemia Risk (> 180 mg/dL)"
                st.error(f"**Predicted Glucose ({selected_horizon}):** {final_pred:.1f} mg/dL  —  {zone_icon} {zone_text}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Glucose", f"{final_pred:.1f} mg/dL")
            c2.metric("Forecast Horizon", selected_horizon)
            c3.metric("Clinical Zone", zone_text.split(" (")[0])
            st.caption(f"**Model:** {final_model_info}")

            if not high_confidence and within_predictions:
                with st.expander("Model Breakdown", expanded=False):
                    for dt, pv in within_predictions.items():
                        mv = within_mae.get(dt)
                        wv = weights.get(dt, 0)
                        st.write(
                            f"**{dt}** ({within_model_names.get(dt,'?')}): **{pv:.2f} mg/dL**"
                            + (f"  —  MAE={mv:.2f}, weight={wv:.4f}" if mv else "")
                        )
        elif final_model_info != "Error during prediction":
            st.error("Could not generate a prediction. Check that models are loaded correctly.")

        # ── Model Performance Metrics ────────────────────────────────────────
        st.divider()
        st.subheader("📈 Model Performance Metrics")
        metrics_df_show = load_metrics()
        required_cols = {"experiment", "horizon", "model", "mae", "rmse", "r2"}
        if not metrics_df_show.empty and required_cols.issubset(metrics_df_show.columns):
            disp = metrics_df_show[metrics_df_show["horizon"] == selected_horizon].copy()
            if not disp.empty:
                rename_map = {"experiment": "Experiment", "model": "Model", "mae": "MAE", "rmse": "RMSE", "r2": "R²"}
                for col, lbl in {"mard": "MARD (%)", "tir": "TIR (%)", "clarke_a": "Clarke A (%)"}.items():
                    if col in disp.columns:
                        rename_map[col] = lbl
                disp_table = disp[[c for c in rename_map if c in disp.columns]].rename(columns=rename_map)
                fmt = {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.3f}",
                       "MARD (%)": "{:.1f}", "TIR (%)": "{:.1f}", "Clarke A (%)": "{:.1f}"}
                st.dataframe(
                    disp_table.style.format({k: v for k, v in fmt.items() if k in disp_table.columns}),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("MAE/RMSE: lower = better · R² higher = better · Clarke A >80% = clinically safe")

                # ── Charts ──────────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                _PALETTE = px.colors.qualitative.Pastel

                # Row 1: MAE + RMSE grouped bar
                ch1, ch2 = st.columns(2)
                with ch1:
                    fig_mae = px.bar(
                        disp, x="experiment", y="mae", color="model",
                        barmode="group", title=f"MAE by Experiment & Model ({selected_horizon})",
                        labels={"experiment": "Experiment", "mae": "MAE (mg/dL)", "model": "Model"},
                        color_discrete_sequence=_PALETTE,
                    )
                    fig_mae.update_layout(xaxis_tickangle=-20, legend_title_text="Model",
                                          plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
                                          font=dict(family="Inter", size=12))
                    st.plotly_chart(fig_mae, use_container_width=True)

                with ch2:
                    fig_rmse = px.bar(
                        disp, x="experiment", y="rmse", color="model",
                        barmode="group", title=f"RMSE by Experiment & Model ({selected_horizon})",
                        labels={"experiment": "Experiment", "rmse": "RMSE (mg/dL)", "model": "Model"},
                        color_discrete_sequence=_PALETTE,
                    )
                    fig_rmse.update_layout(xaxis_tickangle=-20, legend_title_text="Model",
                                           plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
                                           font=dict(family="Inter", size=12))
                    st.plotly_chart(fig_rmse, use_container_width=True)

                # Row 2: R² + Clarke A grouped bar
                ch3, ch4 = st.columns(2)
                with ch3:
                    fig_r2 = px.bar(
                        disp, x="experiment", y="r2", color="model",
                        barmode="group", title=f"R² by Experiment & Model ({selected_horizon})",
                        labels={"experiment": "Experiment", "r2": "R²", "model": "Model"},
                        color_discrete_sequence=_PALETTE,
                    )
                    fig_r2.update_layout(xaxis_tickangle=-20, legend_title_text="Model",
                                         plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
                                         font=dict(family="Inter", size=12))
                    st.plotly_chart(fig_r2, use_container_width=True)

                with ch4:
                    if "clarke_a" in disp.columns:
                        fig_clarke = px.bar(
                            disp, x="experiment", y="clarke_a", color="model",
                            barmode="group", title=f"Clarke Zone A (%) ({selected_horizon})",
                            labels={"experiment": "Experiment", "clarke_a": "Clarke A (%)", "model": "Model"},
                            color_discrete_sequence=_PALETTE,
                        )
                        fig_clarke.add_hline(y=80, line_dash="dash", line_color="red",
                                             annotation_text="80% clinical threshold",
                                             annotation_position="top left")
                        fig_clarke.update_layout(xaxis_tickangle=-20, legend_title_text="Model",
                                                  plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
                                                  font=dict(family="Inter", size=12))
                        st.plotly_chart(fig_clarke, use_container_width=True)

                # Row 3: Best model wins pie + Radar chart
                ch5, ch6 = st.columns(2)
                with ch5:
                    best_per_exp = disp.loc[disp.groupby("experiment")["mae"].idxmin(), ["experiment", "model"]]
                    wins = best_per_exp["model"].value_counts().reset_index()
                    wins.columns = ["Model", "Wins"]
                    fig_pie = px.pie(
                        wins, names="Model", values="Wins",
                        title=f"Best Model Wins (lowest MAE per experiment, {selected_horizon})",
                        color_discrete_sequence=_PALETTE, hole=0.35,
                    )
                    fig_pie.update_traces(textinfo="label+percent", pull=[0.05] * len(wins))
                    fig_pie.update_layout(font=dict(family="Inter", size=12),
                                          paper_bgcolor="#f8fafc")
                    st.plotly_chart(fig_pie, use_container_width=True)

                with ch6:
                    # Radar: avg metrics per model across experiments
                    radar_metrics = ["mae", "rmse", "r2"]
                    radar_df = disp.groupby("model")[radar_metrics].mean().reset_index()
                    # Normalise 0-1 (inverted for mae/rmse so higher = better)
                    for col in ["mae", "rmse"]:
                        col_max = radar_df[col].max()
                        col_min = radar_df[col].min()
                        if col_max != col_min:
                            radar_df[col + "_norm"] = 1 - (radar_df[col] - col_min) / (col_max - col_min)
                        else:
                            radar_df[col + "_norm"] = 1.0
                    r2_min = radar_df["r2"].min()
                    r2_max = radar_df["r2"].max()
                    if r2_max != r2_min:
                        radar_df["r2_norm"] = (radar_df["r2"] - r2_min) / (r2_max - r2_min)
                    else:
                        radar_df["r2_norm"] = 1.0
                    categories = ["MAE (inv)", "RMSE (inv)", "R²"]
                    fig_radar = go.Figure()
                    for _, row in radar_df.iterrows():
                        vals = [row["mae_norm"], row["rmse_norm"], row["r2_norm"]]
                        vals += vals[:1]
                        fig_radar.add_trace(go.Scatterpolar(
                            r=vals, theta=categories + [categories[0]],
                            fill="toself", name=row["model"]
                        ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        title=f"Model Radar — Normalised Metrics ({selected_horizon})",
                        font=dict(family="Inter", size=12),
                        paper_bgcolor="#f8fafc",
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("No metrics found for this horizon.")
        else:
            st.info("Metrics not found. Run the training pipeline first.")

        # ── Wilcoxon Statistical Significance Tests ──────────────────────────
        st.divider()
        st.subheader("📊 Wilcoxon Signed-Rank Test — Model Comparisons")
        wilcoxon_path = OUTPUTS_DIR / "wilcoxon_comparisons.csv"
        if wilcoxon_path.exists():
            wilcoxon_df = pd.read_csv(wilcoxon_path)
            required_wcols = {"experiment", "horizon", "model_1", "model_2", "better_model",
                              "mean_mae_1", "mean_mae_2", "p_value", "significant_p05"}
            if required_wcols.issubset(wilcoxon_df.columns):
                wfiltered = wilcoxon_df[wilcoxon_df["horizon"] == selected_horizon].copy()

                col_exp, col_sig = st.columns([2, 1])
                with col_exp:
                    exp_options = ["All Experiments"] + sorted(wfiltered["experiment"].unique().tolist())
                    selected_wexp = st.selectbox("Filter by Experiment", exp_options, key="wilcoxon_exp_filter")
                with col_sig:
                    sig_only = st.checkbox("Show only significant results (p < 0.05)", value=False, key="wilcoxon_sig_only")

                if selected_wexp != "All Experiments":
                    wfiltered = wfiltered[wfiltered["experiment"] == selected_wexp]
                if sig_only:
                    wfiltered = wfiltered[wfiltered["significant_p05"] == True]

                if not wfiltered.empty:
                    wdisp = wfiltered[["experiment", "model_1", "model_2", "better_model",
                                       "mean_mae_1", "mean_mae_2", "p_value", "significant_p05"]].copy()
                    wdisp = wdisp.rename(columns={
                        "experiment": "Experiment",
                        "model_1": "Model 1",
                        "model_2": "Model 2",
                        "better_model": "Better Model",
                        "mean_mae_1": "MAE (Model 1)",
                        "mean_mae_2": "MAE (Model 2)",
                        "p_value": "p-value",
                        "significant_p05": "Significant (p<0.05)",
                    })

                    def _highlight_sig(row):
                        color = "background-color: #d1fae5" if row["Significant (p<0.05)"] else ""
                        return [color] * len(row)

                    st.dataframe(
                        wdisp.style
                            .apply(_highlight_sig, axis=1)
                            .format({"MAE (Model 1)": "{:.3f}", "MAE (Model 2)": "{:.3f}", "p-value": "{:.4f}"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                    sig_count = wfiltered["significant_p05"].sum()
                    total_count = len(wfiltered)
                    st.caption(
                        f"**{sig_count}/{total_count}** comparisons are statistically significant (p < 0.05). "
                        "Highlighted rows = significant difference between the two models. "
                        "Wilcoxon Signed-Rank Test on per-subject MAE values."
                    )

                    # ── Wilcoxon Charts ──────────────────────────────────────
                    st.markdown("<br>", unsafe_allow_html=True)
                    _W_PALETTE = px.colors.qualitative.Safe

                    # Use the full horizon-filtered data for charts (ignore experiment/sig filter)
                    wfull = wilcoxon_df[wilcoxon_df["horizon"] == selected_horizon].copy()

                    wch1, wch2 = st.columns(2)
                    with wch1:
                        # Significant vs Not-Significant pie per experiment
                        sig_counts = wfull.groupby("experiment")["significant_p05"].sum().reset_index()
                        sig_counts.columns = ["Experiment", "Significant"]
                        sig_counts["Not Significant"] = wfull.groupby("experiment")["significant_p05"].count().values - sig_counts["Significant"].values
                        sig_melt = sig_counts.melt(id_vars="Experiment", var_name="Result", value_name="Count")
                        fig_sig_bar = px.bar(
                            sig_melt, x="Experiment", y="Count", color="Result",
                            barmode="stack",
                            title=f"Significant vs Non-Significant Comparisons ({selected_horizon})",
                            labels={"Count": "# Comparisons"},
                            color_discrete_map={"Significant": "#f59e0b", "Not Significant": "#cbd5e1"},
                        )
                        fig_sig_bar.update_layout(xaxis_tickangle=-20,
                                                   plot_bgcolor="#fffbf0", paper_bgcolor="#fffbf0",
                                                   font=dict(family="Inter", size=12))
                        st.plotly_chart(fig_sig_bar, use_container_width=True)

                    with wch2:
                        # Overall pie: significant vs not
                        ov_sig = int(wfull["significant_p05"].sum())
                        ov_not = len(wfull) - ov_sig
                        fig_pie_w = px.pie(
                            values=[ov_sig, ov_not],
                            names=["Significant (p<0.05)", "Not Significant"],
                            title=f"Overall Significance — All Experiments ({selected_horizon})",
                            color_discrete_sequence=["#f59e0b", "#cbd5e1"],
                            hole=0.4,
                        )
                        fig_pie_w.update_traces(textinfo="label+value+percent")
                        fig_pie_w.update_layout(font=dict(family="Inter", size=12),
                                                paper_bgcolor="#fffbf0")
                        st.plotly_chart(fig_pie_w, use_container_width=True)

                    # p-value heatmap per experiment
                    wch3, wch4 = st.columns(2)
                    with wch3:
                        for exp_name in sorted(wfull["experiment"].unique()):
                            exp_data = wfull[wfull["experiment"] == exp_name]
                            models_all = sorted(set(exp_data["model_1"].tolist() + exp_data["model_2"].tolist()))
                            pval_matrix = pd.DataFrame(index=models_all, columns=models_all, dtype=float)
                            for _, row in exp_data.iterrows():
                                pval_matrix.loc[row["model_1"], row["model_2"]] = row["p_value"]
                                pval_matrix.loc[row["model_2"], row["model_1"]] = row["p_value"]
                            pval_matrix = pval_matrix.fillna(1.0)
                            fig_hm = px.imshow(
                                pval_matrix.astype(float),
                                text_auto=".3f",
                                color_continuous_scale="YlOrRd",
                                zmin=0, zmax=0.1,
                                title=f"p-value Heatmap: {exp_name} ({selected_horizon})",
                                aspect="auto",
                            )
                            fig_hm.update_layout(font=dict(family="Inter", size=11),
                                                  paper_bgcolor="#fffbf0",
                                                  coloraxis_colorbar_title="p-value")
                            st.plotly_chart(fig_hm, use_container_width=True)
                            break  # show first experiment; user can filter above

                    with wch4:
                        # Better model wins bar chart
                        wins_w = wfull[wfull["significant_p05"] == True]["better_model"].value_counts().reset_index()
                        wins_w.columns = ["Model", "Significant Wins"]
                        if not wins_w.empty:
                            fig_wins = px.bar(
                                wins_w, x="Model", y="Significant Wins",
                                title=f"Significant Wins per Model ({selected_horizon})",
                                color="Model",
                                color_discrete_sequence=["#f59e0b", "#fb923c", "#ef4444", "#a78bfa", "#60a5fa"],
                                labels={"Significant Wins": "# Significant Wins"},
                            )
                            fig_wins.update_layout(showlegend=False,
                                                    plot_bgcolor="#fffbf0", paper_bgcolor="#fffbf0",
                                                    font=dict(family="Inter", size=12))
                            st.plotly_chart(fig_wins, use_container_width=True)
                        else:
                            st.info("No significant wins to display.")

                else:
                    st.info("No Wilcoxon comparisons match the selected filters.")
            else:
                st.warning("wilcoxon_comparisons.csv is missing expected columns.")
        else:
            st.info("Wilcoxon results not found. Run the training pipeline first.")


if __name__ == "__main__":
    main()
