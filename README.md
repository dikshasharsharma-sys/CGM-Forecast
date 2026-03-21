# 🩸 CGM Glucose Forecast — Cross-Dataset Short-Term Blood Glucose Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-red)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Short-term blood glucose forecasting for Type 1 Diabetes using machine learning ensembles across two heterogeneous clinical datasets, with 30-minute and 60-minute prediction horizons.**

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features & Novelty](#-key-features--novelty)
3. [Datasets](#-datasets)
4. [Project Structure](#-project-structure)
5. [Models](#-models)
6. [Experiments](#-experiments)
7. [Results Summary](#-results-summary)
8. [Dataset Classifier](#-dataset-classifier)
9. [Streamlit Demo App](#-streamlit-demo-app)
10. [Installation & Setup](#-installation--setup)
11. [How to Run](#-how-to-run)
12. [Evaluation Metrics](#-evaluation-metrics)
13. [Research Contributions](#-research-contributions)

---

## 🔬 Project Overview

This project implements a complete end-to-end **CGM (Continuous Glucose Monitor) glucose forecasting pipeline** that:

- Predicts blood glucose levels **30 minutes** and **60 minutes** ahead
- Evaluates models both **within-dataset** (same domain) and **cross-dataset** (different domain)
- Uses an **automated dataset-origin classifier** that identifies which clinical dataset type a new patient input resembles
- Applies a **MAE-weighted ensemble** when the classifier is uncertain, blending predictions from both dataset domains

This is a **time-series forecasting** problem, not regression on current values. The target is always a future glucose reading shifted forward in time.

---

## ✨ Key Features & Novelty

| Feature | Description |
|--------|-------------|
| **Bidirectional Cross-Dataset Transfer** | All 4 experiment combinations: AZT1D→AZT1D, HUPA→HUPA, AZT1D→HUPA, HUPA→AZT1D |
| **Automated Feature Harmonization** | Only features common to both datasets are used, enabling fair cross-dataset evaluation |
| **Dataset-Origin Classifier** | Logistic Regression classifier (96.3% accuracy, AUC=0.920) auto-detects input data type |
| **Confidence-Gated Ensemble** | When classifier confidence < 65%, predictions are blended from both domains using MAE weights |
| **Clarke Error Grid Analysis** | Clinical safety metric — Zone A+B percentage computed for every model/experiment |
| **Wilcoxon Significance Tests** | Statistical pairwise model comparison across all experiments and horizons |
| **Interactive Streamlit Dashboard** | Live forecasting demo + full experiment result visualization with Plotly charts |

---

## 📊 Datasets

### AZT1D
- **Type:** Real-world Type 1 Diabetes CGM records
- **Sampling interval:** 5 minutes
- **Key columns:** `CGM`, `Basal`, `TotalBolusInsulinDelivered`, `CarbSize`, `EventDateTime`
- **Standardized to:** `glucose`, `basal_rate`, `bolus_volume_delivered`, `carb_input`, `time`, `subject_id`

### HUPA-UCM
- **Type:** IoT wearable + clinical glucose data
- **Sampling interval:** 5 minutes  
- **Key columns:** `glucose`, `basal_rate`, `bolus_volume_delivered`, `carb_input`, `calories`, `heart_rate`, `steps`
- **Raw rows:** ~309,000

### Feature Harmonization
Only columns present in **both** datasets are used for modelling:

```
 glucose          → target variable
 basal_rate       → context feature
 bolus_volume_delivered → context feature
 carb_input       → context feature
 hour             → time feature (derived)
 day_of_week      → time feature (derived)
```

Dropped (not shared): `calories`, `heart_rate`, `steps`, `BolusType`, `CorrectionDelivered`, `DeviceMode`

---

## 📁 Project Structure

```
cgm-forecast/
│
├── app.py                          # Streamlit demo app
├── requirements.txt                # Python dependencies
│
├── src/
│   ├── config.py                   # Path constants
│   ├── data_loader.py              # Dataset loading & harmonization
│   ├── preprocessing.py            # Lag features, forecast targets, time features
│   ├── modeling.py                 # Model definitions
│   ├── evaluation.py               # Metrics, Clarke grid, Wilcoxon tests
│   └── run_experiments.py          # Main training & evaluation pipeline
│
├── data/
│   ├── raw/                        # Original dataset files
│   └── processed/                  # Preprocessed files
│
├── models/
│   ├── *.joblib                    # 40 trained model files (5 models × 4 experiments × 2 horizons)
│   └── dataset_classifier.pkl      # Dataset-origin Logistic Regression classifier
│
├── outputs/
│   ├── metrics.csv                 # All experiment metrics
│   ├── model_comparison.csv        # Best model per experiment
│   ├── wilcoxon_comparisons.csv    # Pairwise Wilcoxon test results
│   └── *.png                       # Metric plots
│
└── reports/
    ├── feature_config.json         # Feature column names & lag/horizon config
    ├── common_columns.json         # Harmonized column list
    ├── dataset_classifier_metrics.json  # Classifier evaluation results
    └── harmonization_report.md     # Dataset harmonization summary
```

---

## 🤖 Models

Five machine learning models are evaluated for every experiment + horizon combination:

| Model | Library | Notes |
|-------|---------|-------|
| **Linear Regression** | scikit-learn | Baseline |
| **Random Forest** | scikit-learn | 10 estimators, max_depth=6 |
| **Histogram Gradient Boosting (HistGBM)** | scikit-learn | Fast GBM with native NaN support |
| **XGBoost** | xgboost | Gradient boosting with regularization |
| **LSTM** | TensorFlow/Keras | Sequence model for temporal patterns |

All models use the same preprocessor (StandardScaler on numerical features) wrapped in a `sklearn.Pipeline`.

---

## 🧪 Experiments

4 experiment types × 2 horizons × 5 models = **40 trained models**

| Experiment | Type | Train Data | Test Data |
|-----------|------|-----------|----------|
| AZT1D → AZT1D | Within-dataset | AZT1D (80%) | AZT1D (20%) |
| HUPA → HUPA | Within-dataset | HUPA (80%) | HUPA (20%) |
| AZT1D → HUPA | Cross-dataset | AZT1D (100%) | HUPA (100%) |
| HUPA → AZT1D | Cross-dataset | HUPA (100%) | AZT1D (100%) |

Horizons: **30 minutes** (6 steps ahead) and **60 minutes** (12 steps ahead) at 5-min sampling intervals.

---

## 📈 Results Summary

### 30-Minute Horizon — Best Models

| Experiment | Best Model | MAE | RMSE | R² | Clarke A% |
|-----------|-----------|-----|------|----|-----------|
| AZT1D → AZT1D | HistGBM | 16.94 | 23.22 | 0.796 | 85.5% |
| HUPA → HUPA | XGBoost | 11.97 | 17.46 | 0.905 | 86.5% |
| AZT1D → HUPA | LinearRegression | 16.21 | 21.78 | 0.855 | 77.8% |
| HUPA → AZT1D | RandomForest | 18.88 | 26.39 | 0.751 | 81.8% |

### 60-Minute Horizon — Best Models

| Experiment | Best Model | MAE | RMSE | R² | Clarke A% |
|-----------|-----------|-----|------|----|-----------|
| AZT1D → AZT1D | XGBoost | 27.93 | 36.69 | 0.511 | 66.3% |
| HUPA → HUPA | XGBoost | 20.81 | 29.26 | 0.736 | 70.8% |
| AZT1D → HUPA | LSTM | 30.96 | 38.81 | 0.543 | 54.5% |
| HUPA → AZT1D | RandomForest | 30.61 | 41.01 | 0.409 | 63.5% |

> **Key finding:** Within-dataset settings consistently outperform cross-dataset transfer, especially at the 60-minute horizon — a clinically important observation about domain shift in CGM data.

---

## 🔍 Dataset Classifier

A **Logistic Regression classifier** is trained on the shared feature space to identify whether new input data resembles the AZT1D or HUPA-UCM distribution.

| Metric | Value |
|--------|-------|
| Accuracy | **96.29%** |
| F1 Score (weighted) | **95.16%** |
| ROC-AUC | **0.920** |
| Train samples | 80,000 |
| Test samples | 20,000 |

**Prediction routing logic:**
- Confidence ≥ 65% → route to the matching within-dataset models
- Confidence < 65% → run both AZT1D→AZT1D and HUPA→HUPA models, blend by inverse MAE weighting

---

## 🖥️ Streamlit Demo App

The interactive dashboard (`app.py`) provides:

- **Live glucose forecasting** — enter CGM lag history + context features → get 30m/60m predictions
- **Dataset-origin detection** — shows which dataset the input resembles and confidence scores
- **Model Performance Metrics** — filterable table + 6 Plotly charts (MAE/RMSE/R²/Clarke bars, pie, radar)
- **Wilcoxon Analysis** — statistical significance table + heatmap + significance charts
- **Clarke Error Grid interpretation** — zone breakdown per experiment

### Run locally:
```bash
streamlit run app.py
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- pip

### Install dependencies
```bash
pip install -r requirements.txt
```

### requirements.txt
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
matplotlib==3.9.0
seaborn==0.13.2
joblib==1.4.2
streamlit==1.38.0
plotly==5.22.0
```

> The trained model `.joblib` files are already included in the `models/` directory — no retraining required to run the app.

---

## ▶️ How to Run

### 1. Run the Streamlit app (no retraining needed)
```bash
streamlit run app.py
```

### 2. Retrain all models from scratch
```bash
python -m src.run_experiments
```
This will:
- Load and harmonize both datasets
- Create lag features and forecast targets
- Train 40 models (5 models × 4 experiments × 2 horizons)
- Train the dataset classifier
- Save all results to `outputs/` and `reports/`

---

## 📐 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error — average prediction error in mg/dL |
| **RMSE** | Root Mean Squared Error — penalises large errors more |
| **R²** | Coefficient of determination — proportion of variance explained |
| **MARD** | Mean Absolute Relative Difference — scale-free relative error |
| **TIR** | Time In Range — % predictions within 70–180 mg/dL |
| **Clarke A%** | Percentage of predictions in the clinically safe Zone A of the Clarke Error Grid |
| **Wilcoxon p-value** | Non-parametric test for statistical significance of model differences |

---

## 🎓 Research Contributions

1. **Bidirectional cross-dataset evaluation framework** — all four train/test dataset combinations evaluated systematically, not just within-dataset performance
2. **Automated feature harmonization** — enables fair and reproducible cross-dataset comparison on heterogeneous clinical datasets
3. **Dataset-origin classifier with confidence routing** — novel mechanism to handle input data from unknown or mixed sources without requiring manual domain labels
4. **MAE-weighted domain ensemble** — uncertainty-aware blending when classifier confidence is below threshold
5. **Clinical safety evaluation** — all experiments evaluated using the Clarke Error Grid in addition to standard regression metrics

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

Developed as part of a research project on short-term CGM glucose forecasting for Type 1 Diabetes.


## Project Overview
This project is a **short-term CGM forecasting pipeline** that predicts **future blood glucose** using recent CGM history and common context features. It supports within-dataset and cross-dataset evaluation across:

1. **AZT1D** – Real-world CGM records for Type 1 diabetes
2. **HUPA-UCM** – Preprocessed glucose and contextual data

The pipeline explicitly predicts:
- **30-minute ahead glucose**
- **60-minute ahead glucose (optional, also implemented)**

## Why This Is Forecasting (Not Current Glucose Regression)
Instead of predicting the current glucose value, the pipeline shifts the CGM series forward in time to create **future targets**. This means each model is trained to forecast what glucose will be **30 or 60 minutes into the future**.

## Target Definition (Time-Series Logic)
We estimate the median CGM sampling interval per dataset and compute steps-ahead:

- **30-minute horizon**: shift by `round(30 / interval_minutes)` steps
- **60-minute horizon**: shift by `round(60 / interval_minutes)` steps

For a 5-minute interval, this becomes:
- 30 minutes → 6 steps ahead
- 60 minutes → 12 steps ahead

The actual estimated intervals and step counts are documented in:
`reports/harmonization_report.md`

## Input Features (Forecasting Inputs)
### 1) Lagged CGM History
We create lag features from past CGM readings:

```
glucose_lag_1  (most recent)
glucose_lag_2
...
glucose_lag_N  (oldest in the window)
```

Lag count is automatically selected based on the datasets’ sampling intervals and is saved in:
`reports/feature_config.json`

### 2) Common Context Features
Only shared features across AZT1D and HUPA are used:
- `basal_rate`
- `bolus_volume_delivered`
- `carb_input`
- `hour`
- `day_of_week`

## Forecasting Experiments
All forecasting experiments are run for **both 30-minute and 60-minute targets**:

1. **AZT1D → AZT1D**
2. **HUPA → HUPA**
3. **AZT1D → HUPA**
4. **HUPA → AZT1D**

## Models Used (Baselines)
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Evaluation Metrics
Each experiment (for each horizon) reports:
- MAE
- RMSE
- R²

Outputs:
- `outputs/metrics.csv`
- `outputs/model_comparison.csv`
- `outputs/mae_by_experiment.png`
- `outputs/rmse_by_experiment.png`
- `outputs/r2_by_experiment.png`

## Interpretation Dashboard (Streamlit App)
The Streamlit app is a **complete interpretation dashboard** so that a single set of inputs reveals:
- Automatic dataset-type detection (AZT1D vs HUPA)
- Confidence-aware routing to the best within-dataset model
- The likely glucose range (min/max/average)
- Whether the selection is within-dataset or cross-dataset
- How strong the cross-dataset performance is compared to within baselines
- The overall prediction range across all dataset settings

### Automatic Dataset Detection & Confidence-Aware Routing
The app no longer asks you to choose a dataset type manually. Instead it:
- Uses a classifier trained on the shared input features to infer whether the input looks more like AZT1D or HUPA
- Reports confidence scores for both datasets
- Routes to the matching within-dataset models only when confidence is high
- Falls back to showing both within-dataset predictions plus a range/mean/median when confidence is low

Cross-dataset models are shown separately in **Research / Cross-Dataset Generalization Analysis** to avoid mixing
research diagnostics with patient-facing predictions.

### Within-Dataset vs Cross-Dataset
- **Within-dataset** means the model is evaluated on the same dataset type it was trained on (AZT1D→AZT1D or HUPA→HUPA). These settings are generally more stable for this project.
- **Cross-dataset** means the model is applied to a different dataset type than it was trained on (AZT1D→HUPA or HUPA→AZT1D). This evaluates generalization across datasets.

### Why Multiple Model Predictions Are Shown
Each experiment has multiple saved models (Linear Regression, Random Forest, Gradient Boosting). Showing all predictions helps reviewers understand agreement or disagreement across modeling approaches.

### Prediction Range Calculation
The dashboard computes:
- **Minimum prediction** across models
- **Maximum prediction** across models
- **Average prediction** across models
This is shown as the expected glucose range for the selected experiment.

### Selected Experiment vs Overall Range
- **Selected experiment prediction** uses only the models under the chosen setting (e.g., AZT1D→HUPA | 30m).
- **Overall prediction range** aggregates predictions across **all** experiment settings (within- and cross-dataset).
- The overall range is an uncertainty-aware summary across all settings rather than a single deterministic output.

### Within-Dataset vs Cross-Dataset Ranges
- **Within-dataset range**: predictions from AZT1D→AZT1D and HUPA→HUPA settings.
- **Cross-dataset range**: predictions from AZT1D→HUPA and HUPA→AZT1D settings.
- Comparing these ranges helps you understand stability (within) versus domain-shift uncertainty (cross).

### Model Agreement / Disagreement
The app looks at the spread between model outputs:
- **High agreement** → models are close together
- **Moderate disagreement** → some variation
- **Large disagreement** → interpret with caution

The default thresholds used in the app are:
- **High agreement:** spread ≤ 10 mg/dL
- **Moderate disagreement:** spread ≤ 30 mg/dL
- **Large disagreement:** spread > 30 mg/dL

### Metrics Interpretation (MAE, RMSE, R²)
- **Lower MAE is better**
- **Lower RMSE is better**
- **Positive R² is better; negative R² suggests weak generalization**

### Why Cross-Dataset Comparison Matters
This project explicitly measures how models trained on one dataset perform on another. Comparing cross-dataset metrics against within-dataset baselines highlights the generalization strength of the pipeline. The dashboard summarizes this by comparing averaged MAE, RMSE, and R² across models for each experiment.

## How This Differs From Real-Time Monitoring
Real-time monitoring shows the **current** glucose. This project forecasts what glucose will be **in the future**, which is crucial for proactive insulin adjustments and early warnings.

## Live Patient Demo (How Inputs Should Be Entered)
For a live demo using the Streamlit app:
1. Collect the most recent CGM readings (at the dataset’s sampling interval).
2. Fill **glucose_lag_1** with the latest reading, **glucose_lag_2** with the previous reading, etc.
3. Enter current contextual values (basal rate, bolus delivered, carbs, hour, day of week).
4. Select the forecast horizon (e.g., 30m or 60m).
5. The app outputs all model predictions plus a clear interpretation summary.

## Folder Structure
```
CGM/
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
├── reports/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── modeling.py
│   ├── preprocessing.py
│   └── run_experiments.py
├── requirements.txt
└── README.md
```

## How to Run
1. Extract datasets into:
   - `data/raw/AZT1D/AZT1D 2025/CGM Records/...`
   - `data/raw/HUPA/HUPA-UCM Diabetes Dataset/Preprocessed/...`

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train forecasting models:
```bash
python -m src.run_experiments
```

## Streamlit App
```bash
streamlit run app.py
```

The app clearly shows:
- Detected dataset type and confidence
- Automatic routing to the best within-dataset model when confidence is high
- Uncertainty handling with ranges when confidence is low
- Selected horizon
- Required CGM lag inputs and context features
- Model-by-model predictions
- Prediction range and agreement assessment
- Overall prediction range across all dataset settings
- Within-dataset vs cross-dataset range breakdown
- Within vs cross-dataset interpretation
- Experiment metrics (MAE, RMSE, R²)
- Cross-dataset comparison against within baselines

### Files the App Reads From
- `models/*.joblib` (saved models for each experiment and horizon)
- `models/dataset_classifier.pkl` (dataset-type classifier)
- `outputs/metrics.csv` (evaluation metrics for each experiment)
- `reports/feature_config.json` (feature list and lag configuration)
- `reports/common_columns.json` (shared standardized columns across datasets)
- `reports/dataset_classifier_metrics.json` (dataset classifier performance summary)

`reports/common_columns.json` is generated during training in `src/run_experiments.py` by taking the
intersection of the standardized AZT1D and HUPA columns. If this file is missing, the Streamlit app
will automatically regenerate it (preferring `reports/harmonization_report.md`, falling back to the
dataset column intersection, and finally using the saved model/config feature metadata). The app will
continue running even if the common column list is unavailable.

## Assumptions & Limitations
- Forecasting assumes regular CGM sampling intervals.
- If AZT1D and HUPA sampling intervals differ, horizons are computed per dataset and documented.
- Lag features represent the most recent window of CGM history, but cross-dataset differences may affect transfer performance.
- For runtime stability, training/testing datasets are capped at 100k rows per split (see `MAX_TRAIN_SAMPLES` and `MAX_TEST_SAMPLES`).
- This is a baseline model; no advanced time-series models or personalization are included.
