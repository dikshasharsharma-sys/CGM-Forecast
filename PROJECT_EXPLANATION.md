# Project Explanation — CGM Glucose Forecasting

> This document provides a plain-language explanation of the entire project: what it does, why each part exists, and how everything connects. Written for reviewers, collaborators, and paper readers who may not have seen the code.

---

## 1. What Is This Project?

This project builds a machine learning system that **predicts a diabetic patient's blood glucose level 30 or 60 minutes into the future**, using only the readings from their Continuous Glucose Monitor (CGM) device and a few treatment context features (insulin dose, carbohydrate intake, time of day).

Predicting glucose ahead of time is clinically important because:
- Hypoglycaemia (blood glucose < 70 mg/dL) can cause seizures, loss of consciousness, or death
- Hyperglycaemia (blood glucose > 180 mg/dL) causes long-term organ damage
- A 30–60 minute warning gives the patient or their care system time to intervene

---

## 2. The Core Problem: Two Different Datasets

The project uses two publicly available Type 1 Diabetes CGM datasets:

| Dataset | Description |
|---------|-------------|
| **AZT1D** | Real-world clinical CGM records from an insulin pump study |
| **HUPA-UCM** | IoT wearable CGM data with additional lifestyle signals |

These datasets were collected independently, by different teams, using different sensor brands and data formats. This creates a real-world challenge called **domain shift** — a model trained on AZT1D data behaves differently when tested on HUPA data, and vice versa.

**Most existing work only evaluates within one dataset.** This project evaluates all four directional combinations:

```
AZT1D → AZT1D   (within-dataset, same domain)
HUPA  → HUPA    (within-dataset, same domain)
AZT1D → HUPA   (cross-dataset, different domain)
HUPA  → AZT1D  (cross-dataset, different domain)
```

---

## 3. How the Data Is Prepared

### Step 1 — Feature Harmonization
Since the two datasets have different column names and feature sets, only the **columns present in both** are used:

```
glucose            → target variable (what we predict)
basal_rate         → background insulin rate
bolus_volume_delivered → meal/correction insulin dose
carb_input         → grams of carbohydrate eaten
hour               → hour of the day (0–23)
day_of_week        → day of the week (0=Monday, 6=Sunday)
```

Columns unique to one dataset (e.g., `heart_rate`, `steps` from HUPA; `DeviceMode`, `BolusType` from AZT1D) are dropped. This ensures the model architecture is identical regardless of which dataset it is trained or tested on.

### Step 2 — Lag Feature Creation
CGM readings arrive every 5 minutes. Rather than treating each reading in isolation, we create **lag features** — a sliding window of the 18 most recent glucose readings:

```
glucose_lag_1   (5 minutes ago   — most recent)
glucose_lag_2   (10 minutes ago)
glucose_lag_3   (15 minutes ago)
...
glucose_lag_18  (90 minutes ago  — oldest in window)
```

This gives the model a 90-minute history of glucose readings to learn patterns from.

### Step 3 — Forecast Target Creation
Instead of predicting the current glucose, we shift the glucose series **forward in time** to create a future target:

```
glucose_t+30m  =  glucose reading 30 minutes from now  (6 steps × 5 min)
glucose_t+60m  =  glucose reading 60 minutes from now  (12 steps × 5 min)
```

This is the key difference between forecasting and regression — the model learns to predict a value it has never seen yet at prediction time.

---

## 4. The Machine Learning Models

Five models are trained for every experiment + horizon combination:

| Model | Type | Strengths |
|-------|------|-----------|
| **Linear Regression** | Linear | Interpretable baseline; fast |
| **Random Forest** | Ensemble (bagging) | Robust to outliers; handles non-linearity |
| **HistGBM** | Ensemble (boosting) | Fast; handles missing values natively |
| **XGBoost** | Ensemble (boosting) | Strong regularisation; often best performer |
| **LSTM** | Deep learning (RNN) | Designed for sequential/time-series data |

All models use the same 23 input features (18 glucose lags + 5 context features) and the same preprocessing (StandardScaler normalization), making results directly comparable.

**Total trained models: 5 models × 4 experiments × 2 horizons = 40 models**

---

## 5. The Dataset-Origin Classifier

### Why it exists
When the Streamlit app receives new input data from a user, it does not know which dataset the user's device resembles. Rather than asking the user to manually select "AZT1D" or "HUPA", a **secondary classifier** is trained to automatically identify the data origin.

### How it works
A **Logistic Regression classifier** is trained on the same 23 input features, with the label being the dataset of origin (`AZT1D` or `HUPA`). It learns the statistical fingerprint of each dataset's typical glucose patterns and treatment intensities.

| Metric | Value |
|--------|-------|
| Accuracy | 96.29% |
| F1 Score | 95.16% |
| ROC-AUC | 0.920 |

### How it routes predictions
```
New input arrives
       │
       ▼
Classifier runs → confidence score for AZT1D vs HUPA
       │
       ├── Confidence ≥ 65%  →  use only the matching within-dataset models
       │                         (e.g., AZT1D→AZT1D if classified as AZT1D)
       │
       └── Confidence < 65%  →  run BOTH AZT1D→AZT1D and HUPA→HUPA models
                                 blend predictions by inverse MAE weight
                                 (better-performing model gets more weight)
```

This makes the system robust to ambiguous or borderline input data.

---

## 6. Evaluation Metrics

Each of the 40 models is evaluated on a held-out test set using:

| Metric | What It Measures |
|--------|-----------------|
| **MAE** (mg/dL) | Average absolute prediction error |
| **RMSE** (mg/dL) | Root mean squared error — punishes large errors more |
| **R²** | 0 = no predictive power, 1 = perfect prediction |
| **MARD** (%) | Mean absolute relative difference — scale-independent error |
| **TIR** (%) | Percentage of predictions landing in 70–180 mg/dL (normal range) |
| **Clarke A%** | Percentage of predictions in the clinically safe Zone A |

The **Clarke Error Grid** is the standard clinical safety metric for glucose predictions:
- **Zone A:** Clinically accurate — within 20% of true value
- **Zone B:** Acceptable — unlikely to cause harm
- **Zone C/D/E:** Potentially dangerous — would lead to clinical mistreatment

---

## 7. Statistical Analysis — Wilcoxon Tests

Beyond point metrics, the project runs **pairwise Wilcoxon signed-rank tests** between all model combinations for each experiment and horizon. This answers the question: *is the difference between Model A and Model B statistically significant, or just random variation?*

Results are stored in `outputs/wilcoxon_comparisons.csv` and visualised as:
- A significance heatmap (p-value matrix)
- A stacked bar chart (significant vs not-significant comparisons per experiment)
- A wins chart (which models significantly outperform others most often)

---

## 8. The Streamlit App

The `app.py` file is a full interactive dashboard with two sections:

### Section 1 — Live Forecasting
The user fills in:
- 18 lag glucose values (the last ~90 minutes of CGM readings)
- Insulin and carb context (basal rate, bolus volume, carb input)
- Time context (hour of day, day of week)

The app then:
1. Runs the dataset classifier → determines data origin and confidence
2. Selects the appropriate model(s)
3. Returns a glucose prediction at 30m or 60m ahead
4. Shows the Clarke zone of the prediction if a true value is known

### Section 2 — Experiment Results Viewer
The user can explore all 40 experiment results:
- Filter by experiment, horizon, model
- View MAE/RMSE/R²/Clarke bar charts
- View radar charts for multi-metric comparison
- View Wilcoxon significance heatmaps and charts

---

## 9. Key Findings

1. **Within-dataset models consistently outperform cross-dataset transfer** — models generalise best when tested on the same data distribution they were trained on
2. **The 30-minute horizon is substantially more accurate than 60-minute** across all experiments and models — glucose becomes harder to predict further ahead
3. **XGBoost and HistGBM are the strongest models** in within-dataset settings; **LSTM and LinearRegression** are surprisingly competitive in cross-dataset transfer
4. **AZT1D → HUPA cross-transfer** is more successful than HUPA → AZT1D, suggesting AZT1D-trained models generalise better
5. **R² drops significantly at 60 minutes** for cross-dataset experiments (e.g., HUPA→AZT1D 60m R² ≈ -0.002 for LinearRegression), showing the limits of cross-domain generalisation

---

## 10. File-by-File Code Guide

| File | What it does |
|------|-------------|
| `src/config.py` | Defines all path constants (models dir, outputs dir, etc.) |
| `src/data_loader.py` | Loads AZT1D and HUPA raw files, standardises column names, builds harmonization report |
| `src/preprocessing.py` | Adds lag features, creates forecast targets, adds time features, builds sklearn preprocessor |
| `src/modeling.py` | Returns a dictionary of all 5 model instances |
| `src/evaluation.py` | Computes all metrics, Clarke grid, Wilcoxon tests, saves outputs |
| `src/run_experiments.py` | Orchestrates everything: loads data → features → trains all 40 models → classifier → saves results |
| `app.py` | Streamlit dashboard: CSS, input form, prediction logic, charts |

---

## 11. Novelty Summary (for Paper Reference)

This work is novel because:

1. **No prior work** has evaluated all four directional cross-dataset transfers (AZT1D↔HUPA) with the same feature set and same model suite in a single controlled study
2. **The dataset-origin classifier** is a new mechanism for automating domain routing in multi-source CGM systems — the system self-adapts to unknown input distributions
3. **The MAE-weighted ensemble fallback** provides uncertainty-aware prediction when the classifier is not confident — standard clinical ML systems use hard routing only
4. **Feature harmonization is automated** from the data, not manually engineered — the pipeline discovers and uses only genuinely shared features, making it reproducible and extensible to other CGM datasets
5. **Clarke Error Grid is primary**, not an afterthought — every model is evaluated clinically, not just statistically
