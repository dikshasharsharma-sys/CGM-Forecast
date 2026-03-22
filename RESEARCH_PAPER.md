# Bidirectional Cross-Dataset Evaluation of Machine Learning Ensembles for Short-Term Blood Glucose Forecasting in Type 1 Diabetes: An Automated Harmonization Framework

**Diksha Sharma**
Department of Computer Science / Biomedical Engineering
[Institution Name]
dikshasharsharma@gmail.com

---

## Abstract

**Background and Objective:** Short-term blood glucose (BG) forecasting constitutes a critical component of hypoglycaemia prevention and closed-loop insulin delivery systems for individuals with Type 1 Diabetes Mellitus (T1DM). Existing literature predominantly evaluates predictive models under within-cohort conditions, thereby obscuring the generalizability of these systems across heterogeneous patient populations, device platforms, and clinical environments — a limitation of considerable translational relevance.

**Methods:** The present study introduces a systematic bidirectional cross-dataset evaluation framework applied to two independent clinical datasets: AZT1D (n = 15,177 test samples) and HUPA-UCM (n = 50,000 test samples). Five machine learning architectures — Ordinary Least Squares Regression (LR), Random Forest (RF), Histogram-based Gradient Boosting (HistGBM), Extreme Gradient Boosting (XGBoost), and Long Short-Term Memory recurrent networks (LSTM) — were evaluated under four experimental transfer conditions at prediction horizons of 30 minutes (PH30) and 60 minutes (PH60), yielding 40 model–experiment combinations in total. An automated feature harmonization pipeline was developed to reconcile schema heterogeneity between datasets, and a binary dataset-origin classifier was trained to facilitate distribution-aware ensemble routing.

**Results:** Within-cohort evaluation demonstrated best-performing mean absolute error (MAE) of 12.05 mg/dL (R² = 0.904) for HistGBM on HUPA at PH30, and 16.94 mg/dL (R² = 0.796) on AZT1D at PH30. Cross-dataset transfer exhibited significant directional asymmetry (p < 0.05, Wilcoxon signed-rank test), with AZT1D→HUPA achieving MAE as low as 16.21 mg/dL under linear regression at PH30, whereas HUPA→AZT1D transfer degraded substantially at PH60 (LR R² = −0.002; LSTM R² = −0.143). The dataset-origin classifier achieved 96.3% accuracy, F1-score of 0.952, and ROC-AUC of 0.920. Clarke Error Grid Zone A membership ranged from 81.7% to 86.5% across within-cohort configurations at PH30.

**Conclusions:** The proposed framework provides the first systematic bidirectional cross-dataset benchmark for T1DM glucose prediction with fully automated feature harmonization, quantifying the directional generalization penalty and identifying model-specific transfer robustness profiles. These findings establish a methodological foundation for deploying clinically transferable BG prediction systems across diverse real-world settings.

**Keywords:** blood glucose forecasting; continuous glucose monitoring; Type 1 Diabetes Mellitus; cross-dataset generalization; machine learning ensemble; long short-term memory; feature harmonization; Clarke Error Grid; domain shift

---

## 1. Introduction

### 1.1 Clinical Motivation

Type 1 Diabetes Mellitus (T1DM) is a chronic autoimmune disorder characterized by the selective destruction of pancreatic beta cells, resulting in absolute insulin deficiency and consequent dysregulation of blood glucose homeostasis [1]. Clinical management of T1DM necessitates continuous monitoring and frequent insulin dose adjustments to maintain euglycaemia (70–180 mg/dL), with failure to do so precipitating life-threatening hyperglycaemic or hypoglycaemic episodes [2]. The widespread adoption of Continuous Glucose Monitoring (CGM) technology has enabled the acquisition of dense temporal BG data at 5-minute sampling intervals, creating a high-resolution physiological signal amenable to data-driven predictive modelling [3].

Predictive BG models capable of anticipating glycaemic excursions 30–60 minutes in advance represent a key enabling component for both alerting systems in open-loop therapy and reference controllers in closed-loop artificial pancreas systems [4]. Such predictions must satisfy stringent clinical accuracy requirements, typically operationalized through the Clarke Error Grid framework [5], wherein predictions must predominantly reside within Zone A to be considered clinically acceptable.

### 1.2 Limitations of Existing Literature

Despite substantial investment in data-driven BG prediction, a critical and persistent methodological limitation pervades the existing literature: the near-universal evaluation paradigm of training and assessing predictive models on a single, homogeneous dataset [6]. This design practice fundamentally conflates within-cohort discriminative performance with generalizable predictive capability. In practical clinical deployment scenarios, a BG prediction model will invariably encounter patient data from heterogeneous sources — different CGM device manufacturers, disparate clinical recording protocols, varied patient demographics, and distinct insulin management regimens — conditions under which within-cohort performance metrics provide no reliable predictive guarantee.

The phenomenon of dataset shift [7], wherein the joint distribution of input features and prediction targets differs between training and deployment environments, constitutes the primary mechanism by which within-cohort models fail upon external validation. In the BG prediction context, dataset shift may manifest through differences in CGM sensor calibration, population-level glycaemic variability patterns, co-recorded physiological signals, and carbohydrate recording conventions. The extent to which individual model families exhibit differential robustness to such shift has not been systematically characterized in the extant literature.

### 1.3 Research Contributions

The present work addresses the foregoing limitations through three methodological contributions:

**Contribution I:** A four-condition bidirectional cross-dataset experimental protocol is formalized, comprising two within-cohort conditions and two complementary cross-dataset transfer conditions, evaluated across two prediction horizons and five model architectures — constituting the first systematic characterization of directional transfer asymmetry in T1DM BG prediction.

**Contribution II:** An automated schema reconciliation pipeline is developed and validated, resolving column-level nomenclature heterogeneity, feature availability discrepancies, and temporal lag representation across two datasets with substantially different recording conventions, without requiring manual feature engineering intervention.

**Contribution III:** A binary dataset-origin classifier achieving 96.3% accuracy in discriminating samples by cohort of origin is trained on the harmonized feature space, providing quantitative evidence of the magnitude of distributional divergence and a practical routing mechanism for directing inference queries to the cohort-optimal predictive model.

### 1.4 Paper Organization

The remainder of this manuscript is structured as follows. Section 2 reviews related work. Section 3 describes the clinical datasets and harmonization procedure. Section 4 details the experimental methodology. Section 5 presents quantitative results. Section 6 provides interpretation and discussion of limitations. Section 7 states conclusions.

---

## 2. Related Work

### 2.1 Data-Driven Blood Glucose Prediction

Autoregressive time series models (ARMA, ARIMA) constituted early computational approaches to BG prediction from CGM signals [8]. The OhioT1DM challenge [9] catalyzed a shift toward standardized within-cohort benchmarking, facilitating direct methodological comparison. Recurrent neural architectures, most notably LSTM networks [10], emerged as a dominant paradigm following demonstrations of superior temporal dependency modelling in glucose time series [11]. Gradient-boosted decision tree ensembles have also demonstrated competitive performance in tabular lag-feature formulations of the prediction problem, benefiting from robustness to feature scaling, native missingness handling, and principled regularization [12].

Ensemble methods combining multiple independently trained base predictors through weighted aggregation have been investigated as a means of reducing prediction variance. Inverse-MAE weighting has been shown to yield consistent improvement over constituent models when base learner predictions are sufficiently diverse [13].

### 2.2 Cross-Dataset Generalization and Domain Adaptation

Systematic cross-dataset evaluation in BG prediction remains substantially underexplored. Oviedo et al. [14] evaluated personalized forecasting models across multiple CGM platforms, observing significant performance degradation attributable to sensor-level distributional differences. Domain adaptation methods employing maximum mean discrepancy minimization [15] and adversarial feature alignment [16] have been applied to reduce cross-cohort prediction error. The present work deliberately avoids domain adaptation techniques, providing a rigorous baseline characterization of unadapted model transfer performance — a necessary precondition for evaluating the marginal benefit of future adaptation strategies.

### 2.3 Feature Harmonization in Clinical Data

Schema heterogeneity constitutes a fundamental obstacle to multi-source clinical data analysis [17]. The OMOP Common Data Model [18] and the HL7 FHIR standard [19] address structured electronic health record harmonization; however, no prior CGM prediction study has described and validated an automated harmonization pipeline as an explicit methodological component.

---

## 3. Datasets and Feature Harmonization

### 3.1 AZT1D Dataset

The AZT1D dataset comprises CGM recordings from T1DM patients acquired at a uniform 5-minute sampling interval. Following temporal alignment and quality filtering, 15,177 samples were retained for evaluation (12,131 training samples at PH30; 12,124 at PH60 after prediction target construction). Raw feature channels include: `CGM` (interstitial glucose, mg/dL), `Basal`, `TotalBolusInsulinDelivered`, `FoodDelivered`, `CarbSize`, `BolusType`, `CorrectionDelivered`, `DeviceMode`, and `EventDateTime`.

### 3.2 HUPA-UCM Dataset

The HUPA-UCM dataset [20] is a publicly available multi-modal T1DM cohort from the Hospital Universitario Príncipe de Asturias (Alcalá de Henares, Spain), comprising 309,392 temporally aligned observations at 5-minute intervals. Integrated modalities include: `glucose` (mg/dL), `basal_rate`, `bolus_volume_delivered`, `carb_input`, `heart_rate` (bpm), `steps` (per interval), and `calories` (kcal). Following stratified subsampling, 50,000 training and 50,000 test samples were utilized.

### 3.3 Automated Feature Harmonization Protocol

The two datasets exhibit substantive schema-level heterogeneity (Table 1), necessitating a systematic reconciliation procedure. The automated harmonization pipeline comprises four sequential operations:

**(i) Nomenclature Standardization:** Dataset-specific column identifiers are mapped to a canonical schema via a predefined correspondence table (e.g., `CGM` → `glucose`; `TotalBolusInsulinDelivered` → `bolus_volume_delivered`).

**(ii) Feature Intersection:** The harmonized feature set is restricted to the intersection of identifiers present in both post-renamed datasets, yielding the retained signal set: {`glucose`, `basal_rate`, `bolus_volume_delivered`, `carb_input`}.

**(iii) Temporal Lag Feature Construction:** For each retained signal $s \in \mathcal{S}$, the lag sequence $\{s_{t-1}, \ldots, s_{t-L}\}$ is computed with $L = 12$ (60-minute retrospective window at 5-minute resolution), yielding feature dimensionality $|\mathcal{S}| \times L = 4 \times 12 = 48$.

**(iv) Missing Value Imputation:** Residual missingness is resolved via forward-fill propagation followed by zero-imputation for leading missing values.

**Table 1: Comparative Characteristics of AZT1D and HUPA-UCM Datasets**

| Characteristic | AZT1D | HUPA-UCM |
|---|---|---|
| Total observations | ~309,000 | 309,392 |
| Sampling interval | 5 minutes | 5 minutes |
| Glucose identifier | `CGM` | `glucose` |
| Bolus insulin identifier | `TotalBolusInsulinDelivered` | `bolus_volume_delivered` |
| Basal insulin identifier | `Basal` | `basal_rate` |
| Carbohydrate identifier | `CarbSize` / `FoodDelivered` | `carb_input` |
| Dataset-exclusive signals | `BolusType`, `CorrectionDelivered`, `DeviceMode` | `heart_rate`, `steps`, `calories` |
| Training samples | 12,131 (PH30) / 12,124 (PH60) | 50,000 |
| Test samples | 3,046 (PH30) / 3,045 (PH60) | 50,000 |

---

## 4. Methodology

### 4.1 Experimental Design

Four experimental conditions are defined, indexed by training cohort → evaluation cohort:

- **A→A (AZT1D→AZT1D):** Within-cohort evaluation on AZT1D.
- **B→B (HUPA→HUPA):** Within-cohort evaluation on HUPA-UCM.
- **A→B (AZT1D→HUPA):** Cross-dataset forward transfer.
- **B→A (HUPA→AZT1D):** Cross-dataset reverse transfer.

Each condition is evaluated at PH30 and PH60, yielding $4 \times 2 = 8$ experiment–horizon configurations and $8 \times 5 = \mathbf{40}$ model–experiment combinations in total.

### 4.2 Prediction Target Formulation

The scalar prediction target at time index $t$ for horizon $h$ is defined as:

$$y_t^{(h)} = g_{t+h}$$

where $g_{t+h}$ denotes interstitial glucose concentration (mg/dL) at time $t+h$, with $h = 6$ for PH30 and $h = 12$ for PH60 at 5-minute resolution. The predictive feature vector is:

$$\mathbf{x}_t = \bigl[s_{t-1}^{(1)}, \ldots, s_{t-L}^{(1)},\; \ldots,\; s_{t-1}^{(|\mathcal{S}|)}, \ldots, s_{t-L}^{(|\mathcal{S}|)}\bigr] \in \mathbb{R}^{|\mathcal{S}| \cdot L}$$

### 4.3 Model Architectures

**Ordinary Least Squares Regression (LR):** A linear baseline estimator minimizing the residual sum of squares, $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$, trained without regularization. Provides a lower bound on representational capacity.

**Random Forest (RF):** A bagged ensemble of $B = 100$ regression trees, each trained on a bootstrap sample with random feature subsampling. Tree depth is unconstrained; minimum samples per terminal node = 2.

**Histogram-based Gradient Boosting (HistGBM):** Implemented as `HistGradientBoostingRegressor` [21]. Configuration: learning rate $\eta = 0.1$, maximum iterations $T = 300$, early stopping on 10% validation hold-out.

**Extreme Gradient Boosting (XGBoost) [22]:** Tree depth = 6, $\eta = 0.1$, $n_{\text{estimators}} = 200$, subsample = 0.8, column subsampling = 0.8, $L_2$ regularization $\lambda = 1$.

**Long Short-Term Memory (LSTM) [10]:** Two recurrent layers with hidden dimensionality 64, inter-layer dropout probability 0.2. Input reshaped to $(N, L, |\mathcal{S}|) = (N, 12, 4)$. Optimized via Adam [23] ($\alpha = 0.001$, 30 epochs, batch size 256, MSE loss).

### 4.4 Evaluation Metrics

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i| \;[\text{mg/dL}], \quad \text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2} \;[\text{mg/dL}]$$

$$R^2 = 1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}, \quad \text{MARD} = \frac{1}{N}\sum_{i=1}^{N}\frac{|y_i-\hat{y}_i|}{y_i}\times 100\%$$

Clarke Error Grid Zone A membership percentage is reported as a clinically motivated criterion, requiring $|\hat{y}_i - y_i| \leq 20\,\text{mg/dL}$ or $|\hat{y}_i - y_i| \leq 0.20\,y_i$ [5].

### 4.5 MAE-Weighted Ensemble

$$\hat{y}_t^{\text{ens}} = \sum_{m=1}^{M} w_m \hat{y}_t^{(m)}, \qquad w_m = \frac{\text{MAE}_m^{-1}}{\sum_{k=1}^{M}\text{MAE}_k^{-1}}$$

Weights $\text{MAE}_m$ are estimated via 5-fold cross-validation on the training partition to preclude test set contamination.

### 4.6 Dataset-Origin Classifier

A Random Forest binary classifier is trained on the 48-dimensional harmonized feature space to discriminate AZT1D from HUPA-UCM samples. Training: 80,000 samples (40,000 per class); evaluation: 20,000 samples (10,000 per class). This classifier (i) quantifies distributional divergence through discriminability and (ii) provides inference-time routing to the cohort-optimal model.

### 4.7 Statistical Significance Testing

The Wilcoxon signed-rank test [24], a non-parametric paired comparison appropriate for non-Gaussian error distributions, is applied to evaluate the significance of MAE differences between within-cohort and cross-dataset conditions over per-subject error vectors. Significance threshold: $\alpha = 0.05$.

---

## 5. Results

### 5.1 Within-Cohort Performance — PH30

**Table 2: Within-Cohort Prediction Performance at PH30 (MAE in mg/dL)**

| Model | AZT1D MAE | AZT1D R² | AZT1D Clarke-A (%) | HUPA MAE | HUPA R² | HUPA Clarke-A (%) |
|---|---|---|---|---|---|---|
| Linear Regression | 19.19 | 0.755 | 81.7 | 13.05 | 0.892 | 85.4 |
| Random Forest | 17.21 | 0.793 | 84.8 | 12.12 | 0.902 | 86.2 |
| HistGBM | **16.94** | **0.796** | **85.5** | **12.05** | 0.904 | **86.5** |
| XGBoost | 16.99 | 0.794 | 84.7 | 11.97 | **0.905** | 86.5 |
| LSTM | 18.79 | 0.760 | 83.2 | 13.36 | 0.890 | 85.0 |

Gradient-boosted ensembles (HistGBM, XGBoost) attain the lowest MAE and highest $R^2$ on both datasets. The HUPA within-cohort task yields uniformly superior performance relative to AZT1D across all model classes, attributable in part to the four-fold larger HUPA training partition (50,000 vs. 12,131 samples) and the comparatively lower glycaemic variability of this cohort.

### 5.2 Cross-Dataset Transfer Performance — PH30

**Table 3: Cross-Dataset Transfer Performance at PH30 (MAE in mg/dL)**

| Model | AZT1D→HUPA MAE | AZT1D→HUPA R² | HUPA→AZT1D MAE | HUPA→AZT1D R² |
|---|---|---|---|---|
| Linear Regression | **16.21** | **0.855** | 23.26 | 0.401 |
| Random Forest | 23.54 | 0.740 | **18.88** | **0.751** |
| HistGBM | 22.38 | 0.769 | 18.95 | 0.747 |
| XGBoost | 22.36 | 0.765 | 19.12 | 0.745 |
| LSTM | 17.25 | 0.840 | 22.68 | 0.419 |

A pronounced inversion of the model performance hierarchy is observed in the A→B direction: linear regression (MAE = 16.21 mg/dL) and LSTM (MAE = 17.25 mg/dL) substantially outperform gradient-boosted methods (MAE range 22.36–23.54 mg/dL), which exhibit an approximately 32–39% MAE increase relative to within-cohort baselines. In the B→A direction, Random Forest achieves the lowest transfer MAE (18.88 mg/dL; $R^2 = 0.751$).

### 5.3 Within-Cohort Performance — PH60

**Table 4: Within-Cohort Prediction Performance at PH60 (MAE in mg/dL)**

| Model | AZT1D MAE | AZT1D R² | HUPA MAE | HUPA R² |
|---|---|---|---|---|
| Linear Regression | 30.67 | 0.433 | 22.40 | 0.705 |
| Random Forest | 28.04 | 0.504 | 21.20 | 0.726 |
| HistGBM | 28.15 | 0.505 | 20.90 | 0.735 |
| XGBoost | **27.93** | **0.511** | **20.81** | **0.736** |
| LSTM | 30.14 | 0.452 | 22.34 | 0.707 |

Predictive accuracy degrades substantially relative to PH30 across all models and datasets. The horizon-induced MAE increase for HistGBM is $\Delta$MAE = 11.21 mg/dL on AZT1D (16.94 → 28.15) versus 8.85 mg/dL on HUPA (12.05 → 20.90), indicating comparatively greater temporal predictability in the HUPA cohort.

### 5.4 Cross-Dataset Transfer Performance — PH60

**Table 5: Cross-Dataset Transfer Performance at PH60 (MAE in mg/dL)**

| Model | AZT1D→HUPA MAE | AZT1D→HUPA R² | HUPA→AZT1D MAE | HUPA→AZT1D R² |
|---|---|---|---|---|
| Linear Regression | 27.38 | 0.626 | 36.70 | −0.002 |
| Random Forest | 37.12 | 0.358 | **30.61** | **0.409** |
| HistGBM | 34.06 | 0.445 | 32.29 | 0.334 |
| XGBoost | 35.12 | 0.417 | 32.13 | 0.333 |
| LSTM | **30.96** | **0.543** | 40.10 | −0.143 |

HUPA→AZT1D transfer at PH60 exhibits severe degradation: linear regression attains $R^2 = -0.002$ and LSTM $R^2 = -0.143$, values statistically indistinguishable from or inferior to a mean-prediction baseline. Random Forest retains the most stable transfer performance in this condition ($R^2 = 0.409$).

### 5.5 Dataset-Origin Classifier

**Table 6: Dataset-Origin Binary Classification Performance**

| Metric | Value |
|---|---|
| Accuracy | 0.9629 |
| F1-Score (macro-averaged) | 0.9516 |
| ROC-AUC | 0.9202 |
| Training samples | 80,000 |
| Test samples | 20,000 |

The attainment of 96.3% discriminative accuracy on the harmonized feature space provides strong empirical evidence that AZT1D and HUPA-UCM occupy substantially non-overlapping regions of the input distribution following harmonization, quantifying the magnitude of covariate shift underpinning the observed transfer penalty.

### 5.6 Clarke Error Grid Analysis

Clarke Error Grid Zone A membership percentages for within-cohort PH30 configurations range from 81.7% (linear regression, AZT1D) to 86.5% (HistGBM and XGBoost, HUPA), indicating that the preponderance of predictions reside within clinically acceptable bounds. Zone A rates diminish progressively with increasing prediction horizon and increasing distributional distance between training and evaluation cohorts.

### 5.7 Statistical Significance

Wilcoxon signed-rank tests reveal statistically significant MAE degradation under cross-dataset transfer relative to within-cohort baselines for all gradient-boosted tree architectures ($p < 0.05$). LSTM exhibits borderline significance in the A→B direction ($p \approx 0.07$), consistent with its attenuated transfer penalty in that condition. Linear regression demonstrates non-significant degradation in the A→B direction ($p > 0.05$), corroborating its distributional robustness.

---

## 6. Discussion

### 6.1 Directional Asymmetry in Cross-Dataset Transfer

A non-trivial and clinically consequential finding of the present study is the pronounced directionality of generalization performance: models trained on AZT1D exhibit systematically superior transfer to HUPA-UCM relative to the reverse configuration, despite the four-fold smaller AZT1D training partition. Two mechanistic explanations are advanced. First, HUPA-UCM provides physiological signals (heart rate, step count, caloric expenditure) that are excluded by harmonization; models trained on HUPA may therefore develop representations dependent on inter-feature correlations involving these excluded channels, yielding less transferable encodings. Second, greater intra-cohort glycaemic variability within the AZT1D corpus may induce representations with broader distributional coverage, yielding improved transferability to the comparatively smoother HUPA glycaemic trajectories.

### 6.2 Transfer Robustness of Low-Complexity Models

The substantially superior cross-dataset performance of ordinary least squares regression in the A→B direction — MAE = 16.21 mg/dL compared to 22.36–23.54 mg/dL for gradient-boosted methods — constitutes a theoretically significant finding. By virtue of their restricted hypothesis class, linear models are structurally precluded from encoding complex dataset-specific feature interactions that are non-predictive in the target domain, yielding implicit distributional robustness through capacity restriction. This observation is consistent with findings in cross-institutional electronic health record prediction [25], and challenges the prevailing assumption that superior within-cohort performers should be preferentially selected for clinical deployment.

### 6.3 Direction-Dependent LSTM Generalization

LSTM architectures exhibit a direction-dependent generalization profile: competitive transfer performance in A→B (MAE = 17.25 and 30.96 mg/dL at PH30 and PH60, respectively), but severe failure in B→A at PH60 ($R^2 = -0.143$). This asymmetry may be attributed to the recurrent state representations encoded from each training corpus. HUPA-derived hidden states, calibrated to smooth low-variability glucose trajectories, are systematically mismatched to the irregular temporal dynamics characteristic of the AZT1D cohort, particularly at the extended 60-minute horizon where trajectory divergence is most pronounced.

### 6.4 Distributional Divergence as a Predictor of Transfer Error

Under the theoretical framework of $\mathcal{H}$-divergence in domain adaptation [26], the high binary discriminability of the dataset-origin classifier (96.3%) implies a large Ben-David divergence bound, predicting large cross-domain transfer error — a prediction that is empirically confirmed across conditions in Sections 5.2 and 5.4. This classifier additionally provides a practical deployment artefact: an inference-time routing system may direct incoming patient data to the cohort-optimal predictive model, thereby mitigating the transfer penalty in multi-source clinical systems.

### 6.5 Clinical Safety Implications

The observation that all within-cohort PH30 predictions achieve Clarke Error Grid Zone A membership exceeding 81.7% is of direct clinical relevance, approaching or meeting the ≥80% threshold generally recommended as a minimum criterion for clinical decision support acceptability [27]. Cross-dataset configurations at PH60, however, exhibit Zone A rates commensurate with their elevated MAE, underscoring the necessity of cohort-specific prospective validation prior to clinical deployment.

### 6.6 Limitations

The following limitations are acknowledged:

1. Evaluation proceeds on a fixed temporal train–test split rather than per-subject leave-one-out cross-validation, which would provide estimates of inter-subject variability and be more reflective of prospective evaluation.

2. The HUPA-UCM dataset is subsampled to 50,000 observations for computational tractability; the full 309,392-observation corpus may contain distributional content not represented in the subsample.

3. No domain adaptation technique is applied. Methods such as instance reweighting, adversarial feature alignment, or distribution matching are expected to reduce the cross-dataset MAE penalty and constitute a natural extension of the present framework.

4. The LSTM architecture is deliberately simplified (two layers, 64 units) to ensure architectural comparability. Transformer-based temporal architectures [28] may capture longer-range dependencies more effectively.

5. Meal announcement and insulin bolus timing are not modelled as discrete events. Incorporation of event-triggered input representations is known to substantially improve post-prandial prediction accuracy [29] and represents an important avenue for improving cross-dataset robustness.

6. Inter-subject personalization is not investigated; individual patient glycaemic dynamics may modulate cross-cohort transfer performance in ways not captured by population-level evaluation.

---

## 7. Conclusion

The present study introduced a systematic bidirectional cross-dataset evaluation framework for short-term blood glucose prediction in T1DM, encompassing two heterogeneous clinical datasets, five machine learning architectures, two prediction horizons, and 40 model–experiment combinations, accompanied by an automated feature harmonization pipeline and a distribution-aware dataset-origin classifier.

The principal findings are as follows: (i) within-cohort predictive performance is clinically competitive, with gradient-boosted ensembles achieving MAE values of 12.05–16.94 mg/dL at PH30 and Clarke Zone A membership exceeding 81%; (ii) cross-dataset transfer is characterized by statistically significant and directionally asymmetric performance degradation ($p < 0.05$), with AZT1D→HUPA transfer yielding substantially better outcomes than the reverse; (iii) contrary to prevailing practice, low-complexity linear models demonstrate superior cross-dataset robustness in specific transfer conditions, outperforming gradient-boosted methods by up to 7.33 mg/dL MAE; and (iv) the dataset-origin classifier produces 96.3% discriminative accuracy, providing both quantitative characterization of inter-dataset distributional divergence and an operational routing mechanism.

These findings collectively establish a benchmark and evaluation protocol that the authors advocate be adopted as a standard methodological component in future BG prediction research targeting multi-cohort or prospective clinical deployment. Future investigations will pursue domain adaptation methods, transformer-based temporal architectures, per-subject personalization, and event-driven input representations to further reduce the cross-dataset generalization penalty.

---

## Acknowledgements

The authors acknowledge the creators of the HUPA-UCM dataset for its public availability, and the open-source communities maintaining scikit-learn, XGBoost, PyTorch, and Streamlit.

---

## Data and Code Availability

All source code, trained model artefacts, experimental configurations, and evaluation outputs are available at: **https://github.com/dikshasharsharma-sys/CGM-Forecast**

---

## References

[1] American Diabetes Association. (2022). Standards of medical care in diabetes — 2022. *Diabetes Care*, 45(Suppl. 1), S1–S264.

[2] Kovatchev, B. P. (2019). Metrics for glycaemic control — from HbA1c to continuous glucose monitoring. *Nature Reviews Endocrinology*, 13(7), 425–436.

[3] Cappon, G., Vettoretti, M., Sparacino, G., & Facchinetti, A. (2019). Continuous glucose monitoring sensors for diabetes management. *Diabetes & Metabolism Journal*, 43(4), 383–397.

[4] Doyle, F. J., Huyett, L. M., Lee, J. B., Zisser, H. C., & Dassau, E. (2014). Closed-loop artificial pancreas systems: Engineering the algorithms. *Diabetes Care*, 37(5), 1191–1197.

[5] Clarke, W. L., Cox, D., Gonder-Frederick, L. A., Carter, W., & Pohl, S. L. (1987). Evaluating clinical accuracy of systems for self-monitoring of blood glucose. *Diabetes Care*, 10(5), 622–628.

[6] Woldaregay, A. Z., et al. (2019). Data-driven modeling and prediction of blood glucose dynamics: Machine learning applications in type 1 diabetes. *Artificial Intelligence in Medicine*, 98, 109–134.

[7] Quionero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (2009). *Dataset Shift in Machine Learning*. MIT Press.

[8] Sparacino, G., et al. (2007). Glucose concentration can be predicted ahead in time from continuous glucose monitoring sensor time-series. *IEEE Transactions on Biomedical Engineering*, 54(5), 931–937.

[9] Marling, C., & Bunescu, R. (2020). The OhioT1DM dataset for blood glucose level prediction: Update 2020. *CEUR Workshop Proceedings*, Vol. 2675.

[10] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

[11] Martinsson, J., Schliep, A., Eliasson, B., & Mogren, O. (2020). Blood glucose prediction with variance estimation using recurrent neural networks. *Journal of Healthcare Informatics Research*, 4(1), 1–18.

[12] Khadem, H., Nemat, H., Elliott, J., & Benaissa, M. (2021). Blood glucose level forecasting on type-1-diabetes subjects during physical activity. *Sensors*, 21(17), 5839.

[13] Li, K., Daniels, J., Liu, C., Herrero-Viñas, P., & Georgiou, P. (2019). Convolutional recurrent neural networks for glucose prediction. *IEEE Journal of Biomedical and Health Informatics*, 24(2), 603–613.

[14] Oviedo, S., Vehí, J., Calm, R., & Armengol, J. (2017). A review of personalized blood glucose prediction strategies for T1DM patients. *International Journal for Numerical Methods in Biomedical Engineering*, 33(6), e2833.

[15] Gretton, A., et al. (2012). A kernel two-sample test. *Journal of Machine Learning Research*, 13(1), 723–773.

[16] Ganin, Y., et al. (2016). Domain-adversarial training of neural networks. *Journal of Machine Learning Research*, 17(1), 2096–2030.

[17] Beaulieu-Jones, B. K., et al. (2018). Characterizing and managing missing structured data in electronic health records. *JMIR Medical Informatics*, 6(1), e11.

[18] Garza, M., et al. (2016). Evaluating common data models for use with a longitudinal community registry. *Journal of Biomedical Informatics*, 64, 333–341.

[19] Bender, D., & Sartipi, K. (2013). HL7 FHIR: An agile and RESTful approach to healthcare information exchange. *Proceedings of IEEE CBMS*, pp. 326–331.

[20] Rodríguez-Rodríguez, I., et al. (2023). HUPA dataset: A multimodal wearable dataset to monitor type 1 diabetes patients. *Data in Brief*, 46, 108814.

[21] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

[22] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of ACM SIGKDD*, pp. 785–794.

[23] Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *Proceedings of ICLR 2015*.

[24] Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80–83.

[25] Futoma, J., et al. (2017). An improved multi-output Gaussian process RNN with real-time validation for early sepsis detection. *Proceedings of ML for Healthcare Conference*, pp. 243–254.

[26] Ben-David, S., et al. (2010). A theory of learning from different domains. *Machine Learning*, 79(1), 151–175.

[27] International Diabetes Federation. (2021). *IDF Diabetes Atlas*, 10th Edition. Brussels: IDF.

[28] Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

[29] Herrero, P., et al. (2012). A bio-inspired glucose controller based on pancreatic beta-cell physiology. *Journal of Diabetes Science and Technology*, 6(3), 606–616.
