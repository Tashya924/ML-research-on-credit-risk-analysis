# Machine Learning Research on Credit Risk Analysis: An Empirical Replication and Methodological Audit

[![Replication Study](https://img.shields.io/badge/Replication-Xu_et_al._(2024)-blue.svg)](https://link.springer.com/article/10.1007/s10479-024-06134-x)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical research replication and methodological audit of the explainable credit risk modeling paper:
> **Xu, Q. A., Benson, V., & Chang, V. (2024).** *Prediction of bank credit worthiness through credit risk analysis: an explainable machine learning study.* **Annals of Operations Research**, 354(1), 247–271. [DOI: 10.1007/s10479-024-06134-x](https://link.springer.com/article/10.1007/s10479-024-06134-x).

This project benchmarks 16 machine-learning and deep-learning classifiers on the UCI "Default of Credit Card Clients" dataset, **empirically proves and quantifies how class oversampling before partitioning induces catastrophic data leakage** (inflating reported $F_1$ scores by **+0.30 on average**), establishes the true performance baseline using `SMOTENC` after partitioning, calibrates discrimination thresholds via 10-fold cross-validation, augments training using a zero-leakage **Conditional Tabular GAN (CTGAN)**, and unpacks model decisions through **SHAP, LIME, and Permutation Feature Importance**.

---

## Key Research Findings

> **Headline finding:** The "oversample-then-split" pipeline used in the published study inflates reported $F_1$ scores by **+0.305 on average** across all models, and by as much as **+0.508** for tree-based memorizers (e.g., Decision Tree jumping from an honest $F_1 = 0.380$ to an inflated $0.885$). Once data leakage is eliminated, the true predictive performance ceiling on this dataset is **$F_1 \approx 0.51$ / $\text{ROC-AUC} \approx 0.76$**.

```
                           DATA LEAKAGE INFLATION GAP
                   (Published Leaky vs. Honest Corrected F1)
                   
Decision Tree        [======= Honest 0.389 =======][==== +0.495 INFLATION ====>] 0.885
Extra Tree           [======= Honest 0.380 =======][==== +0.508 INFLATION ====>] 0.889
Extra Trees          [============== Honest 0.481 =============][= +0.462 ====>] 0.943
Random Forest        [============== Honest 0.486 =============][= +0.447 ====>] 0.932
XGBoost              [============== Honest 0.480 =============][= +0.332 ====>] 0.812
KNN                  [============ Honest 0.442 ===========][=== +0.324 ======>] 0.766
Gaussian NB          [========== Honest 0.389 ==========][==== +0.290 ========>] 0.679
Hist Gradient Boost  [============== Honest 0.488 =============][= +0.257 ====>] 0.746
MLP Classifier       [============== Honest 0.495 =============][= +0.245 ====>] 0.740
Gradient Boosting    [=============== Honest 0.511 ============][= +0.191 ====>] 0.701
Logistic Regression  [============= Honest 0.465 ============][== +0.192 =====>] 0.658
LDA / Ridge          [============= Honest 0.468 ============][== +0.186 =====>] 0.654
AdaBoost             [=============== Honest 0.501 ============][= +0.160 ====>] 0.661
```

---

## Research Architecture

![Credit Risk Research Architecture](architecture_pipeline.png)

---

## Table of Contents

- [Why This Study Exists](#why-this-study-exists)
- [Repository Layout](#repository-layout)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. The Two Parallel Pipelines (Leaky vs. Corrected)](#1-the-two-parallel-pipelines-leaky-vs-corrected)
  - [2. The 16 Classification Algorithms](#2-the-16-classification-algorithms)
  - [3. Discrimination Threshold Calibration](#3-discrimination-threshold-calibration)
  - [4. Zero-Leakage CTGAN Augmentation & Deep Learning](#4-zero-leakage-ctgan-augmentation--deep-learning)
  - [5. Model Explainability Suite (XAI)](#5-model-explainability-suite-xai)
- [Audit Results & Paper Replication](#audit-results--paper-replication)
  - [Replication Comparison with Xu et al. (2024) Table 2](#replication-comparison-with-xu-et-al-2024-table-2)
  - [The Data Leakage Gap](#the-data-leakage-gap)
  - [CTGAN Augmented Deep Learning Performance](#ctgan-augmented-deep-learning-performance)
- [Getting Started](#getting-started)
- [Citation & References](#citation--references)

---

## Why This Study Exists

Credit risk default prediction datasets are inherently imbalanced (~22% default rate in the UCI dataset). In academic literature, researchers frequently attempt to counter this class imbalance using oversampling techniques such as Random Oversampling (ROS) or SMOTE. 

However, **applying oversampling before partitioning the dataset into training and test splits contaminates the test set with duplicated or synthesized copies of minority instances that also appear in the training partition**. Furthermore, fitting feature standardizers (`StandardScaler`) on the entire dataset prior to splitting leaks the global mean and variance into test evaluations.

The methodology in **Xu, Benson, & Chang (2024)** suffered from both issues:
> *"Since not all the data sets were scaled evenly, the whole dataset was rescaled using a Z-score standardization... a random over-sampling strategy was used for the target variable. The data were then randomly divided into a training and a test set using a 75/25% split."* (Section 4, p. 257)

This project runs **every model side-by-side across both pipelines** on the identical underlying dataset to isolate and measure the exact magnitude of performance inflation caused by this leakage.

---

## Repository Layout

```
.
├── credit_risk_research_pipeline.ipynb  # Consolidated end-to-end research notebook
├── requirements.txt                    # Python virtual environment dependencies
├── architecture_pipeline.png           # Research pipeline architecture diagram
├── UCI_Credit_Card.csv                 # UCI credit card default dataset (30,000 rows)
├── results/                            # All generated research outputs & artifacts
│   ├── architecture_pipeline.png       # High-resolution pipeline schematic
│   ├── ctgan_synthetic_120000.parquet  # Cached CTGAN synthetic dataset (120k samples)
│   ├── charts/                         # Benchmark comparison plots
│   │   ├── chart_1_roc_auc.png         # ROC-AUC across scenarios & thresholds
│   │   ├── chart_2_f1_score.png        # F1 score comparison (the leakage gap)
│   │   ├── chart_3_precision.png       # Precision comparison
│   │   ├── chart_4_recall.png          # Recall comparison
│   │   ├── chart_gan_f1_score.png      # CTGAN augmented F1 benchmarks
│   │   ├── shap_summary_gb.png         # SHAP global beeswarm plot
│   │   ├── shap_dep_PAY_0.png          # SHAP dependence plot for PAY_0
│   │   ├── permutation_importance.png  # Model-agnostic permutation importance
│   │   └── shap_waterfall_sample_0.png # Local SHAP waterfall attribution
│   ├── metrics/                        # Quantitative metric summaries
│   │   ├── results_summary.csv         # Full 64-row benchmark results table
│   │   ├── results_summary.txt         # Text summary of model benchmark
│   │   ├── paper_replication_comparison.csv # Side-by-side audit vs Paper Table 2
│   │   └── gan_results_summary.csv     # CTGAN extended dataset performance
│   └── threshold_plots/                # Stratified 10-fold CV threshold curves
│       ├── corrected_Gradient_Boosting_threshold.png
│       ├── corrected_Random_Forest_threshold.png
│       ├── corrected_AdaBoost_threshold.png
│       └── corrected_Logistic_Regression_threshold.png
├── newa.py                             # Standalone legacy benchmark script
├── ctgan_synthetic_credit_default_modeling.ipynb # Legacy CTGAN prototyping notebook
└── ml_credit_report_intern.ipynb       # Legacy XAI prototyping notebook
```

---

## Dataset

- **Source:** UCI Machine Learning Repository — *Default of Credit Card Clients Dataset*
- **Size:** 30,000 observations (Taiwanese credit card holders)
- **Target:** `default.payment.next.month` (Binary: 0 = non-default, 1 = default)
- **Natural Class Balance:** 77.88% non-default (23,364 cases), 22.12% default (6,636 cases)
- **Predictor Groups (23 features):**
  - **Demographics & Credit Limit:** `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
  - **Repayment History:** `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6` (repayment status from April to September; delayed months)
  - **Monthly Bill Statements:** `BILL_AMT1` through `BILL_AMT6`
  - **Previous Monthly Payments:** `PAY_AMT1` through `PAY_AMT6`

---

## Methodology

### 1. The Two Parallel Pipelines (Leaky vs. Corrected)

| Step | Leaky Pipeline (Xu et al., 2024 Flaw) | Corrected Pipeline (Methodologically Sound) |
|---|---|---|
| **Data Partitioning** | Resampled first, then split 75/25 | Split raw data 75/25 **first** |
| **Resampling** | `RandomOverSampler` applied globally to full dataset | `SMOTENC` applied **strictly to the training partition** |
| **Feature Scaling** | `StandardScaler` fitted globally on full dataset | `ColumnTransformer` fitted **strictly on training partition**; test set transformed |
| **Categorical Integrity** | Categorical features standardized numerically | Categoricals passed through unscaled to preserve discrete structure |
| **Evaluation Integrity** | Contaminated test set containing duplicate train rows | Pristine test set with natural 77.9% / 22.1% distribution |

### 2. The 16 Classification Algorithms

The benchmark evaluates 16 diverse machine-learning and deep-learning architectures:
1. **Linear & Discriminant Models:** Logistic Regression, Ridge Classifier (calibrated), Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA)
2. **Tree & Ensemble Models:** Decision Tree, Extra Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, Histogram Gradient Boosting, XGBoost, LightGBM
3. **Instance & Probabilistic Models:** K-Nearest Neighbors (KNN), Gaussian Naive Bayes
4. **Neural & Deep Learning Models:** Multi-Layer Perceptron (`MLPClassifier`), PyTorch `TabularTransformer`

### 3. Discrimination Threshold Calibration

A standard classification cutoff of $t = 0.50$ is suboptimal on class-imbalanced datasets. Every model is evaluated under two regimes:
- **Default Threshold:** $p \ge 0.50$
- **Optimal Threshold:** Decision threshold $t^* \in [0.00, 0.99]$ dynamically searched to maximize $F_1$.

Furthermore, **Stratified 10-Fold Cross-Validation** is conducted on the training data to generate threshold sensitivity curves plotting Precision, Recall, $F_1$, and Queue Rate with $\pm 1\sigma$ confidence bands.

### 4. Zero-Leakage CTGAN Augmentation & Deep Learning

To augment the training distribution without contaminating test metrics:
1. A **Conditional Tabular GAN (CTGAN)** is trained **exclusively on the training split** (`X_train_raw`, 22,500 rows).
2. The generator samples **120,000 synthetic observations**.
3. Real training observations and synthetic records are concatenated (142,500 total training samples).
4. Models, including a custom **PyTorch Tabular Transformer** (multi-head self-attention tabular encoder) and deep neural networks, are trained on the augmented distribution and evaluated on the untouched real test set.

### 5. Model Explainability Suite (XAI)

- **Global Attribution:** SHAP `TreeExplainer` beeswarm summary plots and mean $|SHAP|$ rankings identifying dominant features.
- **Feature Interactions:** SHAP dependence plots examining non-linear feature relationships and threshold boundaries for `PAY_0`, `LIMIT_BAL`, etc.
- **Model-Agnostic Validation:** Permutation Feature Importance measuring test $F_1$ degradation across 10 random permutations.
- **Local Explanations:** Individual applicant waterfall attributions (SHAP) and rule-based explanations (LIME) for risk profiling.

---

## Audit Results & Paper Replication

### Replication Comparison with Xu et al. (2024) Table 2

The table below contrasts the published metrics from Table 2 of Xu et al. (2024) with our replicated Leaky pipeline and the honest Corrected pipeline (all evaluated at default threshold $t = 0.50$ on held-out test data):

| Algorithm | Published Paper $F_1$ | Replicated Leaky $F_1$ | Honest Corrected $F_1$ | $F_1$ Inflation Gap | Published Recall | Replicated Leaky Recall | Honest Corrected Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Decision Tree** | 0.80 | **0.8848** | 0.3894 | **+0.495** | 0.81 | 0.9599 | 0.4745 |
| **Random Forest** | 0.80 | **0.9323** | 0.4855 | **+0.447** | 0.82 | 0.9673 | 0.4800 |
| **KNN** | 0.71 | **0.7660** | 0.4424 | **+0.324** | 0.75 | 0.8282 | 0.6060 |
| **Gaussian Naive Bayes** | 0.39 | **0.6791** | 0.3890 | **+0.290** | 0.39 | 0.8054 | 0.9060 |
| **LightGBM** | 0.78 | **0.7498** | 0.4883 | **+0.262** | 0.79 | 0.7124 | 0.5175 |
| **MLP Classifier** | 0.73 | **0.7403** | 0.4954 | **+0.245** | 0.74 | 0.7561 | 0.5802 |
| **Gradient Boosting** | 0.80 | **0.7012** | 0.5105 | **+0.191** | 0.82 | 0.6461 | 0.5593 |
| **Logistic Regression** | 0.68 | **0.6576** | 0.4653 | **+0.192** | 0.78 | 0.6350 | 0.6663 |
| **LDA** | 0.78 | **0.6540** | 0.4684 | **+0.186** | 0.81 | 0.6220 | 0.6509 |
| **AdaBoost** | 0.79 | **0.6607** | 0.5005 | **+0.160** | 0.82 | 0.5775 | 0.5839 |

> **Audit Conclusion:** The replicated Leaky pipeline reproduces the published paper's inflated metrics. Tree-based memorizers (Decision Tree, Random Forest) exhibit near-perfect test scores (~0.93 $F_1$) because identical training rows were duplicated into the test partition. In the honest pipeline, no model exceeds $F_1 = 0.51$.

### The Data Leakage Gap

When ranked by the magnitude of $F_1$ score inflation:
1. **Decision Tree & Extra Tree:** $+0.495$ to $+0.508$ inflation. High-variance single trees overfit to duplicated samples.
2. **Random Forest & Extra Trees:** $+0.447$ to $+0.462$ inflation. Ensembles of memorizers achieve artificial near-perfection.
3. **XGBoost & LightGBM:** $+0.262$ to $+0.332$ inflation.
4. **Linear Models (LogReg, Ridge, LDA):** $+0.160$ to $+0.192$ inflation. Linear boundaries cannot isolate individual points, resulting in less severe but significant leakage inflation.

### CTGAN Augmented Deep Learning Performance

Evaluated on the unpolluted real test set (7,500 held-out observations with natural 22.1% default rate):

| Algorithm | Scenario | Accuracy | Recall | Precision | $F_1$ Score | ROC-AUC | Optimal Threshold |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hist Gradient Boosting** | GAN Scaled (Opt. Thresh) | 0.7863 | 0.5575 | 0.5067 | **0.5309** | 0.7621 | 0.25 |
| **XGBoost** | GAN Scaled (Opt. Thresh) | 0.7848 | 0.5612 | 0.5036 | **0.5308** | 0.7585 | 0.27 |
| **Tabular Transformer (DL)** | GAN Scaled (Opt. Thresh) | 0.7836 | 0.5372 | 0.5011 | **0.5185** | 0.7562 | 0.23 |
| **MLP Classifier** | GAN Scaled (Opt. Thresh) | 0.7755 | 0.5513 | 0.4846 | **0.5158** | 0.7591 | 0.25 |
| **Random Forest** | GAN Scaled (Opt. Thresh) | 0.7687 | 0.5556 | 0.4718 | **0.5103** | 0.7494 | 0.28 |
| **Gradient Boosting** | GAN Scaled (Opt. Thresh) | 0.7871 | 0.4985 | 0.5094 | **0.5039** | 0.7492 | 0.25 |
| **Tabular Transformer (DL)** | GAN Scaled (t=0.50) | 0.8160 | 0.3276 | 0.6508 | 0.4358 | 0.7562 | 0.50 |
| **Random Forest** | GAN Scaled (t=0.50) | 0.8088 | 0.3073 | 0.6196 | 0.4108 | 0.7494 | 0.50 |
| **Hist Gradient Boosting** | GAN Scaled (t=0.50) | 0.8119 | 0.2637 | 0.6682 | 0.3781 | 0.7621 | 0.50 |
| **XGBoost** | GAN Scaled (t=0.50) | 0.8075 | 0.2631 | 0.6360 | 0.3722 | 0.7585 | 0.50 |
| **MLP Classifier** | GAN Scaled (t=0.50) | 0.8069 | 0.2545 | 0.6379 | 0.3638 | 0.7591 | 0.50 |
| **Gradient Boosting** | GAN Scaled (t=0.50) | 0.8045 | 0.1801 | 0.6894 | 0.2856 | 0.7492 | 0.50 |

---

## Getting Started

### Prerequisites

- Python 3.10, 3.11, or 3.12
- macOS, Linux, or Windows WSL2

### 1. Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Tashya924/ML-research-on-credit-risk-analysis.git
cd ML-research-on-credit-risk-analysis

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Execute Research Notebook

Open and run the consolidated notebook in Jupyter or VS Code:

```bash
jupyter notebook credit_risk_research_pipeline.ipynb
```

The notebook executes top-to-bottom, reproduces all tables, trains the models, generates the threshold curves, runs CTGAN augmentation and deep learning models, and computes the complete explainability suite. All plots and metric summaries are automatically exported to the `results/` directory.

---

## Citation & References

If you build upon this replication and audit study, please cite both the original article and this repository:

```bibtex
@article{xu2024prediction,
  title={Prediction of bank credit worthiness through credit risk analysis: an explainable machine learning study},
  author={Xu, Qianwen Ariel and Benson, Vladlena and Chang, Victor},
  journal={Annals of Operations Research},
  volume={354},
  number={1},
  pages={247--271},
  year={2024},
  publisher={Springer},
  doi={10.1007/s10479-024-06134-x}
}
```
