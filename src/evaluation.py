"""
Model evaluation, discrimination threshold calibration, and paper replication audit.
"""

import os
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns

# Ground truth metrics from Table 2 of Xu et al. (2024)
PAPER_TABLE_2 = {
    "Gradient Boosting": {"Paper Precision": 0.81, "Paper Recall": 0.82, "Paper F1": 0.80, "Paper Accuracy": 0.82, "Paper AUC": 0.66},
    "Random Forest": {"Paper Precision": 0.81, "Paper Recall": 0.82, "Paper F1": 0.80, "Paper Accuracy": 0.82, "Paper AUC": 0.66},
    "Decision Tree": {"Paper Precision": 0.80, "Paper Recall": 0.81, "Paper F1": 0.80, "Paper Accuracy": 0.81, "Paper AUC": 0.66},
    "AdaBoost": {"Paper Precision": 0.81, "Paper Recall": 0.82, "Paper F1": 0.79, "Paper Accuracy": 0.82, "Paper AUC": 0.64},
    "LightGBM": {"Paper Precision": 0.77, "Paper Recall": 0.79, "Paper F1": 0.78, "Paper Accuracy": 0.79, "Paper AUC": 0.63},
    "LDA": {"Paper Precision": 0.80, "Paper Recall": 0.81, "Paper F1": 0.78, "Paper Accuracy": 0.81, "Paper AUC": 0.61},
    "Logistic Regression": {"Paper Precision": 0.61, "Paper Recall": 0.78, "Paper F1": 0.68, "Paper Accuracy": 0.78, "Paper AUC": 0.50},
    "KNN": {"Paper Precision": 0.70, "Paper Recall": 0.75, "Paper F1": 0.71, "Paper Accuracy": 0.75, "Paper AUC": 0.54},
    "MLP Classifier": {"Paper Precision": 0.72, "Paper Recall": 0.74, "Paper F1": 0.73, "Paper Accuracy": 0.74, "Paper AUC": 0.59},
    "Gaussian Naive Bayes": {"Paper Precision": 0.74, "Paper Recall": 0.39, "Paper F1": 0.39, "Paper Accuracy": 0.39, "Paper AUC": 0.56}
}


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    """
    Computes Accuracy, Recall, Precision, F1, and ROC-AUC for given predictions and threshold.
    """
    preds = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "F1": f1_score(y_true, preds, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "Threshold": threshold
    }


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, num_steps: int = 100) -> Tuple[float, float]:
    """
    Searches discrimination thresholds t in [0.00, 0.99] to identify the optimal F1 score.
    """
    thresholds = np.linspace(0.0, 0.99, num_steps)
    best_t = 0.50
    best_f1 = f1_score(y_true, (y_prob >= 0.50).astype(int), zero_division=0)

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = t

    return float(best_t), float(best_f1)


def generate_cv_threshold_plot(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    output_filepath: str,
    n_splits: int = 10,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Performs Stratified 10-Fold CV evaluating Precision, Recall, F1, and Queue Rate
    across discrimination thresholds with +- 1 std error bands.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    thresholds = np.linspace(0.0, 0.99, 100)
    metrics = {'precision': [], 'recall': [], 'f1': [], 'queue_rate': []}

    X_arr = np.array(X)
    y_arr = np.array(y)

    for train_idx, test_idx in cv.split(X_arr, y_arr):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        model_clone = clone(model)
        model_clone.fit(X_tr, y_tr)
        probs = model_clone.predict_proba(X_te)[:, 1]

        fold_p, fold_r, fold_f, fold_q = [], [], [], []
        for t in thresholds:
            p_bin = (probs >= t).astype(int)
            fold_p.append(precision_score(y_te, p_bin, zero_division=0))
            fold_r.append(recall_score(y_te, p_bin, zero_division=0))
            fold_f.append(f1_score(y_te, p_bin, zero_division=0))
            fold_q.append(np.mean(p_bin))

        metrics['precision'].append(fold_p)
        metrics['recall'].append(fold_r)
        metrics['f1'].append(fold_f)
        metrics['queue_rate'].append(fold_q)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    colors = {'precision': '#1f77b4', 'recall': '#2ca02c', 'f1': '#d62728', 'queue_rate': '#9467bd'}

    f1_means = np.mean(np.array(metrics['f1']), axis=0)
    best_idx = np.argmax(f1_means)
    best_t = thresholds[best_idx]
    best_f1 = f1_means[best_idx]

    for m_name, color in colors.items():
        arr = np.array(metrics[m_name])
        mean_val = np.mean(arr, axis=0)
        std_val = np.std(arr, axis=0)
        plt.plot(thresholds, mean_val, label=m_name.capitalize(), color=color, linewidth=2)
        plt.fill_between(
            thresholds,
            np.clip(mean_val - std_val, 0, 1),
            np.clip(mean_val + std_val, 0, 1),
            color=color, alpha=0.15
        )

    plt.axvline(best_t, color='black', linestyle='--', label=f'Optimal t={best_t:.2f} (F1={best_f1:.3f})')
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Discrimination Threshold')
    plt.ylabel('Metric Score')
    plt.legend(loc='best')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath, dpi=300)
    plt.close()

    return best_t, best_f1


def replicate_paper_table2(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_splits: int = 5
) -> pd.DataFrame:
    """
    Executes the exact Stratified 5-Fold Cross-Validation evaluation matching
    Xu et al. (2024) Table 2 methodology. Computes weighted Precision, Recall,
    F1 Score, Accuracy, and PR-AUC, returning a comparison DataFrame with exact deltas.
    """
    from src.models import get_paper_replication_models

    models = get_paper_replication_models(random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    rows = []
    print(f"[*] Replicating Xu et al. (2024) Table 2 across {len(models)} algorithms using Stratified {n_splits}-Fold CV...")

    for name, model in models.items():
        if name not in PAPER_TABLE_2:
            continue

        p_folds, r_folds, f_folds, acc_folds, auc_folds = [], [], [], [], []

        for tr_idx, te_idx in cv.split(X, y):
            # As noted in Xu et al. (2024) Section 4.1, Z-score standardization was applied for Logistic Regression
            if name == "Logistic Regression":
                X_tr, X_te = X_scaled.iloc[tr_idx], X_scaled.iloc[te_idx]
            else:
                X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            m_clone = clone(model)
            m_clone.fit(X_tr, y_tr)
            preds = m_clone.predict(X_te)
            probs = m_clone.predict_proba(X_te)[:, 1] if hasattr(m_clone, "predict_proba") else preds

            acc_folds.append(accuracy_score(y_te, preds))
            p_folds.append(precision_score(y_te, preds, average="weighted", zero_division=0))
            r_folds.append(recall_score(y_te, preds, average="weighted", zero_division=0))
            f_folds.append(f1_score(y_te, preds, average="weighted", zero_division=0))

            p_curve, r_curve, _ = precision_recall_curve(y_te, probs)
            auc_folds.append(auc(r_curve, p_curve))

        rep_p = float(np.mean(p_folds))
        rep_r = float(np.mean(r_folds))
        rep_f = float(np.mean(f_folds))
        rep_acc = float(np.mean(acc_folds))
        rep_auc = float(np.mean(auc_folds))

        ref = PAPER_TABLE_2[name]
        diff_r = rep_r - ref["Paper Recall"]
        diff_f = rep_f - ref["Paper F1"]
        diff_acc = rep_acc - ref["Paper Accuracy"]

        rows.append({
            "Algorithm": name,
            "Paper Recall": ref["Paper Recall"],
            "Replicated Recall": round(rep_r, 4),
            "Diff Recall": round(diff_r, 4),
            "Paper F1": ref["Paper F1"],
            "Replicated F1": round(rep_f, 4),
            "Diff F1": round(diff_f, 4),
            "Paper Accuracy": ref["Paper Accuracy"],
            "Replicated Accuracy": round(rep_acc, 4),
            "Diff Accuracy": round(diff_acc, 4),
            "Paper Precision": ref["Paper Precision"],
            "Replicated Precision": round(rep_p, 4),
            "Paper PR-AUC": ref["Paper AUC"],
            "Replicated PR-AUC": round(rep_auc, 4)
        })

    rep_df = pd.DataFrame(rows)
    return rep_df


def audit_paper_replication(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a comparative audit DataFrame measuring the exact leakage inflation gap
    between published paper values, naive leaky values, and honest corrected values
    on minority credit default detection (class 1).
    """
    comparison_rows = []
    for algo, paper_metrics in PAPER_TABLE_2.items():
        leaky_match = results_df[
            (results_df["Algorithm"] == algo) & (results_df["Scenario"] == "Leaky")
        ]
        corr_match = results_df[
            (results_df["Algorithm"] == algo) & (results_df["Scenario"] == "Corrected")
        ]

        if len(leaky_match) > 0 and len(corr_match) > 0:
            leaky_f1 = leaky_match.iloc[0]["F1"]
            corr_f1 = corr_match.iloc[0]["F1"]
            leaky_rec = leaky_match.iloc[0]["Recall"]
            corr_rec = corr_match.iloc[0]["Recall"]
            leaky_acc = leaky_match.iloc[0]["Accuracy"]
            corr_acc = corr_match.iloc[0]["Accuracy"]

            inflation_gap = leaky_f1 - corr_f1

            comparison_rows.append({
                "Algorithm": algo,
                "Paper F1": paper_metrics["Paper F1"],
                "Leaky F1 (Naive ROS)": round(leaky_f1, 4),
                "Corrected F1 (Honest)": round(corr_f1, 4),
                "F1 Inflation Gap": round(inflation_gap, 4),
                "Paper Recall": paper_metrics["Paper Recall"],
                "Leaky Recall (Naive ROS)": round(leaky_rec, 4),
                "Corrected Recall (Honest)": round(corr_rec, 4),
                "Paper Accuracy": paper_metrics["Paper Accuracy"],
                "Leaky Accuracy (Naive ROS)": round(leaky_acc, 4),
                "Corrected Accuracy (Honest)": round(corr_acc, 4)
            })

    audit_df = pd.DataFrame(comparison_rows).sort_values(by="F1 Inflation Gap", ascending=False)
    return audit_df
