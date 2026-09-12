"""
Model evaluation, discrimination threshold calibration, and paper replication audit.
"""

import os
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns

# Ground truth metrics from Table 2 of Xu et al. (2024)
PAPER_TABLE_2 = {
    "Decision Tree": {"Paper F1": 0.80, "Paper Recall": 0.81, "Paper Accuracy": 0.81},
    "Random Forest": {"Paper F1": 0.80, "Paper Recall": 0.82, "Paper Accuracy": 0.82},
    "KNN": {"Paper F1": 0.71, "Paper Recall": 0.75, "Paper Accuracy": 0.75},
    "Gaussian Naive Bayes": {"Paper F1": 0.39, "Paper Recall": 0.39, "Paper Accuracy": 0.39},
    "LightGBM": {"Paper F1": 0.78, "Paper Recall": 0.79, "Paper Accuracy": 0.79},
    "MLP Classifier": {"Paper F1": 0.73, "Paper Recall": 0.74, "Paper Accuracy": 0.74},
    "Logistic Regression": {"Paper F1": 0.68, "Paper Recall": 0.78, "Paper Accuracy": 0.78},
    "Gradient Boosting": {"Paper F1": 0.80, "Paper Recall": 0.82, "Paper Accuracy": 0.82},
    "LDA": {"Paper F1": 0.78, "Paper Recall": 0.81, "Paper Accuracy": 0.81},
    "AdaBoost": {"Paper F1": 0.79, "Paper Recall": 0.82, "Paper Accuracy": 0.82}
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


def audit_paper_replication(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a comparative audit DataFrame measuring the exact leakage inflation gap
    between published paper values, replicated leaky values, and honest corrected values.
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
                "Leaky F1 (Replicated)": round(leaky_f1, 4),
                "Corrected F1 (Honest)": round(corr_f1, 4),
                "F1 Inflation Gap": round(inflation_gap, 4),
                "Paper Recall": paper_metrics["Paper Recall"],
                "Leaky Recall": round(leaky_rec, 4),
                "Corrected Recall": round(corr_rec, 4),
                "Paper Accuracy": paper_metrics["Paper Accuracy"],
                "Leaky Accuracy": round(leaky_acc, 4),
                "Corrected Accuracy": round(corr_acc, 4)
            })

    audit_df = pd.DataFrame(comparison_rows).sort_values(by="F1 Inflation Gap", ascending=False)
    return audit_df
