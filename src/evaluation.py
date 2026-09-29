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


def format_leaky_vs_corrected(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the comparative benchmark into a compact, viewable 2-row-per-model table:
      Row 1: Standard (t=0.50)
      Row 2: Optimal Threshold
    Comparing Leaky vs Corrected datasets side-by-side with F1 Leakage Gap.
    """
    df = benchmark_df.copy()
    if "Threshold_Type" in df.columns:
        return df

    if "ROC_AUC" in df.columns and "ROC-AUC" not in df.columns:
        df = df.rename(columns={"ROC_AUC": "ROC-AUC"})

    scenario_map = {
        "data_normal_leaky": "Leaky (t=0.50)",
        "data_normal_leaky (Opt. Thresh)": "Leaky (Opt. Thresh)",
        "data_normal_corrected": "Corrected (t=0.50)",
        "data_normal_corrected (Opt. Thresh)": "Corrected (Opt. Thresh)",
        "Leaky (t=0.50)": "Leaky (t=0.50)",
        "Leaky (Opt. Thresh)": "Leaky (Opt. Thresh)",
        "Corrected (t=0.50)": "Corrected (t=0.50)",
        "Corrected (Opt. Thresh)": "Corrected (Opt. Thresh)"
    }
    df["Scenario"] = df["Scenario"].map(lambda s: scenario_map.get(s, s))

    rows = []
    for algo, grp in df.groupby("Algorithm", sort=True):
        l_def = grp[grp["Scenario"] == "Leaky (t=0.50)"]
        l_opt = grp[grp["Scenario"] == "Leaky (Opt. Thresh)"]
        c_def = grp[grp["Scenario"] == "Corrected (t=0.50)"]
        c_opt = grp[grp["Scenario"] == "Corrected (Opt. Thresh)"]

        if l_def.empty or c_def.empty:
            continue

        l_def = l_def.iloc[0]
        l_opt = l_opt.iloc[0] if not l_opt.empty else l_def
        c_def = c_def.iloc[0]
        c_opt = c_opt.iloc[0] if not c_opt.empty else c_def

        rows.append({
            "Algorithm": algo,
            "Threshold_Type": "Standard (t=0.50)",
            "Leaky_Threshold": round(float(l_def["Threshold"]), 2),
            "Leaky_F1": round(float(l_def["F1"]), 4),
            "Leaky_ROC_AUC": round(float(l_def["ROC-AUC"]), 4),
            "Leaky_Recall": round(float(l_def["Recall"]), 4),
            "Leaky_Precision": round(float(l_def["Precision"]), 4),
            "Leaky_Accuracy": round(float(l_def["Accuracy"]), 4) if "Accuracy" in l_def else np.nan,
            "Corrected_Threshold": round(float(c_def["Threshold"]), 2),
            "Corrected_F1": round(float(c_def["F1"]), 4),
            "Corrected_ROC_AUC": round(float(c_def["ROC-AUC"]), 4),
            "Corrected_Recall": round(float(c_def["Recall"]), 4),
            "Corrected_Precision": round(float(c_def["Precision"]), 4),
            "Corrected_Accuracy": round(float(c_def["Accuracy"]), 4) if "Accuracy" in c_def else np.nan,
            "F1_Leakage_Gap": round(float(l_def["F1"]) - float(c_def["F1"]), 4)
        })
        rows.append({
            "Algorithm": algo,
            "Threshold_Type": "Optimal Threshold",
            "Leaky_Threshold": round(float(l_opt["Threshold"]), 2),
            "Leaky_F1": round(float(l_opt["F1"]), 4),
            "Leaky_ROC_AUC": round(float(l_opt["ROC-AUC"]), 4),
            "Leaky_Recall": round(float(l_opt["Recall"]), 4),
            "Leaky_Precision": round(float(l_opt["Precision"]), 4),
            "Leaky_Accuracy": round(float(l_opt["Accuracy"]), 4) if "Accuracy" in l_opt else np.nan,
            "Corrected_Threshold": round(float(c_opt["Threshold"]), 2),
            "Corrected_F1": round(float(c_opt["F1"]), 4),
            "Corrected_ROC_AUC": round(float(c_opt["ROC-AUC"]), 4),
            "Corrected_Recall": round(float(c_opt["Recall"]), 4),
            "Corrected_Precision": round(float(c_opt["Precision"]), 4),
            "Corrected_Accuracy": round(float(c_opt["Accuracy"]), 4) if "Accuracy" in c_opt else np.nan,
            "F1_Leakage_Gap": round(float(l_opt["F1"]) - float(c_opt["F1"]), 4)
        })

    pivoted_df = pd.DataFrame(rows)
    return pivoted_df


def format_gan_vs_diffusion(gan_df: pd.DataFrame, diff_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Formats the GAN vs Diffusion comparative results into a compact, viewable 2-row-per-model table:
      Row 1: Standard (t=0.50)
      Row 2: Optimal Threshold
    Comparing GAN vs Diffusion datasets side-by-side with F1 Delta.
    """
    if "Threshold_Type" in gan_df.columns:
        return gan_df

    g = gan_df.copy()
    d = diff_df.copy() if diff_df is not None else pd.DataFrame()

    if "ROC_AUC" in g.columns and "ROC-AUC" not in g.columns:
        g = g.rename(columns={"ROC_AUC": "ROC-AUC"})
    if not d.empty and "ROC_AUC" in d.columns and "ROC-AUC" not in d.columns:
        d = d.rename(columns={"ROC_AUC": "ROC-AUC"})

    g_map = {
        "data_gan_corrected (t=0.50)": "GAN (t=0.50)",
        "data_gan_corrected (Opt. Thresh)": "GAN (Opt. Thresh)",
        "GAN (t=0.50)": "GAN (t=0.50)",
        "GAN (Opt. Thresh)": "GAN (Opt. Thresh)"
    }
    d_map = {
        "data_diffusion_corrected (t=0.50)": "Diffusion (t=0.50)",
        "data_diffusion_corrected (Opt. Thresh)": "Diffusion (Opt. Thresh)",
        "Diffusion (t=0.50)": "Diffusion (t=0.50)",
        "Diffusion (Opt. Thresh)": "Diffusion (Opt. Thresh)"
    }
    g["Scenario"] = g["Scenario"].map(lambda s: g_map.get(s, s))
    if not d.empty:
        d["Scenario"] = d["Scenario"].map(lambda s: d_map.get(s, s))
        combined = pd.concat([g, d], axis=0)
    else:
        combined = g

    rows = []
    for algo, grp in combined.groupby("Algorithm", sort=True):
        g_def = grp[grp["Scenario"] == "GAN (t=0.50)"]
        g_opt = grp[grp["Scenario"] == "GAN (Opt. Thresh)"]
        d_def = grp[grp["Scenario"] == "Diffusion (t=0.50)"]
        d_opt = grp[grp["Scenario"] == "Diffusion (Opt. Thresh)"]

        if g_def.empty or d_def.empty:
            continue

        g_def = g_def.iloc[0]
        g_opt = g_opt.iloc[0] if not g_opt.empty else g_def
        d_def = d_def.iloc[0]
        d_opt = d_opt.iloc[0] if not d_opt.empty else d_def

        rows.append({
            "Algorithm": algo,
            "Threshold_Type": "Standard (t=0.50)",
            "GAN_Threshold": round(float(g_def["Threshold"]), 2),
            "GAN_F1": round(float(g_def["F1"]), 4),
            "GAN_ROC_AUC": round(float(g_def["ROC-AUC"]), 4),
            "GAN_Recall": round(float(g_def["Recall"]), 4),
            "GAN_Precision": round(float(g_def["Precision"]), 4),
            "GAN_Accuracy": round(float(g_def["Accuracy"]), 4) if "Accuracy" in g_def else np.nan,
            "Diffusion_Threshold": round(float(d_def["Threshold"]), 2),
            "Diffusion_F1": round(float(d_def["F1"]), 4),
            "Diffusion_ROC_AUC": round(float(d_def["ROC-AUC"]), 4),
            "Diffusion_Recall": round(float(d_def["Recall"]), 4),
            "Diffusion_Precision": round(float(d_def["Precision"]), 4),
            "Diffusion_Accuracy": round(float(d_def["Accuracy"]), 4) if "Accuracy" in d_def else np.nan,
            "F1_Delta (GAN - Diff)": round(float(g_def["F1"]) - float(d_def["F1"]), 4)
        })
        rows.append({
            "Algorithm": algo,
            "Threshold_Type": "Optimal Threshold",
            "GAN_Threshold": round(float(g_opt["Threshold"]), 2),
            "GAN_F1": round(float(g_opt["F1"]), 4),
            "GAN_ROC_AUC": round(float(g_opt["ROC-AUC"]), 4),
            "GAN_Recall": round(float(g_opt["Recall"]), 4),
            "GAN_Precision": round(float(g_opt["Precision"]), 4),
            "GAN_Accuracy": round(float(g_opt["Accuracy"]), 4) if "Accuracy" in g_opt else np.nan,
            "Diffusion_Threshold": round(float(d_opt["Threshold"]), 2),
            "Diffusion_F1": round(float(d_opt["F1"]), 4),
            "Diffusion_ROC_AUC": round(float(d_opt["ROC-AUC"]), 4),
            "Diffusion_Recall": round(float(d_opt["Recall"]), 4),
            "Diffusion_Precision": round(float(d_opt["Precision"]), 4),
            "Diffusion_Accuracy": round(float(d_opt["Accuracy"]), 4) if "Accuracy" in d_opt else np.nan,
            "F1_Delta (GAN - Diff)": round(float(g_opt["F1"]) - float(d_opt["F1"]), 4)
        })

    pivoted_df = pd.DataFrame(rows)
    return pivoted_df


def format_size_scaling_table(size_df: pd.DataFrame, key_col: str = "Dataset_Size") -> pd.DataFrame:
    """
    Pivots raw scaling records into a compact 2-row-per-model table:
      Row 1: Standard (t=0.50)
      Row 2: Optimal Threshold
    With one column per value of key_col (e.g. F1_100k, ... for sizes, F1_50% ... for default ratios).
    """
    if "Threshold_Type" in size_df.columns:
        return size_df

    df = size_df.copy()
    if "ROC_AUC" in df.columns and "ROC-AUC" not in df.columns:
        df = df.rename(columns={"ROC_AUC": "ROC-AUC"})

    keys = list(dict.fromkeys(df[key_col]))
    if key_col == "Dataset_Size":
        keys = sorted(keys)
        key_strs = [f"{k//1000}k" if k < 1000000 else "1M" for k in keys]
    else:
        key_strs = [str(k) for k in keys]

    rows = []
    for algo, grp in df.groupby("Algorithm", sort=True):
        g_def = grp[grp["Scenario"].str.contains("t=0.50")]
        g_opt = grp[grp["Scenario"].str.contains("Opt. Thresh")]

        r_def = {"Algorithm": algo, "Threshold_Type": "Standard (t=0.50)"}
        r_opt = {"Algorithm": algo, "Threshold_Type": "Optimal Threshold"}

        metrics = [
            ("F1", "F1", 4),
            ("ROC-AUC", "AUC", 4),
            ("Recall", "Recall", 4),
            ("Precision", "Precision", 4),
            ("Threshold", "Thresh", 2),
            ("Train_Time_Sec", "Time_Sec", 2)
        ]

        for col_name, prefix, decimals in metrics:
            if col_name not in grp.columns:
                continue
            for k, k_s in zip(keys, key_strs):
                m_def = g_def[g_def[key_col] == k]
                m_opt = g_opt[g_opt[key_col] == k]
                r_def[f"{prefix}_{k_s}"] = round(float(m_def.iloc[0][col_name]), decimals) if not m_def.empty else None
                r_opt[f"{prefix}_{k_s}"] = round(float(m_opt.iloc[0][col_name]), decimals) if not m_opt.empty else None

        rows.append(r_def)
        rows.append(r_opt)

    pivoted_df = pd.DataFrame(rows)
    return pivoted_df


def _evaluate_augmented_training(
    syn_df: pd.DataFrame,
    models_dict: Dict[str, Any],
    X_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    X_test_raw: pd.DataFrame,
    y_test_raw: pd.Series,
    num_cols: List[str],
    generator_name: str,
    target_col: str = "default.payment.next.month"
) -> List[Dict[str, Any]]:
    """
    Trains every model on real training rows + syn_df (which may be empty) and scores the
    untouched real test set at t=0.50 and at the F1-optimal threshold.
    """
    import time
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    train_clean = pd.concat([X_train_raw, y_train_raw], axis=1)
    parts = [train_clean, syn_df] if len(syn_df) else [train_clean]
    combined_df = pd.concat(parts, axis=0).reset_index(drop=True)
    X_tr = combined_df.drop(columns=[target_col])
    y_tr = combined_df[target_col].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), num_cols)],
        remainder='passthrough'
    )
    X_tr_sc = preprocessor.fit_transform(X_tr)
    X_te_sc = preprocessor.transform(X_test_raw)

    print(f"[*] Training models on {len(X_tr):,} combined records ({y_tr.mean():.1%} defaults)...")
    records = []
    for algo_name, model_template in models_dict.items():
        t_tr = time.time()
        if hasattr(model_template, "model_class"):
            # Re-instantiate PyTorch wrapper for the new input_dim
            from src.models import PyTorchModelWrapper, TabularTransformer
            m = PyTorchModelWrapper(
                model_class=lambda dim: TabularTransformer(num_features=dim, d_model=64, nhead=4),
                input_dim=X_tr_sc.shape[1], epochs=model_template.epochs, batch_size=model_template.batch_size
            )
        else:
            m = clone(model_template)

        m.fit(X_tr_sc, y_tr)
        train_time = round(time.time() - t_tr, 2)
        probs = m.predict_proba(X_te_sc)[:, 1]

        opt_t, _ = optimize_threshold(y_test_raw, probs)
        for label, t in [("t=0.50", 0.50), ("Opt. Thresh", opt_t)]:
            m_eval = evaluate_predictions(y_test_raw, probs, threshold=t)
            records.append({
                "Total_Train_Rows": len(X_tr),
                "Algorithm": algo_name,
                "Scenario": f"{generator_name} ({label})",
                "Threshold": round(t, 4),
                "Accuracy": round(m_eval["Accuracy"], 4),
                "Precision": round(m_eval["Precision"], 4),
                "Recall": round(m_eval["Recall"], 4),
                "F1": round(m_eval["F1"], 4),
                "ROC-AUC": round(m_eval["ROC_AUC"], 4),
                "Train_Time_Sec": train_time
            })
    return records


def benchmark_dataset_sizes(
    generator_fn: Any,
    sizes: List[int],
    models_dict: Dict[str, Any],
    X_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    X_test_raw: pd.DataFrame,
    y_test_raw: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    target_col: str = "default.payment.next.month",
    generator_name: str = "GAN"
) -> pd.DataFrame:
    """
    Trains models across scaling dataset sizes (e.g. 100k, 200k, 300k, 400k, 500k, 1M)
    and logs evaluation metrics and training runtime. Size 0 = real training data only.
    """
    import time

    records = []
    for sz in sizes:
        if sz > 0:
            print(f"\n[{generator_name} Size Scaling] Generating {sz:,} synthetic samples...")
            t_gen = time.time()
            syn_df = generator_fn(total_samples=sz)
            print(f"[*] Generation took {time.time() - t_gen:.2f}s | Synthetic shape: {syn_df.shape}")
        else:
            syn_df = pd.DataFrame(columns=list(X_train_raw.columns) + [target_col])

        for rec in _evaluate_augmented_training(
            syn_df, models_dict, X_train_raw, y_train_raw, X_test_raw, y_test_raw,
            num_cols, generator_name, target_col
        ):
            records.append({"Dataset_Size": sz, **rec})

    return pd.DataFrame(records)


def benchmark_default_ratios(
    generator_fn: Any,
    ratios: List[float],
    total_samples: int,
    models_dict: Dict[str, Any],
    X_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    X_test_raw: pd.DataFrame,
    y_test_raw: pd.Series,
    num_cols: List[str],
    target_col: str = "default.payment.next.month",
    generator_name: str = "Diffusion"
) -> pd.DataFrame:
    """
    Adds a fixed number of synthetic rows with a varying default (risk) : non-default (no risk)
    ratio to the real training data and evaluates on the untouched real test set.
    A 'real_only' baseline (no synthetic rows) is evaluated first.
    """
    records = []
    empty = pd.DataFrame(columns=list(X_train_raw.columns) + [target_col])
    for rec in _evaluate_augmented_training(
        empty, models_dict, X_train_raw, y_train_raw, X_test_raw, y_test_raw,
        num_cols, generator_name, target_col
    ):
        records.append({"Default_Ratio": "real_only", **rec})

    for r in ratios:
        print(f"\n[{generator_name} Ratio] {total_samples:,} synthetic rows at {r:.0%} defaults...")
        syn_df = generator_fn(total_samples=total_samples, default_ratio=r)
        for rec in _evaluate_augmented_training(
            syn_df, models_dict, X_train_raw, y_train_raw, X_test_raw, y_test_raw,
            num_cols, generator_name, target_col
        ):
            records.append({"Default_Ratio": f"{r:.0%}", **rec})

    return pd.DataFrame(records)
