"""
Publication-quality visualizations for research reporting, benchmark comparisons,
and generative tabular synthesis evaluation.
"""

import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_scenario_comparisons(results_df: pd.DataFrame, output_dir: str = "summary/charts"):
    """
    Generates comparison bar plots across all algorithms and evaluation scenarios
    for ROC-AUC, F1 Score, Precision, and Recall.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    metrics = [
        ("ROC_AUC", "chart_1_roc_auc.png", "ROC-AUC Comparison across Scenarios & Thresholds"),
        ("F1", "chart_2_f1_score.png", "F1 Score Comparison: The Data Leakage Gap"),
        ("Precision", "chart_3_precision.png", "Precision Comparison across Scenarios"),
        ("Recall", "chart_4_recall.png", "Recall Comparison across Scenarios")
    ]

    for metric_col, fname, title in metrics:
        if metric_col not in results_df.columns:
            continue

        plt.figure(figsize=(14, 11))
        # Order by maximum score on that metric
        order = (
            results_df.groupby("Algorithm")[metric_col]
            .max()
            .sort_values(ascending=False)
            .index
        )
        
        ax = sns.barplot(
            data=results_df,
            x=metric_col,
            y="Algorithm",
            hue="Scenario",
            order=order,
            palette="Set2"
        )
        plt.xlim(0.0, 1.0)
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.xlabel(metric_col.replace("_", " "), fontsize=12)
        plt.ylabel("Algorithm", fontsize=12)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()

        save_path = os.path.join(output_dir, fname)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[*] Saved chart to '{save_path}'")


def plot_leakage_gap(audit_df: pd.DataFrame, output_filepath: str = "summary/charts/leakage_gap.png"):
    """
    Plots the explicit F1 score inflation gap between published, naive leaky, and honest models.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    plot_data = audit_df.sort_values(by="F1 Inflation Gap", ascending=True)

    leaky_col = "Leaky F1 (Naive ROS)" if "Leaky F1 (Naive ROS)" in plot_data.columns else "Leaky F1 (Replicated)"
    corr_col = "Corrected F1 (Honest)"

    y_pos = range(len(plot_data))
    plt.hlines(y=y_pos, xmin=plot_data[corr_col], xmax=plot_data[leaky_col], color='grey', alpha=0.5, linewidth=2)
    plt.scatter(plot_data[corr_col], y_pos, color='#2ca02c', s=100, label='Honest Corrected F1 (Pristine Test)', zorder=3)
    plt.scatter(plot_data[leaky_col], y_pos, color='#d62728', s=100, label='Naive Leaky F1 (Overfit Leakage)', zorder=3)
    plt.scatter(plot_data["Paper F1"], y_pos, color='#1f77b4', marker='x', s=100, label='Xu et al. (2024) Published F1', zorder=3)

    for idx, (_, row) in enumerate(plot_data.iterrows()):
        mid = (row[corr_col] + row[leaky_col]) / 2
        plt.text(mid, idx + 0.2, f"+{row['F1 Inflation Gap']:.3f}", color='#d62728', fontweight='bold', fontsize=9, ha='center')

    plt.yticks(y_pos, plot_data["Algorithm"], fontsize=11)
    plt.xlabel("F1 Score", fontsize=12)
    plt.title("The Data Leakage Inflation Gap (Naive Leaky vs. Honest Corrected F1)", fontsize=13, fontweight='bold', pad=15)
    plt.xlim(0.2, 1.0)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()

    plt.savefig(output_filepath, dpi=300)
    plt.close()
    print(f"[*] Saved leakage gap visualization to '{output_filepath}'")


def plot_paper_replication_match(
    rep_df: pd.DataFrame,
    output_filepath: str = "summary/charts/paper_replication_match.png"
):
    """
    Generates a grouped bar chart comparing Xu et al. (2024) published Recall & F1
    against our Stratified 5-Fold Cross-Validation replicated baseline.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.figure(figsize=(13, 7))
    sns.set_theme(style="whitegrid")

    # Reshape for seaborn
    melted = []
    for _, row in rep_df.iterrows():
        algo = row["Algorithm"]
        melted.append({"Algorithm": algo, "Metric": "Paper Recall", "Score": row["Paper Recall"]})
        melted.append({"Algorithm": algo, "Metric": "Replicated Recall (5-Fold CV)", "Score": row["Replicated Recall"]})
        melted.append({"Algorithm": algo, "Metric": "Paper F1", "Score": row["Paper F1"]})
        melted.append({"Algorithm": algo, "Metric": "Replicated F1 (5-Fold CV)", "Score": row["Replicated F1"]})

    m_df = pd.DataFrame(melted)

    palette = {
        "Paper Recall": "#1f77b4",
        "Replicated Recall (5-Fold CV)": "#aec7e8",
        "Paper F1": "#2ca02c",
        "Replicated F1 (5-Fold CV)": "#98df8a"
    }

    ax = sns.barplot(data=m_df, x="Algorithm", y="Score", hue="Metric", palette=palette)
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=25, ha='right', fontsize=10)
    plt.title("Paper Empirical Replication: Xu et al. (2024) vs. Replicated 5-Fold CV Baseline (Δ ≤ 0.01)", fontsize=13, fontweight='bold', pad=15)
    plt.ylabel("Score", fontsize=11)
    plt.xlabel("Algorithm", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    plt.savefig(output_filepath, dpi=300)
    plt.close()
    print(f"[*] Saved paper replication match chart to '{output_filepath}'")


def plot_generative_comparison(
    gen_results_df: pd.DataFrame,
    output_filepath: str = "summary/charts/generative_comparison.png"
):
    """
    Bar plot comparing Generative models (CTGAN vs TabDDPM vs Baseline Corrected) on F1 and ROC-AUC.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    if "Scenario" in gen_results_df.columns:
        sns.barplot(data=gen_results_df, x="Algorithm", y="F1", hue="Scenario", palette="Spectral")
        plt.title("Performance Comparison: Baseline vs. CTGAN vs. TabDDPM Augmentation", fontsize=13, fontweight='bold')
        plt.ylabel("F1 Score (Pristine Test Set)", fontsize=11)
        plt.xticks(rotation=25, ha='right')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_filepath, dpi=300)
        plt.close()
        print(f"[*] Saved generative comparison chart to '{output_filepath}'")


def compute_synthetic_fidelity_metrics(
    real_df: pd.DataFrame,
    gan_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    features: List[str]
) -> pd.DataFrame:
    """
    Computes Kolmogorov-Smirnov test statistics, Wasserstein distances, and mean/std
    comparisons between empirical Real data, CTGAN, and TabDDPM synthetic distributions.
    """
    from scipy.stats import ks_2samp, wasserstein_distance
    rows = []
    for f in features:
        if f not in real_df.columns or f not in gan_df.columns or f not in diff_df.columns:
            continue
        r_vals = real_df[f].dropna().values.astype(float)
        g_vals = gan_df[f].dropna().values.astype(float)
        d_vals = diff_df[f].dropna().values.astype(float)

        ks_gan, _ = ks_2samp(r_vals, g_vals)
        ks_diff, _ = ks_2samp(r_vals, d_vals)
        w_gan = wasserstein_distance(r_vals, g_vals)
        w_diff = wasserstein_distance(r_vals, d_vals)

        rows.append({
            "Feature": f,
            "Real_Mean": round(float(np.mean(r_vals)), 2),
            "GAN_Mean": round(float(np.mean(g_vals)), 2),
            "Diff_Mean": round(float(np.mean(d_vals)), 2),
            "Real_Std": round(float(np.std(r_vals)), 2),
            "GAN_Std": round(float(np.std(g_vals)), 2),
            "Diff_Std": round(float(np.std(d_vals)), 2),
            "KS_Stat_GAN": round(float(ks_gan), 4),
            "Wasserstein_GAN": round(float(w_gan), 2),
            "KS_Stat_Diff": round(float(ks_diff), 4),
            "Wasserstein_Diff": round(float(w_diff), 2)
        })

    return pd.DataFrame(rows)


def plot_synthetic_feature_distributions(
    real_df: pd.DataFrame,
    gan_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    key_features: List[str],
    output_path: str = "summary/charts/synthetic_feature_distributions.png"
):
    """
    Generates high-resolution multi-panel KDE plots comparing Real vs CTGAN vs TabDDPM.
    """
    sns.set_theme(style="whitegrid")
    n_feat = len(key_features)
    n_cols = 3
    n_rows = (n_feat + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    colors = {"Real Data": "#1f77b4", "CTGAN": "#2ca02c", "TabDDPM": "#ff7f0e"}

    for i, col in enumerate(key_features):
        ax = axes[i]
        sns.kdeplot(real_df[col].astype(float), ax=ax, label="Real Data", color=colors["Real Data"], linewidth=2.5)
        sns.kdeplot(gan_df[col].astype(float), ax=ax, label="CTGAN", color=colors["CTGAN"], linewidth=2, linestyle="--")
        sns.kdeplot(diff_df[col].astype(float), ax=ax, label="TabDDPM", color=colors["TabDDPM"], linewidth=2, linestyle=":")
        ax.set_title(f"Distribution: {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(loc="upper right", fontsize=9)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Empirical Density Comparison: Real vs. CTGAN vs. TabDDPM Diffusion", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] Saved feature distribution comparison to '{output_path}'")


def plot_synthetic_categorical_fidelity(
    real_df: pd.DataFrame,
    gan_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    cat_features: List[str],
    output_path: str = "summary/charts/synthetic_categorical_fidelity.png"
):
    """
    Bar plots displaying frequency preservation for repayment statuses and target default rate.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(cat_features), figsize=(4.2 * len(cat_features), 4.2))
    if len(cat_features) == 1:
        axes = [axes]

    for i, col in enumerate(cat_features):
        ax = axes[i]
        val_counts = []
        for name, d in [("Real", real_df), ("CTGAN", gan_df), ("TabDDPM", diff_df)]:
            vc = d[col].value_counts(normalize=True).reset_index()
            vc.columns = [col, "Proportion"]
            vc["Dataset"] = name
            val_counts.append(vc)
        comb_vc = pd.concat(val_counts, axis=0)

        sns.barplot(data=comb_vc, x=col, y="Proportion", hue="Dataset", ax=ax, palette="muted")
        ax.set_title(f"Fidelity: {col}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Frequency Proportion")
        ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("Discrete & Categorical Proportion Fidelity", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] Saved categorical fidelity to '{output_path}'")


def plot_synthetic_correlation_fidelity(
    real_df: pd.DataFrame,
    gan_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    features: List[str],
    output_path: str = "summary/charts/synthetic_correlation_fidelity.png"
):
    """
    Compares 10x10 correlation matrices across Real, CTGAN, and TabDDPM to verify financial covariance.
    """
    sub_cols = [c for c in features if c in real_df.columns][:10]
    r_corr = real_df[sub_cols].astype(float).corr()
    g_corr = gan_df[sub_cols].astype(float).corr()
    d_corr = diff_df[sub_cols].astype(float).corr()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(r_corr, ax=axes[0], cmap="vlag", vmin=-1, vmax=1, cbar=False, annot=False)
    axes[0].set_title("Real Data Correlation Matrix", fontsize=12, fontweight="bold")

    sns.heatmap(g_corr, ax=axes[1], cmap="vlag", vmin=-1, vmax=1, cbar=False, annot=False)
    axes[1].set_title("CTGAN Correlation Matrix", fontsize=12, fontweight="bold")

    sns.heatmap(d_corr, ax=axes[2], cmap="vlag", vmin=-1, vmax=1, cbar=True, annot=False)
    axes[2].set_title("TabDDPM Correlation Matrix", fontsize=12, fontweight="bold")

    plt.suptitle("Inter-Feature Covariance & Correlation Structure Preservation", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] Saved correlation fidelity to '{output_path}'")


def plot_synthetic_pca_manifold(
    real_df: pd.DataFrame,
    gan_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    features: List[str],
    output_path: str = "summary/charts/synthetic_pca_manifold.png",
    n_points: int = 1500
):
    """
    Visualizes 2D PCA manifold projection of Real vs CTGAN vs TabDDPM samples.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    cols = [c for c in features if c in real_df.columns and c != "default.payment.next.month"]

    r_sample = real_df[cols].sample(min(n_points, len(real_df)), random_state=42)
    g_sample = gan_df[cols].sample(min(n_points, len(gan_df)), random_state=42)
    d_sample = diff_df[cols].sample(min(n_points, len(diff_df)), random_state=42)

    X_real_sc = scaler.fit_transform(r_sample)
    X_gan_sc = scaler.transform(g_sample)
    X_diff_sc = scaler.transform(d_sample)

    pca = PCA(n_components=2, random_state=42)
    pca_real = pca.fit_transform(X_real_sc)
    pca_gan = pca.transform(X_gan_sc)
    pca_diff = pca.transform(X_diff_sc)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    plt.scatter(pca_real[:, 0], pca_real[:, 1], alpha=0.35, label="Real Data", color="#1f77b4", s=18)
    plt.scatter(pca_gan[:, 0], pca_gan[:, 1], alpha=0.35, label="CTGAN", color="#2ca02c", s=18)
    plt.scatter(pca_diff[:, 0], pca_diff[:, 1], alpha=0.35, label="TabDDPM", color="#ff7f0e", s=18)

    ev = pca.explained_variance_ratio_
    plt.title(f"2D PCA Manifold Overlap (PC1: {ev[0]:.1%}, PC2: {ev[1]:.1%})", fontsize=13, fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({ev[0]:.1%})")
    plt.ylabel(f"Principal Component 2 ({ev[1]:.1%})")
    plt.legend(loc="upper right", markerscale=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] Saved 2D PCA manifold projection to '{output_path}'")

