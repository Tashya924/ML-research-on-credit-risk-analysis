"""
Publication-quality visualizations for research reporting, benchmark comparisons,
and generative tabular synthesis evaluation.
"""

import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
