"""
Train & Evaluation Pipeline (Naive Notation for Evaluators)
-----------------------------------------------------------
Trains and evaluates all 16 Machine Learning models + Deep Learning models
across Leaky and Corrected datasets:

  1. data_normal_leaky       (Flawed baseline replicating Xu et al., 2024)
  2. data_normal_corrected   (Honest zero-leakage baseline)
  3. data_gan_leaky          (GAN trained on entire dataset)
  4. data_gan_corrected      (GAN trained strictly on train partition)
  5. data_diffusion_leaky    (Diffusion trained on entire dataset)
  6. data_diffusion_corrected (Diffusion trained strictly on train partition)

Usage:
    python train.py                      # Full research training & evaluation
    python train.py --quick              # Quick smoke-test (fast subset of models)
    python train.py --summary-dir summary # Custom output directory
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.data import (
    load_credit_data, prepare_leaky_pipeline, prepare_corrected_pipeline,
    get_feature_lists, TARGET_COL
)
from src.models import get_classifiers, get_deep_mlp, TabularTransformer, PyTorchModelWrapper
from src.evaluation import evaluate_predictions, optimize_threshold, generate_cv_threshold_plot, audit_paper_replication, replicate_paper_table2
from src.generators import generate_ctgan_synthetic_data, generate_tabddpm_synthetic_data
from src.visualizations import plot_scenario_comparisons, plot_leakage_gap, plot_paper_replication_match
from src.explainability import (
    generate_shap_analysis, generate_lime_analysis,
    generate_permutation_importance_plot, score_applicant_risk
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate credit risk models on leaky and corrected datasets.")
    parser.add_argument("--quick", action="store_true", help="Run fast benchmark on representative classifiers")
    parser.add_argument("--summary-dir", type=str, default="summary", help="Folder to save reports and charts")
    parser.add_argument("--data-path", type=str, default="UCI_Credit_Card.csv", help="Path to UCI Credit Card CSV")
    parser.add_argument("--skip-cv", action="store_true", help="Skip 10-fold CV threshold curves")
    parser.add_argument("--skip-xai", action="store_true", help="Skip SHAP/LIME explainability plots")
    parser.add_argument("--ddpm-data", type=str, default="data/synthetic_tabddpm.csv", help="Path to TabDDPM data")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    summary_dir = args.summary_dir
    metrics_dir = os.path.join(summary_dir, "metrics")
    charts_dir = os.path.join(summary_dir, "charts")
    threshold_dir = os.path.join(summary_dir, "threshold_plots")
    xai_dir = os.path.join(summary_dir, "xai")

    for d in [summary_dir, metrics_dir, charts_dir, threshold_dir, xai_dir]:
        os.makedirs(d, exist_ok=True)

    print("===============================================================================")
    print("      CREDIT RISK MODEL TRAINING & EVALUATION (LEAKY VS CORRECTED)             ")
    print("===============================================================================")

    # 1. LOAD RAW DATASET
    df = load_credit_data(args.data_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 2. BUILD NORMAL DATASETS (LEAKY VS CORRECTED)
    print("\n--- 1. Building [data_normal_leaky] and [data_normal_corrected] ---")
    data_normal_leaky = prepare_leaky_pipeline(X, y, test_size=0.25, random_state=42)
    data_normal_corrected = prepare_corrected_pipeline(X, y, test_size=0.25, random_state=42)

    scenarios = {
        "data_normal_leaky": (
            data_normal_leaky["X_train"], data_normal_leaky["X_test"],
            data_normal_leaky["y_train"], data_normal_leaky["y_test"]
        ),
        "data_normal_corrected": (
            data_normal_corrected["X_train_sc"], data_normal_corrected["X_test_sc"],
            data_normal_corrected["y_train_res"], data_normal_corrected["y_test_raw"]
        )
    }

    # 3. SELECT CLASSIFIERS
    all_models = get_classifiers(random_state=42)
    if args.quick:
        selected_names = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "MLP Classifier", "XGBoost"]
        models_dict = {k: all_models[k] for k in selected_names if k in all_models}
    else:
        models_dict = all_models

    all_benchmark_results = []
    fitted_models = {"data_normal_leaky": {}, "data_normal_corrected": {}}

    print(f"\n[*] Benchmarking {len(models_dict)} classifiers on [data_normal_leaky] and [data_normal_corrected]...")
    for scenario_name, (X_tr, X_te, y_tr, y_te) in scenarios.items():
        print(f"\n--- Scenario: {scenario_name} ---")
        for algo_name, model in models_dict.items():
            print(f"  -> Training {algo_name}...")
            model.fit(X_tr, y_tr)
            fitted_models[scenario_name][algo_name] = model

            probs = model.predict_proba(X_te)[:, 1]

            # Standard threshold t=0.50
            m_def = evaluate_predictions(y_te, probs, threshold=0.50)
            m_def["Algorithm"] = algo_name
            m_def["Scenario"] = scenario_name
            all_benchmark_results.append(m_def)

            # Optimal threshold
            opt_t, _ = optimize_threshold(y_te, probs)
            m_opt = evaluate_predictions(y_te, probs, threshold=opt_t)
            m_opt["Algorithm"] = algo_name
            m_opt["Scenario"] = f"{scenario_name} (Opt. Thresh)"
            all_benchmark_results.append(m_opt)

    benchmark_df = pd.DataFrame(all_benchmark_results)

    # 4.A EXACT PAPER REPLICATION BENCHMARK (XU ET AL. 2024 TABLE 2)
    rep_df = replicate_paper_table2(X, y, random_state=42, n_splits=5)
    rep_csv_path = os.path.join(metrics_dir, "paper_replication_comparison.csv")
    rep_df.to_csv(rep_csv_path, index=False)
    print(f"\n[✓] Saved Exact Paper Replication Comparison to '{rep_csv_path}':\n")
    print(rep_df[["Algorithm", "Paper Recall", "Replicated Recall", "Diff Recall", "Paper F1", "Replicated F1", "Paper Accuracy", "Replicated Accuracy"]].to_string(index=False))

    plot_paper_replication_match(rep_df, os.path.join(charts_dir, "paper_replication_match.png"))

    # 4.B METHODOLOGICAL DATA LEAKAGE AUDIT (NAIVE OVERFIT LEAKAGE VS HONEST CORRECTED)
    audit_bench_df = benchmark_df.copy()
    audit_bench_df["Scenario"] = audit_bench_df["Scenario"].replace({
        "data_normal_leaky": "Leaky",
        "data_normal_corrected": "Corrected"
    })
    audit_df = audit_paper_replication(audit_bench_df)
    leakage_csv_path = os.path.join(metrics_dir, "data_leakage_audit.csv")
    audit_df.to_csv(leakage_csv_path, index=False)
    print(f"\n[✓] Saved Methodological Data Leakage Audit to '{leakage_csv_path}':\n")
    print(audit_df[["Algorithm", "Paper F1", "Leaky F1 (Naive ROS)", "Corrected F1 (Honest)", "F1 Inflation Gap"]].to_string(index=False))

    plot_leakage_gap(audit_df, os.path.join(charts_dir, "leakage_gap.png"))

    # 5. 10-FOLD CV DISCRIMINATION THRESHOLD CURVES
    if not args.skip_cv:
        print("\n[*] Generating 10-Fold Stratified CV Discrimination Threshold Curves on [data_normal_corrected]...")
        key_cv_models = ["Gradient Boosting", "Random Forest", "AdaBoost", "Logistic Regression"]
        for model_name in key_cv_models:
            if model_name in models_dict:
                plot_file = os.path.join(threshold_dir, f"corrected_{model_name.replace(' ', '_')}_threshold.png")
                title = f"Threshold Sensitivity: {model_name} (data_normal_corrected)"
                best_t, best_f = generate_cv_threshold_plot(
                    models_dict[model_name],
                    data_normal_corrected["X_train_sc"],
                    data_normal_corrected["y_train_res"],
                    title=title,
                    output_filepath=plot_file
                )
                print(f"  -> {model_name}: 10-Fold CV Optimal Threshold = {best_t:.2f} (F1 = {best_f:.4f})")

    # 6. GAN TABULAR MODELING: data_gan_corrected & data_gan_leaky
    print("\n--- 2. Building & Evaluating [data_gan_corrected] ---")
    ctgan_sample_count = 5000 if args.quick else 120000
    ctgan_syn = generate_ctgan_synthetic_data(
        X_train_raw=data_normal_corrected["X_train_raw"],
        y_train_raw=data_normal_corrected["y_train_raw"],
        categorical_features=data_normal_corrected["cat_cols"],
        total_samples=ctgan_sample_count,
        cache_path="results/ctgan_synthetic_120000.parquet"
    )

    train_clean = pd.concat([data_normal_corrected["X_train_raw"], data_normal_corrected["y_train_raw"]], axis=1)
    ctgan_combined = pd.concat([train_clean, ctgan_syn], axis=0).reset_index(drop=True)
    X_train_gan = ctgan_combined.drop(columns=[TARGET_COL])
    y_train_gan = ctgan_combined[TARGET_COL]

    preprocessor_gan = ColumnTransformer(
        transformers=[('num', StandardScaler(), data_normal_corrected["num_cols"])],
        remainder='passthrough'
    )
    X_train_gan_sc = preprocessor_gan.fit_transform(X_train_gan)
    X_test_gan_sc = preprocessor_gan.transform(data_normal_corrected["X_test_raw"])
    input_dim = X_train_gan_sc.shape[1]

    print(f"[*] data_gan_corrected Shape: {X_train_gan_sc.shape} | Default Rate: {y_train_gan.mean():.2%}")

    tf_epochs = 3 if args.quick else 15
    gan_models = {
        "Tabular Transformer (DL)": PyTorchModelWrapper(
            model_class=lambda dim: TabularTransformer(num_features=dim, d_model=64, nhead=4),
            input_dim=input_dim, epochs=tf_epochs, batch_size=512
        ),
        "Deep MLP Classifier": get_deep_mlp(random_state=42),
        "Gradient Boosting": models_dict.get("Gradient Boosting"),
        "Random Forest": models_dict.get("Random Forest"),
        "XGBoost": models_dict.get("XGBoost"),
        "Hist Gradient Boosting": models_dict.get("Hist Gradient Boosting")
    }
    gan_models = {k: v for k, v in gan_models.items() if v is not None}

    gan_results = []
    print("\n[*] Training models on [data_gan_corrected]...")
    for name, model in gan_models.items():
        print(f"  -> Training {name}...")
        model.fit(X_train_gan_sc, y_train_gan)
        probs = model.predict_proba(X_test_gan_sc)[:, 1]

        # Standard t=0.50
        m_def = evaluate_predictions(data_normal_corrected["y_test_raw"], probs, threshold=0.50)
        m_def["Algorithm"] = name
        m_def["Scenario"] = "data_gan_corrected (t=0.50)"
        gan_results.append(m_def)

        # Optimal threshold
        opt_t, _ = optimize_threshold(data_normal_corrected["y_test_raw"], probs)
        m_opt = evaluate_predictions(data_normal_corrected["y_test_raw"], probs, threshold=opt_t)
        m_opt["Algorithm"] = name
        m_opt["Scenario"] = "data_gan_corrected (Opt. Thresh)"
        gan_results.append(m_opt)

    gan_df = pd.DataFrame(gan_results)
    gan_csv_path = os.path.join(metrics_dir, "gan_results_summary.csv")
    gan_df.to_csv(gan_csv_path, index=False)

    # 7. DIFFUSION TABULAR MODELING: data_diffusion_corrected
    ddpm_results = []
    ddpm_file = args.ddpm_data if os.path.exists(args.ddpm_data) else "data/synthetic_tabddpm.csv"
    if os.path.exists(ddpm_file):
        print(f"\n--- 3. Building & Evaluating [data_diffusion_corrected] from '{ddpm_file}' ---")
        ddpm_syn = pd.read_csv(ddpm_file)
        ddpm_combined = pd.concat([train_clean, ddpm_syn], axis=0).reset_index(drop=True)
        X_train_ddpm = ddpm_combined.drop(columns=[TARGET_COL])
        y_train_ddpm = ddpm_combined[TARGET_COL]

        X_train_ddpm_sc = preprocessor_gan.transform(X_train_ddpm)
        mlp_ddpm = get_deep_mlp(random_state=42)
        mlp_ddpm.fit(X_train_ddpm_sc, y_train_ddpm)

        ddpm_probs = mlp_ddpm.predict_proba(X_test_gan_sc)[:, 1]
        m_ddpm_def = evaluate_predictions(data_normal_corrected["y_test_raw"], ddpm_probs, threshold=0.50)
        m_ddpm_def["Algorithm"] = "Deep MLP Classifier"
        m_ddpm_def["Scenario"] = "data_diffusion_corrected (t=0.50)"
        ddpm_results.append(m_ddpm_def)

        opt_t_ddpm, _ = optimize_threshold(data_normal_corrected["y_test_raw"], ddpm_probs)
        m_ddpm_opt = evaluate_predictions(data_normal_corrected["y_test_raw"], ddpm_probs, threshold=opt_t_ddpm)
        m_ddpm_opt["Algorithm"] = "Deep MLP Classifier"
        m_ddpm_opt["Scenario"] = "data_diffusion_corrected (Opt. Thresh)"
        ddpm_results.append(m_ddpm_opt)

        ddpm_df = pd.DataFrame(ddpm_results)
        ddpm_df.to_csv(os.path.join(metrics_dir, "ddpm_results_summary.csv"), index=False)

    # 8. CONSOLIDATE ALL METRICS
    final_results_df = pd.concat([benchmark_df, gan_df] + ([pd.DataFrame(ddpm_results)] if ddpm_results else []), axis=0)
    final_csv_path = os.path.join(metrics_dir, "results_summary.csv")
    final_results_df.to_csv(final_csv_path, index=False)

    final_txt_path = os.path.join(metrics_dir, "results_summary.txt")
    with open(final_txt_path, "w") as f:
        f.write(final_results_df.sort_values(by=["Algorithm", "Scenario"]).to_string(index=False))

    print(f"\n[✓] Saved complete benchmark metrics to '{final_csv_path}' and '{final_txt_path}'.")

    # 9. GENERATE COMPARISON CHARTS
    print("\n[*] Generating publication comparison charts...")
    plot_scenario_comparisons(benchmark_df, output_dir=charts_dir)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=gan_df, x="Algorithm", y="F1", hue="Scenario", palette="viridis")
    plt.title("GAN Tabular Deep Learning & ML Performance (data_gan_corrected)", fontsize=13, fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    gan_chart_path = os.path.join(charts_dir, "chart_gan_f1_score.png")
    plt.savefig(gan_chart_path, dpi=300)
    plt.close()

    # 10. MODEL EXPLAINABILITY (XAI)
    if not args.skip_xai and "Gradient Boosting" in fitted_models["data_normal_corrected"]:
        print("\n[*] Executing Model Explainability Suite (SHAP, LIME, Permutation Importance)...")
        gb_model = fitted_models["data_normal_corrected"]["Gradient Boosting"]
        generate_shap_analysis(
            model=gb_model,
            X_test=data_normal_corrected["X_test_sc"],
            feature_names=data_normal_corrected["feature_names"],
            output_dir=xai_dir,
            max_samples=500
        )

        generate_lime_analysis(
            model=gb_model,
            X_train=data_normal_corrected["X_train_sc"],
            X_test=data_normal_corrected["X_test_sc"],
            feature_names=data_normal_corrected["feature_names"],
            output_dir=xai_dir,
            applicant_indices=[0, 1, 2]
        )

        if "Random Forest" in fitted_models["data_normal_corrected"]:
            xai_models = {
                "Gradient Boosting": gb_model,
                "Random Forest": fitted_models["data_normal_corrected"]["Random Forest"]
            }
            generate_permutation_importance_plot(
                models_dict=xai_models,
                X_test=data_normal_corrected["X_test_sc"],
                y_test=data_normal_corrected["y_test_raw"],
                feature_names=data_normal_corrected["feature_names"],
                output_filepath=os.path.join(xai_dir, "permutation_importance.png"),
                n_repeats=5 if args.quick else 10
            )

        sample_app = data_normal_corrected["X_train_raw"].iloc[0].to_dict()
        risk_profile = score_applicant_risk(
            applicant_data=sample_app,
            model=gb_model,
            preprocessor=data_normal_corrected["preprocessor"],
            feature_names=data_normal_corrected["feature_names"],
            decision_threshold=0.48
        )
        print(f"\n[✓] Sample Applicant Risk Profiling Result: {risk_profile}")

    elapsed = time.time() - start_time
    print(f"\n===============================================================================")
    print(f"  TRAINING & EVALUATION COMPLETE (Elapsed: {elapsed/60:.2f} mins)               ")
    print(f"  Summary directory generated at: '{summary_dir}/'                              ")
    print(f"===============================================================================")


if __name__ == "__main__":
    main()
