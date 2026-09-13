"""
Train & Evaluation Pipeline (Standardized Research Benchmark)
-------------------------------------------------------------
Generates the 5 standardized research CSV benchmark files in summary/metrics/:

  1. leaky_replication.csv    (Exact replication comparison vs Xu et al. 2024 Table 2)
  2. leaky_vs_corrected.csv   (4 results per model: Leaky vs Corrected x Normal vs Opt Threshold)
  3. GAN_vs_Diffusion.csv     (4 results per model: GAN vs Diffusion x Normal vs Opt Threshold)
  4. Gan_size.csv             (Dataset size scaling: 100k, 200k, 300k, 400k, 500k, 1M)
  5. Diffusion_size.csv       (Dataset size scaling on Diffusion model data)

Usage:
    python train.py                     # Full research training & size evaluation
    python train.py --quick             # Fast smoke-test mode (~2 mins)
    python train.py --skip-sizes        # Run benchmarks 1, 2, 3 only
    python train.py --gan-sizes 100000 200000 500000 1000000
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
from sklearn.base import clone

from src.data import (
    load_credit_data, prepare_leaky_pipeline, prepare_corrected_pipeline,
    get_feature_lists, TARGET_COL
)
from src.models import get_classifiers, get_deep_mlp, TabularTransformer, PyTorchModelWrapper
from src.evaluation import (
    evaluate_predictions, optimize_threshold, generate_cv_threshold_plot,
    replicate_paper_table2, format_leaky_vs_corrected, format_gan_vs_diffusion,
    benchmark_dataset_sizes
)
from src.generator_gan import generate_ctgan_synthetic_data
from src.generator_diffusion import generate_tabddpm_synthetic_data
from src.visualizations import plot_scenario_comparisons, plot_paper_replication_match
from src.explainability import (
    generate_shap_analysis, generate_lime_analysis,
    generate_permutation_importance_plot, score_applicant_risk
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate credit risk models on leaky, corrected, and generative datasets.")
    parser.add_argument("--quick", action="store_true", help="Run fast benchmark on representative classifiers and smaller sizes")
    parser.add_argument("--summary-dir", type=str, default="summary", help="Folder to save reports and charts")
    parser.add_argument("--data-path", type=str, default="UCI_Credit_Card.csv", help="Path to UCI Credit Card CSV")
    parser.add_argument("--skip-cv", action="store_true", help="Skip 10-fold CV threshold curves")
    parser.add_argument("--skip-xai", action="store_true", help="Skip SHAP/LIME explainability plots")
    parser.add_argument("--skip-sizes", action="store_true", help="Skip dataset size scaling experiments")
    parser.add_argument("--gan-sizes", nargs="+", type=int, default=None,
                        help="List of GAN dataset sizes (default: 100k 200k 300k 400k 500k 1000k)")
    parser.add_argument("--diffusion-sizes", nargs="+", type=int, default=None,
                        help="List of Diffusion dataset sizes (default: 100k 200k 300k 400k 500k 1000k)")
    parser.add_argument("--ddpm-data", type=str, default="data/synthetic_tabddpm.csv", help="Path to TabDDPM data")
    return parser.parse_args()


def plot_size_scaling(df: pd.DataFrame, title: str, output_path: str):
    """Generates size scaling curves showing F1 score vs synthetic dataset size."""
    if df.empty or "Dataset_Size" not in df.columns:
        return
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    plot_df = df[df["Scenario"].str.contains("Opt. Thresh")].copy()
    if plot_df.empty:
        plot_df = df.copy()

    sns.lineplot(data=plot_df, x="Dataset_Size", y="F1", hue="Algorithm", marker="o", linewidth=2)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Synthetic Dataset Size (Rows)", fontsize=11)
    plt.ylabel("Test F1 Score (Optimal Threshold)", fontsize=11)
    plt.xscale("log")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()



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

    raw_bench_df = pd.DataFrame(all_benchmark_results)

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: LEAKY REPLICATION (Xu et al., 2024 Table 2)
    # -------------------------------------------------------------------------
    print("\n[*] Running Experiment 1: Exact Paper Replication Benchmark...")
    rep_df = replicate_paper_table2(X, y, random_state=42, n_splits=5)
    rep_path = os.path.join(metrics_dir, "leaky_replication.csv")
    rep_df.to_csv(rep_path, index=False)
    print(f"\n[✓] Saved Experiment 1 to '{rep_path}':\n")
    print(rep_df[["Algorithm", "Paper Recall", "Replicated Recall", "Diff Recall", "Paper F1", "Replicated F1", "Paper Accuracy", "Replicated Accuracy"]].to_string(index=False))
    plot_paper_replication_match(rep_df, os.path.join(charts_dir, "paper_replication_match.png"))

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: LEAKY VS CORRECTED (Original 30k, 4 Results Per Model)
    # -------------------------------------------------------------------------
    print("\n[*] Running Experiment 2: Leaky vs Corrected Comparison...")
    leaky_vs_corr_df = format_leaky_vs_corrected(raw_bench_df)
    leaky_vs_corr_path = os.path.join(metrics_dir, "leaky_vs_corrected.csv")
    leaky_vs_corr_df.to_csv(leaky_vs_corr_path, index=False)
    print(f"\n[✓] Saved Experiment 2 (4 results per model) to '{leaky_vs_corr_path}':")
    print(leaky_vs_corr_df.head(8).to_string(index=False))

    plot_scenario_comparisons(raw_bench_df, output_dir=charts_dir)

    # 10-Fold CV threshold curves
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

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: GAN VS DIFFUSION (Corrected, 4 Results Per Model)
    # -------------------------------------------------------------------------
    print("\n[*] Running Experiment 3: GAN vs Diffusion Benchmark (All Models)...")
    train_clean = pd.concat([data_normal_corrected["X_train_raw"], data_normal_corrected["y_train_raw"]], axis=1)

    # 3.A Build GAN Training Data
    ctgan_sample_count = 5000 if args.quick else 120000
    ctgan_syn = generate_ctgan_synthetic_data(
        X_train_raw=data_normal_corrected["X_train_raw"],
        y_train_raw=data_normal_corrected["y_train_raw"],
        categorical_features=data_normal_corrected["cat_cols"],
        total_samples=ctgan_sample_count,
        cache_path="data/ctgan_synthetic_120000.parquet"
    )
    ctgan_combined = pd.concat([train_clean, ctgan_syn], axis=0).reset_index(drop=True)
    X_train_gan = ctgan_combined.drop(columns=[TARGET_COL])
    y_train_gan = ctgan_combined[TARGET_COL].astype(int)

    preprocessor_gan = ColumnTransformer(
        transformers=[('num', StandardScaler(), data_normal_corrected["num_cols"])],
        remainder='passthrough'
    )
    X_train_gan_sc = preprocessor_gan.fit_transform(X_train_gan)
    X_test_gan_sc = preprocessor_gan.transform(data_normal_corrected["X_test_raw"])
    input_dim = X_train_gan_sc.shape[1]

    # 3.B Build Diffusion Training Data
    ddpm_file = args.ddpm_data if os.path.exists(args.ddpm_data) else "data/synthetic_tabddpm.csv"
    if not os.path.exists(ddpm_file) and os.path.exists("data/synthetic_tabddpm_pool.parquet"):
        ddpm_file = "data/synthetic_tabddpm_pool.parquet"
    
    if ddpm_file.endswith(".parquet"):
        ddpm_syn = pd.read_parquet(ddpm_file)
    else:
        ddpm_syn = pd.read_csv(ddpm_file)

    if args.quick and len(ddpm_syn) > 2000:
        ddpm_syn = ddpm_syn.sample(n=2000, random_state=42)

    ddpm_combined = pd.concat([train_clean, ddpm_syn], axis=0).reset_index(drop=True)
    X_train_ddpm = ddpm_combined.drop(columns=[TARGET_COL])
    y_train_ddpm = ddpm_combined[TARGET_COL].astype(int)

    preprocessor_ddpm = ColumnTransformer(
        transformers=[('num', StandardScaler(), data_normal_corrected["num_cols"])],
        remainder='passthrough'
    )
    X_train_ddpm_sc = preprocessor_ddpm.fit_transform(X_train_ddpm)
    X_test_ddpm_sc = preprocessor_ddpm.transform(data_normal_corrected["X_test_raw"])

    tf_epochs = 3 if args.quick else 15
    eval_models = {
        "Tabular Transformer (DL)": PyTorchModelWrapper(
            model_class=lambda dim: TabularTransformer(num_features=dim, d_model=64, nhead=4),
            input_dim=input_dim, epochs=tf_epochs, batch_size=512
        ),
        "Deep MLP Classifier": get_deep_mlp(random_state=42),
        "Gradient Boosting": models_dict.get("Gradient Boosting"),
        "Random Forest": models_dict.get("Random Forest"),
        "XGBoost": models_dict.get("XGBoost"),
        "Logistic Regression": models_dict.get("Logistic Regression"),
        "Decision Tree": models_dict.get("Decision Tree"),
        "AdaBoost": models_dict.get("AdaBoost")
    }
    eval_models = {k: v for k, v in eval_models.items() if v is not None}

    gan_records = []
    print("\n  -> Evaluating models on GAN-augmented distribution...")
    for name, model_tmpl in eval_models.items():
        if hasattr(model_tmpl, "model_class"):
            m = PyTorchModelWrapper(
                model_class=lambda dim: TabularTransformer(num_features=dim, d_model=64, nhead=4),
                input_dim=input_dim, epochs=tf_epochs, batch_size=512
            )
        else:
            m = clone(model_tmpl)

        m.fit(X_train_gan_sc, y_train_gan)
        probs = m.predict_proba(X_test_gan_sc)[:, 1]

        m_def = evaluate_predictions(data_normal_corrected["y_test_raw"], probs, threshold=0.50)
        m_def["Algorithm"] = name
        m_def["Scenario"] = "GAN (t=0.50)"
        gan_records.append(m_def)

        opt_t, _ = optimize_threshold(data_normal_corrected["y_test_raw"], probs)
        m_opt = evaluate_predictions(data_normal_corrected["y_test_raw"], probs, threshold=opt_t)
        m_opt["Algorithm"] = name
        m_opt["Scenario"] = "GAN (Opt. Thresh)"
        gan_records.append(m_opt)

    diff_records = []
    print("\n  -> Evaluating models on Diffusion-augmented distribution...")
    for name, model_tmpl in eval_models.items():
        if hasattr(model_tmpl, "model_class"):
            m = PyTorchModelWrapper(
                model_class=lambda dim: TabularTransformer(num_features=dim, d_model=64, nhead=4),
                input_dim=input_dim, epochs=tf_epochs, batch_size=512
            )
        else:
            m = clone(model_tmpl)

        m.fit(X_train_ddpm_sc, y_train_ddpm)
        probs = m.predict_proba(X_test_ddpm_sc)[:, 1]

        m_def = evaluate_predictions(data_normal_corrected["y_test_raw"], probs, threshold=0.50)
        m_def["Algorithm"] = name
        m_def["Scenario"] = "Diffusion (t=0.50)"
        diff_records.append(m_def)

        opt_t, _ = optimize_threshold(data_normal_corrected["y_test_raw"], probs)
        m_opt = evaluate_predictions(data_normal_corrected["y_test_raw"], probs, threshold=opt_t)
        m_opt["Algorithm"] = name
        m_opt["Scenario"] = "Diffusion (Opt. Thresh)"
        diff_records.append(m_opt)

    gan_df = pd.DataFrame(gan_records)
    diff_df = pd.DataFrame(diff_records)
    gan_vs_diff_df = format_gan_vs_diffusion(gan_df, diff_df)
    gan_vs_diff_path = os.path.join(metrics_dir, "GAN_vs_Diffusion.csv")
    gan_vs_diff_df.to_csv(gan_vs_diff_path, index=False)
    print(f"\n[✓] Saved Experiment 3 (4 results per model) to '{gan_vs_diff_path}':")
    print(gan_vs_diff_df.head(8).to_string(index=False))

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=gan_vs_diff_df, x="Algorithm", y="F1", hue="Scenario", palette="Spectral")
    plt.title("GAN vs Diffusion Performance on Corrected Distribution", fontsize=13, fontweight="bold")
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "chart_gan_vs_diffusion.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # EXPERIMENTS 4 & 5: DATASET SIZE SCALING (100k, 200k, 300k, 400k, 500k, 1M)
    # -------------------------------------------------------------------------
    if not args.skip_sizes:
        print("\n[*] Running Experiments 4 & 5: Dataset Size Scaling Benchmarks...")
        rf_model = models_dict.get("Random Forest")
        if rf_model is not None:
            rf_scaled = clone(rf_model).set_params(n_estimators=50, max_depth=15, n_jobs=-1)
        else:
            rf_scaled = None

        size_models = {
            "Deep MLP Classifier": get_deep_mlp(random_state=42),
            "Gradient Boosting": all_models.get("Hist Gradient Boosting") or models_dict.get("Gradient Boosting"),
            "Random Forest": rf_scaled,
            "Logistic Regression": models_dict.get("Logistic Regression")
        }
        size_models = {k: v for k, v in size_models.items() if v is not None}


        # 4. GAN Size Scaling
        gan_sizes = args.gan_sizes
        if gan_sizes is None:
            gan_sizes = [5000, 10000] if args.quick else [100000, 200000, 300000, 400000, 500000, 1000000]

        print(f"\n[*] Experiment 4: Benchmarking GAN across sizes: {gan_sizes}...")
        gan_size_fn = lambda total_samples: generate_ctgan_synthetic_data(
            X_train_raw=data_normal_corrected["X_train_raw"],
            y_train_raw=data_normal_corrected["y_train_raw"],
            categorical_features=data_normal_corrected["cat_cols"],
            total_samples=total_samples,
            cache_path="data/ctgan_synthetic_120000.parquet"
        )
        gan_size_df = benchmark_dataset_sizes(
            generator_fn=gan_size_fn,
            sizes=gan_sizes,
            models_dict=size_models,
            X_train_raw=data_normal_corrected["X_train_raw"],
            y_train_raw=data_normal_corrected["y_train_raw"],
            X_test_raw=data_normal_corrected["X_test_raw"],
            y_test_raw=data_normal_corrected["y_test_raw"],
            num_cols=data_normal_corrected["num_cols"],
            cat_cols=data_normal_corrected["cat_cols"],
            generator_name="GAN"
        )
        gan_size_path = os.path.join(metrics_dir, "Gan_size.csv")
        gan_size_df.to_csv(gan_size_path, index=False)
        print(f"\n[✓] Saved Experiment 4 (Gan_size.csv) to '{gan_size_path}':")
        print(gan_size_df.head(6).to_string(index=False))
        plot_size_scaling(gan_size_df, "CTGAN Dataset Size Scaling Effect on F1", os.path.join(charts_dir, "chart_gan_size_scaling.png"))

        # 5. Diffusion Size Scaling
        diffusion_sizes = args.diffusion_sizes
        if diffusion_sizes is None:
            diffusion_sizes = [2000, 5000] if args.quick else [100000, 200000, 300000, 400000, 500000, 1000000]

        print(f"\n[*] Experiment 5: Benchmarking Diffusion across sizes: {diffusion_sizes}...")
        diff_size_fn = lambda total_samples: generate_tabddpm_synthetic_data(
            X_train_raw=data_normal_corrected["X_train_raw"],
            y_train_raw=data_normal_corrected["y_train_raw"],
            total_samples=total_samples,
            cache_dir="data"
        )
        diff_size_df = benchmark_dataset_sizes(
            generator_fn=diff_size_fn,
            sizes=diffusion_sizes,
            models_dict=size_models,
            X_train_raw=data_normal_corrected["X_train_raw"],
            y_train_raw=data_normal_corrected["y_train_raw"],
            X_test_raw=data_normal_corrected["X_test_raw"],
            y_test_raw=data_normal_corrected["y_test_raw"],
            num_cols=data_normal_corrected["num_cols"],
            cat_cols=data_normal_corrected["cat_cols"],
            generator_name="Diffusion"
        )
        diff_size_path = os.path.join(metrics_dir, "Diffusion_size.csv")
        diff_size_df.to_csv(diff_size_path, index=False)
        print(f"\n[✓] Saved Experiment 5 (Diffusion_size.csv) to '{diff_size_path}':")
        print(diff_size_df.head(6).to_string(index=False))
        plot_size_scaling(diff_size_df, "TabDDPM Diffusion Dataset Size Scaling Effect on F1", os.path.join(charts_dir, "chart_diffusion_size_scaling.png"))


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
