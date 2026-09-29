"""
Train & Evaluation Pipeline (Standardized Research Benchmark)
-------------------------------------------------------------
Generates the standardized research CSV benchmark files in summary/metrics/:

  1. leaky_replication.csv    (Exact replication comparison vs Xu et al. 2024 Table 2)
  2. leaky_vs_corrected.csv   (4 results per model: Leaky vs Corrected x Normal vs Opt Threshold)
  3. GAN_vs_Diffusion.csv     (4 results per model: GAN vs Diffusion x Normal vs Opt Threshold)
  4. Gan_size.csv             (Dataset size scaling: 100k, 200k, 300k, 400k, 500k, 1M)
  5. Diffusion_size.csv       (Dataset size scaling on Diffusion model data)
  6. Diffusion_ratio.csv      (Default : non-default ratio of the synthetic rows)
  7. AGSS_size.csv            (Adaptive Generative Synthetic Sampling: size scaling)
  8. AGSS_ratio.csv           (AGSS: default : non-default ratio)
  +  synthetic_data_statistical_fidelity.csv

Usage:
    python train.py                     # Full research training & size evaluation
    python train.py --quick             # Fast smoke-test mode (~2 mins)
    python train.py --skip-sizes        # Run benchmarks 1, 2, 3 only
    python train.py --gan-sizes 100000 200000 500000 1000000
"""

import os
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone

from src.data import (
    load_credit_data, prepare_leaky_pipeline, prepare_corrected_pipeline, TARGET_COL
)
from src.models import get_classifiers, get_deep_mlp, TabularTransformer, PyTorchModelWrapper
from src.evaluation import (
    evaluate_predictions, optimize_threshold, generate_cv_threshold_plot,
    replicate_paper_table2, format_leaky_vs_corrected, format_gan_vs_diffusion,
    format_size_scaling_table, benchmark_dataset_sizes, benchmark_default_ratios,
    evaluate_augmented_training
)
from src.generator_gan import generate_ctgan_synthetic_data
from src.generator_diffusion import generate_tabddpm_synthetic_data, build_tabddpm_pool
from src.generator_agss import AGSSSampler, make_agss_generator
from src.visualizations import (
    plot_scenario_comparisons, plot_paper_replication_match, compute_synthetic_fidelity_metrics,
    plot_synthetic_feature_distributions, plot_synthetic_categorical_fidelity,
    plot_synthetic_correlation_fidelity, plot_synthetic_pca_manifold
)
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
                        help="List of Diffusion/AGSS dataset sizes (default: 0 100k 200k 300k 400k 500k 1000k)")
    parser.add_argument("--diffusion-ratios", nargs="+", type=float, default=None,
                        help="Default ratios of the Diffusion/AGSS synthetic rows (default: 0.1 0.22 0.3 0.5 0.7 0.9)")
    parser.add_argument("--ratio-samples", type=int, default=60000,
                        help="Synthetic rows added per ratio in the ratio experiment (default: 60,000)")
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
            model = clone(model)
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
    print(f"\n[✓] Saved Experiment 2 (2 rows per model) to '{leaky_vs_corr_path}':")
    print(leaky_vs_corr_df.head(6).to_string(index=False))

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

    # 3.B Build Diffusion Training Data
    if os.path.exists(args.ddpm_data):
        ddpm_syn = pd.read_parquet(args.ddpm_data) if args.ddpm_data.endswith(".parquet") else pd.read_csv(args.ddpm_data)
    else:
        # No pre-generated file: train/sample TabDDPM on the training partition only (3,000 + 3,000 rows)
        ddpm_syn = generate_tabddpm_synthetic_data(
            X_train_raw=data_normal_corrected["X_train_raw"],
            y_train_raw=data_normal_corrected["y_train_raw"],
            num_defaults=3000, num_non_defaults=3000,
            cache_dir="data"
        )

    if args.quick and len(ddpm_syn) > 2000:
        ddpm_syn = ddpm_syn.sample(n=2000, random_state=42)

    # Synthetic data fidelity (real = training partition) -> synthetic_data_statistical_fidelity.csv
    fidelity_df = compute_synthetic_fidelity_metrics(train_clean, ctgan_syn, ddpm_syn, list(train_clean.columns))
    fidelity_df.to_csv(os.path.join(metrics_dir, "synthetic_data_statistical_fidelity.csv"), index=False)
    features = [c for c in train_clean.columns if c != TARGET_COL]
    plot_synthetic_feature_distributions(train_clean, ctgan_syn, ddpm_syn, ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", "BILL_AMT2", "PAY_AMT2"],
                                         os.path.join(charts_dir, "synthetic_feature_distributions.png"))
    plot_synthetic_categorical_fidelity(train_clean, ctgan_syn, ddpm_syn, ["SEX", "EDUCATION", "PAY_0", TARGET_COL],
                                        os.path.join(charts_dir, "synthetic_categorical_fidelity.png"))
    plot_synthetic_correlation_fidelity(train_clean, ctgan_syn, ddpm_syn, features,
                                        os.path.join(charts_dir, "synthetic_correlation_fidelity.png"))
    plot_synthetic_pca_manifold(train_clean, ctgan_syn, ddpm_syn, features,
                                os.path.join(charts_dir, "synthetic_pca_manifold.png"))

    tf_epochs = 3 if args.quick else 15
    eval_models = {
        "Tabular Transformer (DL)": PyTorchModelWrapper(
            model_class=lambda dim: TabularTransformer(num_features=dim, d_model=64, nhead=4),
            input_dim=len(features), epochs=tf_epochs, batch_size=512
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

    split_args = dict(
        models_dict=eval_models,
        X_train_raw=data_normal_corrected["X_train_raw"],
        y_train_raw=data_normal_corrected["y_train_raw"],
        X_test_raw=data_normal_corrected["X_test_raw"],
        y_test_raw=data_normal_corrected["y_test_raw"],
        num_cols=data_normal_corrected["num_cols"]
    )
    print("\n  -> Evaluating models on GAN-augmented distribution...")
    gan_df = pd.DataFrame(evaluate_augmented_training(ctgan_syn, generator_name="GAN", **split_args))
    print("\n  -> Evaluating models on Diffusion-augmented distribution...")
    diff_df = pd.DataFrame(evaluate_augmented_training(ddpm_syn, generator_name="Diffusion", **split_args))

    # Plot GAN vs Diffusion comparison using combined unpivoted df
    combined_gan_diff = pd.concat([gan_df, diff_df], axis=0)
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=combined_gan_diff, x="Algorithm", y="F1", hue="Scenario", palette="Spectral")
    plt.title("GAN vs Diffusion Performance on Corrected Distribution", fontsize=13, fontweight="bold")
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "chart_gan_vs_diffusion.png"), dpi=300)
    plt.close()

    # Format into compact 2-row table and save
    gan_vs_diff_df = format_gan_vs_diffusion(gan_df, diff_df)
    gan_vs_diff_path = os.path.join(metrics_dir, "GAN_vs_Diffusion.csv")
    gan_vs_diff_df.to_csv(gan_vs_diff_path, index=False)
    print(f"\n[✓] Saved Experiment 3 (2 rows per model) to '{gan_vs_diff_path}':")
    print(gan_vs_diff_df.head(6).to_string(index=False))

    # -------------------------------------------------------------------------
    # EXPERIMENTS 4-8: DATASET SIZE SCALING AND DEFAULT RATIO (GAN, DIFFUSION, AGSS)
    # -------------------------------------------------------------------------
    if not args.skip_sizes:
        print("\n[*] Running Experiments 4-8: Dataset Size Scaling and Default Ratio Benchmarks...")
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
        gan_size_raw_df = benchmark_dataset_sizes(
            generator_fn=gan_size_fn,
            sizes=gan_sizes,
            models_dict=size_models,
            X_train_raw=data_normal_corrected["X_train_raw"],
            y_train_raw=data_normal_corrected["y_train_raw"],
            X_test_raw=data_normal_corrected["X_test_raw"],
            y_test_raw=data_normal_corrected["y_test_raw"],
            num_cols=data_normal_corrected["num_cols"],
            generator_name="GAN"
        )
        plot_size_scaling(gan_size_raw_df, "CTGAN Dataset Size Scaling Effect on F1", os.path.join(charts_dir, "chart_gan_size_scaling.png"))
        gan_size_df = format_size_scaling_table(gan_size_raw_df)
        gan_size_path = os.path.join(metrics_dir, "Gan_size.csv")
        gan_size_df.to_csv(gan_size_path, index=False)
        print(f"\n[✓] Saved Experiment 4 (Gan_size.csv - 2 rows per model) to '{gan_size_path}':")
        print(gan_size_df.head(6).to_string(index=False))

        # 5-8. Diffusion and AGSS: dataset size scaling + default (risk : no-risk) ratio test.
        # Both sample, without replacement, from one pool of fresh TabDDPM rows.
        diffusion_sizes = args.diffusion_sizes
        if diffusion_sizes is None:
            diffusion_sizes = [0, 2000, 5000] if args.quick else [0, 100000, 200000, 300000, 400000, 500000, 1000000]
        diffusion_ratios = args.diffusion_ratios or [0.1, 0.22, 0.3, 0.5, 0.7, 0.9]
        ratio_samples = 5000 if args.quick else args.ratio_samples
        rows_per_class = max(max(diffusion_sizes) // 2, int(ratio_samples * max(diffusion_ratios)),
                             int(ratio_samples * (1 - min(diffusion_ratios))))

        ddpm_pool = build_tabddpm_pool(
            data_normal_corrected["X_train_raw"], data_normal_corrected["y_train_raw"],
            rows_per_class=rows_per_class, cache_dir="data"
        )
        diff_fn = lambda total_samples, default_ratio=None: generate_tabddpm_synthetic_data(
            X_train_raw=data_normal_corrected["X_train_raw"],
            y_train_raw=data_normal_corrected["y_train_raw"],
            total_samples=total_samples,
            default_ratio=default_ratio,
            cache_dir="data"
        )
        agss_fn = make_agss_generator(AGSSSampler(
            data_normal_corrected["X_train_raw"], data_normal_corrected["y_train_raw"], ddpm_pool
        ))
        split_args = dict(
            models_dict=size_models,
            X_train_raw=data_normal_corrected["X_train_raw"],
            y_train_raw=data_normal_corrected["y_train_raw"],
            X_test_raw=data_normal_corrected["X_test_raw"],
            y_test_raw=data_normal_corrected["y_test_raw"],
            num_cols=data_normal_corrected["num_cols"]
        )

        for exp_no, (gen_name, gen_fn) in zip([5, 7], [("Diffusion", diff_fn), ("AGSS", agss_fn)]):
            print(f"\n[*] Experiment {exp_no}: Benchmarking {gen_name} across sizes: {diffusion_sizes}...")
            size_raw_df = benchmark_dataset_sizes(gen_fn, diffusion_sizes, generator_name=gen_name, **split_args)
            plot_size_scaling(size_raw_df, f"{gen_name} Dataset Size Scaling Effect on F1",
                              os.path.join(charts_dir, f"chart_{gen_name.lower()}_size_scaling.png"))
            size_path = os.path.join(metrics_dir, f"{gen_name}_size.csv")
            format_size_scaling_table(size_raw_df).to_csv(size_path, index=False)
            print(f"[✓] Saved Experiment {exp_no} to '{size_path}'")

            print(f"\n[*] Experiment {exp_no + 1}: Benchmarking {gen_name} default ratios {diffusion_ratios} at {ratio_samples:,} rows...")
            ratio_raw_df = benchmark_default_ratios(gen_fn, diffusion_ratios, ratio_samples, generator_name=gen_name, **split_args)
            ratio_path = os.path.join(metrics_dir, f"{gen_name}_ratio.csv")
            format_size_scaling_table(ratio_raw_df, key_col="Default_Ratio").to_csv(ratio_path, index=False)
            print(f"[✓] Saved Experiment {exp_no + 1} to '{ratio_path}'")


    # MODEL EXPLAINABILITY (XAI)
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
    print("\n===============================================================================")
    print(f"  TRAINING & EVALUATION COMPLETE (Elapsed: {elapsed/60:.2f} mins)               ")
    print(f"  Summary directory generated at: '{summary_dir}/'                              ")
    print("===============================================================================")


if __name__ == "__main__":
    main()
