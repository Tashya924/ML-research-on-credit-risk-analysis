"""
Model Explainability (XAI) Suite: SHAP global & local waterfall, LIME,
Permutation Feature Importance, and customer credit risk scoring.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False


def generate_shap_analysis(
    model: Any,
    X_test: np.ndarray,
    feature_names: List[str],
    output_dir: str = "summary/xai",
    max_samples: int = 500
):
    """
    Generates SHAP summary beeswarm plot, key feature dependence plots, and local waterfall plots.
    """
    if not HAS_SHAP:
        print("[!] SHAP not installed. Skipping SHAP analysis.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Generating SHAP explainability suite (sampling {max_samples} test records)...")

    # Sample test set for efficient computation
    n_sample = min(max_samples, len(X_test))
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), n_sample, replace=False)
    X_sample = X_test[sample_indices]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Handle shape differences across SHAP versions (binary classification)
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]
    elif len(getattr(shap_values, 'shape', [])) == 3:
        shap_pos = shap_values[:, :, 1]
    else:
        shap_pos = shap_values

    # 1. Global Beeswarm Summary Plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_pos, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Global Feature Importance (Honest Model)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "shap_summary_beeswarm.png")
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"[*] Saved SHAP summary beeswarm to '{summary_path}'")

    # 2. Key Dependence Plots
    for target_feat in ["PAY_0", "LIMIT_BAL", "BILL_AMT1"]:
        feat_idx = None
        for i, fname in enumerate(feature_names):
            if target_feat in fname:
                feat_idx = i
                break
        if feat_idx is not None:
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(
                feat_idx, shap_pos, X_sample,
                feature_names=feature_names,
                interaction_index="auto",
                show=False
            )
            plt.title(f"SHAP Dependence Plot: {target_feat}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            dep_path = os.path.join(output_dir, f"shap_dependence_{target_feat}.png")
            plt.savefig(dep_path, dpi=300)
            plt.close()

    # 3. Waterfall attributions for individual applicants
    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > 1:
        base_val = expected_val[1]
    else:
        base_val = float(expected_val)

    for applicant_idx in [0, 1, 2]:
        if applicant_idx < len(X_sample):
            exp = shap.Explanation(
                values=shap_pos[applicant_idx],
                base_values=base_val,
                data=X_sample[applicant_idx],
                feature_names=feature_names
            )
            plt.figure(figsize=(9, 5))
            shap.waterfall_plot(exp, show=False)
            plt.title(f"SHAP Waterfall Attribution: Applicant #{applicant_idx}", fontsize=11, fontweight='bold')
            plt.tight_layout()
            wf_path = os.path.join(output_dir, f"shap_waterfall_applicant_{applicant_idx}.png")
            plt.savefig(wf_path, dpi=300)
            plt.close()


def generate_lime_analysis(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    output_dir: str = "summary/xai",
    applicant_indices: List[int] = [0, 1, 2]
):
    """
    Generates LIME local explanations for sample applicants.
    """
    if not HAS_LIME:
        print("[!] LIME not installed. Skipping LIME analysis.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print("[*] Generating LIME local explanations...")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=['Non-Default', 'Default'],
        mode='classification',
        random_state=42
    )

    for idx in applicant_indices:
        if idx >= len(X_test):
            continue
        exp = explainer.explain_instance(
            data_row=X_test[idx],
            predict_fn=model.predict_proba,
            num_features=8
        )
        fig = exp.as_pyplot_figure()
        plt.title(f"LIME Local Attribution: Applicant #{idx}", fontsize=11, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"lime_applicant_{idx}.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[*] Saved LIME explanation to '{save_path}'")


def generate_permutation_importance_plot(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    output_filepath: str = "summary/xai/permutation_importance.png",
    n_repeats: int = 10,
    random_state: int = 42
):
    """
    Computes test F1 degradation via model-agnostic permutation feature importance.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    print(f"[*] Computing Permutation Feature Importance ({n_repeats} repeats)...")

    all_results = []
    for model_name, model in models_dict.items():
        res = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="f1"
        )
        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance_Mean": res.importances_mean,
            "Importance_Std": res.importances_std,
            "Model": model_name
        }).sort_values(by="Importance_Mean", ascending=False)
        all_results.append(df)

    comb_df = pd.concat(all_results, axis=0)

    # Top features across all evaluated models
    top_features = (
        comb_df.groupby("Feature")["Importance_Mean"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .index
    )
    plot_df = comb_df[comb_df["Feature"].isin(top_features)]

    plt.figure(figsize=(11, 8))
    sns.set_theme(style="whitegrid")
    sns.barplot(
        data=plot_df,
        y="Feature",
        x="Importance_Mean",
        hue="Model",
        order=top_features,
        palette="viridis"
    )
    plt.title("Permutation Feature Importance (Test F1 Score Degradation)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Mean Test F1 Degradation Upon Feature Permutation", fontsize=11)
    plt.ylabel("Tabular Feature", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(output_filepath, dpi=300)
    plt.close()
    print(f"[*] Saved permutation feature importance to '{output_filepath}'")


def score_applicant_risk(
    applicant_data: Dict[str, Any],
    model: Any,
    preprocessor: Any,
    feature_names: List[str],
    decision_threshold: float = 0.48
) -> Dict[str, Any]:
    """
    Scores an applicant record: returns predicted probability, binary decision,
    calibrated risk tier, and top contributing risk factors.
    """
    raw_df = pd.DataFrame([applicant_data])
    scaled_features = preprocessor.transform(raw_df)

    prob = float(model.predict_proba(scaled_features)[0, 1])
    decision = int(prob >= decision_threshold)

    if prob >= 0.50:
        tier = "HIGH RISK"
    elif prob >= 0.30:
        tier = "MODERATE RISK"
    else:
        tier = "LOW RISK"

    # Extract feature impacts using TreeExplainer if model is tree-based
    drivers = []
    if HAS_SHAP and hasattr(model, "estimators_"):
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(scaled_features)
            if isinstance(sv, list):
                sv_pos = sv[1][0]
            elif len(getattr(sv, 'shape', [])) == 3:
                sv_pos = sv[0, :, 1]
            else:
                sv_pos = sv[0]
            driver_df = pd.DataFrame({"Feature": feature_names, "Impact": sv_pos})
            top_drivers = driver_df.sort_values(by="Impact", key=abs, ascending=False).head(5)
            drivers = top_drivers.to_dict(orient="records")
        except Exception:
            pass

    return {
        "default_probability": round(prob, 4),
        "prediction": decision,
        "risk_tier": tier,
        "decision_threshold": decision_threshold,
        "top_risk_drivers": drivers
    }
