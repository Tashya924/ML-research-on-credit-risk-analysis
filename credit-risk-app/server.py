"""
Credit Risk Analysis — Flask Inference Server
Loads all 15 trained models and serves predictions + per-input feature importance.
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify
from flask_cors import CORS

# Suppress sklearn version warnings (models trained on 1.6.1, running on 1.7.0)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────
# Feature Configuration
# ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

# Human-readable feature descriptions
FEATURE_DESCRIPTIONS = {
    "LIMIT_BAL": "Credit Limit",
    "SEX": "Gender",
    "EDUCATION": "Education Level",
    "MARRIAGE": "Marital Status",
    "AGE": "Age",
    "PAY_0": "Repayment Status (Sep)",
    "PAY_2": "Repayment Status (Aug)",
    "PAY_3": "Repayment Status (Jul)",
    "PAY_4": "Repayment Status (Jun)",
    "PAY_5": "Repayment Status (May)",
    "PAY_6": "Repayment Status (Apr)",
    "BILL_AMT1": "Bill Amount (Sep)",
    "BILL_AMT2": "Bill Amount (Aug)",
    "BILL_AMT3": "Bill Amount (Jul)",
    "BILL_AMT4": "Bill Amount (Jun)",
    "BILL_AMT5": "Bill Amount (May)",
    "BILL_AMT6": "Bill Amount (Apr)",
    "PAY_AMT1": "Payment Amount (Sep)",
    "PAY_AMT2": "Payment Amount (Aug)",
    "PAY_AMT3": "Payment Amount (Jul)",
    "PAY_AMT4": "Payment Amount (Jun)",
    "PAY_AMT5": "Payment Amount (May)",
    "PAY_AMT6": "Payment Amount (Apr)",
}

# ─────────────────────────────────────────────────────────────
# Model Registry — maps display name → folder name
# ─────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Random Forest": "random_forest",
    "Naive Bayes": "naive_bayes",
    "LGBM": "lgbm",
    "AdaBoost": "adaboost",
    "Gradient Boost": "gradient_boost",
    "LDA": "lda",
    "MLP": "mlp",
    "SVM": "svm",
    "XGBoost": "xgboost",
    "CatBoost": "catboost",
    "Extra Trees": "extra_trees",
    "SGD Classifier": "sgd_classifier",
}

# Which models support SHAP TreeExplainer
TREE_MODEL_NAMES = {
    "Decision Tree", "Random Forest", "Gradient Boost", "AdaBoost",
    "Extra Trees", "XGBoost", "LGBM", "CatBoost",
}

# Which models have linear coefficients
LINEAR_MODEL_NAMES = {"Logistic Regression", "SGD Classifier", "LDA"}

# ─────────────────────────────────────────────────────────────
# Load Models at Startup
# ─────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
loaded_models = {}
shap_explainers = {}

print("Loading models...")
for display_name, folder_name in MODEL_REGISTRY.items():
    model_path = os.path.join(MODELS_DIR, folder_name, "weights", "model.joblib")
    try:
        loaded_models[display_name] = joblib.load(model_path)
        print(f"  ✓ {display_name}")
    except Exception as e:
        print(f"  ✗ {display_name}: {e}")

print(f"\nLoaded {len(loaded_models)}/{len(MODEL_REGISTRY)} models.")

# ─────────────────────────────────────────────────────────────
# Create SHAP TreeExplainers for tree-based models
# ─────────────────────────────────────────────────────────────
print("\nInitializing SHAP explainers for tree-based models...")
for name in TREE_MODEL_NAMES:
    if name in loaded_models:
        try:
            shap_explainers[name] = shap.TreeExplainer(loaded_models[name])
            print(f"  ✓ SHAP: {name}")
        except Exception as e:
            print(f"  ✗ SHAP: {name}: {e}")

print(f"Initialized {len(shap_explainers)} SHAP explainers.\n")


def get_per_input_importance(model, model_name, input_df):
    """
    Compute PER-INPUT feature importance that changes with each prediction.

    Methods by model type:
      - Tree models: SHAP TreeExplainer (per-input SHAP values)
      - Linear models: |coef × input_value| (exact linear contribution)
      - MLP: |first-layer-weights × input_value|
      - Naive Bayes: Per-feature log-likelihood difference between classes
      - KNN/SVM: Falls back to model-level static importance
    """
    importance = None
    input_values = input_df.iloc[0].values.astype(float)

    # ── Method 1: SHAP for tree-based models (per-input) ──
    if model_name in shap_explainers:
        try:
            explainer = shap_explainers[model_name]
            shap_values = explainer.shap_values(input_df)

            # Handle different SHAP output formats for binary classification
            if isinstance(shap_values, list):
                # Older SHAP: returns [class_0_values, class_1_values]
                values = np.array(shap_values[1]).flatten()
            elif hasattr(shap_values, "values"):
                vals = shap_values.values
                if len(vals.shape) == 3:
                    values = vals[0, :, 1]
                else:
                    values = vals.flatten()
            else:
                if len(shap_values.shape) == 3:
                    values = shap_values[0, :, 1]
                elif len(shap_values.shape) == 2:
                    values = shap_values[0]
                else:
                    values = shap_values

            importance = np.abs(values).astype(float)
        except Exception as e:
            print(f"  SHAP failed for {model_name}: {e}")

    # ── Method 2: Linear models — coef × input_value (per-input) ──
    if importance is None and hasattr(model, "coef_"):
        coef = np.array(model.coef_).flatten()
        importance = np.abs(coef * input_values)

    # ── Method 3: MLP — first layer weights × input_value (per-input) ──
    if importance is None and hasattr(model, "coefs_") and len(model.coefs_) > 0:
        weight_magnitude = np.abs(model.coefs_[0]).sum(axis=1)
        importance = weight_magnitude * np.abs(input_values)

    # ── Method 4: Naive Bayes — per-feature log-likelihood (per-input) ──
    if importance is None and hasattr(model, "theta_") and hasattr(model, "var_"):
        # Compute how much each feature contributes to class 1 vs class 0
        # log P(x_i | class=1) - log P(x_i | class=0)
        theta = model.theta_  # (n_classes, n_features) — class means
        var = model.var_      # (n_classes, n_features) — class variances
        log_prob_diff = np.zeros(len(FEATURE_NAMES))
        for i in range(len(FEATURE_NAMES)):
            x = input_values[i]
            # log-likelihood under class 1
            ll_1 = -0.5 * np.log(2 * np.pi * var[1, i]) - (x - theta[1, i])**2 / (2 * var[1, i])
            # log-likelihood under class 0
            ll_0 = -0.5 * np.log(2 * np.pi * var[0, i]) - (x - theta[0, i])**2 / (2 * var[0, i])
            log_prob_diff[i] = ll_1 - ll_0
        importance = np.abs(log_prob_diff)

    # ── Method 5: Fallback — model-level static importance ──
    if importance is None:
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_.copy()
        else:
            return None

    # Normalize to sum to 1
    total = importance.sum()
    if total > 0:
        importance = importance / total

    return {name: float(val) for name, val in zip(FEATURE_NAMES, importance)}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": len(loaded_models),
        "shap_explainers": len(shap_explainers),
        "feature_count": len(FEATURE_NAMES),
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Validate all required features are present
        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Build input DataFrame in correct feature order
        input_dict = {f: [float(data[f])] for f in FEATURE_NAMES}
        input_df = pd.DataFrame(input_dict)

        # Run inference on all loaded models
        results = []
        all_importances = []

        for model_name, model in loaded_models.items():
            try:
                pred_class = int(model.predict(input_df)[0])

                # Get probability (handle models without predict_proba)
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(input_df)[0]
                    prob_default = float(prob[1])
                    prob_no_default = float(prob[0])
                elif hasattr(model, "decision_function"):
                    decision = model.decision_function(input_df)[0]
                    prob_default = float(1 / (1 + np.exp(-decision)))
                    prob_no_default = 1.0 - prob_default
                else:
                    prob_default = float(pred_class)
                    prob_no_default = 1.0 - prob_default

                # Risk flag
                if prob_default >= 0.5:
                    risk_flag = "HIGH"
                elif prob_default >= 0.3:
                    risk_flag = "MODERATE"
                else:
                    risk_flag = "LOW"

                # Per-input feature importance
                importance = get_per_input_importance(model, model_name, input_df)

                model_result = {
                    "model_name": model_name,
                    "prediction": pred_class,
                    "prediction_label": "DEFAULT" if pred_class == 1 else "NO DEFAULT",
                    "probability_default": round(prob_default, 4),
                    "probability_no_default": round(prob_no_default, 4),
                    "risk_flag": risk_flag,
                    "feature_importance": importance,
                }
                results.append(model_result)

                if importance:
                    all_importances.append(importance)

            except Exception as e:
                results.append({
                    "model_name": model_name,
                    "error": str(e),
                })

        # ── Aggregate worst parameters across all models ──
        aggregated_importance = {}
        if all_importances:
            for feat in FEATURE_NAMES:
                values = [imp[feat] for imp in all_importances if feat in imp]
                if values:
                    aggregated_importance[feat] = round(float(np.mean(values)), 6)

        # Sort by worst (highest importance)
        sorted_features = sorted(
            aggregated_importance.items(), key=lambda x: x[1], reverse=True
        )

        worst_parameters = [
            {
                "feature": feat,
                "description": FEATURE_DESCRIPTIONS.get(feat, feat),
                "importance": imp,
                "input_value": float(data[feat]),
            }
            for feat, imp in sorted_features
        ]

        # ── Summary statistics ──
        valid_results = [r for r in results if "error" not in r]
        avg_default_prob = (
            float(np.mean([r["probability_default"] for r in valid_results]))
            if valid_results
            else 0
        )
        default_count = sum(1 for r in valid_results if r["prediction"] == 1)

        if avg_default_prob >= 0.5:
            consensus = "HIGH"
        elif avg_default_prob >= 0.3:
            consensus = "MODERATE"
        else:
            consensus = "LOW"

        return jsonify({
            "summary": {
                "total_models": len(valid_results),
                "models_predicting_default": default_count,
                "models_predicting_no_default": len(valid_results) - default_count,
                "average_default_probability": round(avg_default_prob, 4),
                "consensus_risk": consensus,
            },
            "model_results": results,
            "worst_parameters": worst_parameters,
            "feature_descriptions": FEATURE_DESCRIPTIONS,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Credit Risk Analysis API on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)
