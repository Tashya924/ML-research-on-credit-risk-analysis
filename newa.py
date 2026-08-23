import os
import pandas as pd
import numpy as np
import kagglehub
import joblib
import json
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              AdaBoostClassifier, GradientBoostingClassifier, 
                              HistGradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

def custom_threshold_plot(model, X, y, title, filename):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    thresholds = np.linspace(0.01, 0.99, 50)
    
    metrics = {'precision': [], 'recall': [], 'f1': [], 'queue_rate': []}
    
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    for train_idx, test_idx in cv.split(X_arr, y_arr):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        
        fold_metrics = {'precision': [], 'recall': [], 'f1': [], 'queue_rate': []}
        for t in thresholds:
            preds = (probs >= t).astype(int)
            fold_metrics['precision'].append(precision_score(y_te, preds, zero_division=0))
            fold_metrics['recall'].append(recall_score(y_te, preds, zero_division=0))
            fold_metrics['f1'].append(f1_score(y_te, preds, zero_division=0))
            fold_metrics['queue_rate'].append(np.mean(preds))
            
        for k in metrics.keys():
            metrics[k].append(fold_metrics[k])
            
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    colors = {'precision': '#1f77b4', 'recall': '#2ca02c', 'f1': '#d62728', 'queue_rate': '#9467bd'}
    
    best_t = 0.5
    best_f1 = 0
    
    for metric_name, color in colors.items():
        arr = np.array(metrics[metric_name])
        mean_val = np.mean(arr, axis=0)
        std_val = np.std(arr, axis=0)
        
        plt.plot(thresholds, mean_val, label=metric_name, color=color, linewidth=2)
        plt.fill_between(thresholds, np.clip(mean_val - std_val, 0, 1), np.clip(mean_val + std_val, 0, 1), color=color, alpha=0.2)
        
        if metric_name == 'f1':
            max_idx = np.argmax(mean_val)
            best_t = thresholds[max_idx]
            best_f1 = mean_val[max_idx]

    plt.axvline(best_t, color='black', linestyle='--', label=f'Optimal t={best_t:.2f}')
    plt.title(title)
    plt.xlabel('Discrimination Threshold')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def generate_custom_plots(X_train_raw, y_train_raw, X_train_corr, y_train_corr, models_dict):
    leaky_dir = "leaky_threshold_plots"
    fixed_dir = "fixed_threshold_plots"
    os.makedirs(leaky_dir, exist_ok=True)
    os.makedirs(fixed_dir, exist_ok=True)
    
    datasets = {
        "leaky": (X_train_raw, y_train_raw, leaky_dir),       
        "fixed": (X_train_corr, y_train_corr, fixed_dir)  
    }
    
    for scenario, (X_data, y_data, folder_path) in datasets.items():
        for name, model_instance in models_dict.items():
            model = clone(model_instance)
            
            clean_name = name.replace(" ", "_")
            filename = os.path.join(folder_path, f"{scenario}_{clean_name}_threshold_plot.png")
            title = f"Threshold Plot for {name} ({scenario.capitalize()})"
            
            custom_threshold_plot(model, X_data, y_data, title, filename)

def main():
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "uciml/default-of-credit-card-clients-dataset",
        "UCI_Credit_Card.csv"
    )

    target = "default.payment.next.month"
    X = df.drop(columns=["ID", target])
    y = df[target]

    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    numerical_features = [col for col in X.columns if col not in categorical_features]
    cat_indices = [X.columns.get_loc(col) for col in categorical_features]

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.25, random_state=42, shuffle=True
    )

    scaler_global = StandardScaler()
    X_scaled_global = pd.DataFrame(scaler_global.fit_transform(X), columns=X.columns)
    
    ros = RandomOverSampler(random_state=42)
    X_res_leaky, y_res_leaky = ros.fit_resample(X_scaled_global, y)
    X_train_leaky, X_test_leaky, y_train_leaky, y_test_leaky = train_test_split(
        X_res_leaky, y_res_leaky, test_size=0.25, random_state=42, shuffle=True
    )
    
    X_train_raw_global_sc = pd.DataFrame(scaler_global.transform(X_train_raw), columns=X.columns)

    smote_nc = SMOTENC(categorical_features=cat_indices, random_state=42)
    X_train_res, y_train_res = smote_nc.fit_resample(X_train_raw, y_train_raw)

    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numerical_features)],
        remainder='passthrough'
    )
    X_train_corr_sc = preprocessor.fit_transform(X_train_res)
    X_test_corr_sc = preprocessor.transform(X_test_raw)
    
    X_train_raw_corr_sc = preprocessor.transform(X_train_raw)

    scenarios = {
        "Leaky": (X_train_leaky, X_test_leaky, y_train_leaky, y_test_leaky, X_train_raw_global_sc),
        "Corrected": (X_train_corr_sc, X_test_corr_sc, y_train_res, y_test_raw, X_train_raw_corr_sc)
    }

    models_dict = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Ridge Classifier": CalibratedClassifierCV(RidgeClassifier(random_state=42)),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Extra Tree": ExtraTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Extra Trees": ExtraTreesClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "MLP Classifier": MLPClassifier(random_state=42, max_iter=1000),
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }

    all_results = []
    
    for scenario, (X_tr, X_te, ytr, yte, X_train_raw_sc) in scenarios.items():
        for name, model in models_dict.items():
            model.fit(X_tr, ytr)
            
            clean_name = name.replace(" ", "_")
            clean_scenario = scenario.lower().replace(" ", "_")
            model_dir = f"models/{clean_scenario}/{clean_name}"
            os.makedirs(f"{model_dir}/config_files", exist_ok=True)
            os.makedirs(f"{model_dir}/weights", exist_ok=True)
            
            try:
                params = model.get_params()
                safe_params = {k: str(v) for k, v in params.items()}
                with open(f"{model_dir}/config_files/config.json", "w") as f:
                    json.dump(safe_params, f, indent=4)
            except Exception:
                pass
                
            joblib.dump(model, f"{model_dir}/weights/model.joblib")

            probs_test = model.predict_proba(X_te)[:, 1]

            preds_default = (probs_test >= 0.5).astype(int)
            default_f1 = f1_score(yte, preds_default, zero_division=0)
            
            all_results.append({
                "Algorithm": name, 
                "Scenario": scenario,
                "Accuracy": accuracy_score(yte, preds_default),
                "Recall": recall_score(yte, preds_default, zero_division=0),
                "Precision": precision_score(yte, preds_default, zero_division=0), 
                "F1": default_f1, 
                "ROC_AUC": roc_auc_score(yte, probs_test),
                "Threshold": 0.50
            })

            best_thresh = 0.5
            best_f1 = default_f1
            
            for t in np.linspace(0.1, 0.9, 81):
                t_preds = (probs_test >= t).astype(int)
                t_f1 = f1_score(yte, t_preds, zero_division=0)
                if t_f1 > best_f1:
                    best_f1 = t_f1
                    best_thresh = t
                    
            opt_preds_test = (probs_test >= best_thresh).astype(int)
            all_results.append({
                "Algorithm": name, 
                "Scenario": f"{scenario} (Opt. Thresh)",
                "Accuracy": accuracy_score(yte, opt_preds_test),
                "Recall": recall_score(yte, opt_preds_test, zero_division=0),
                "Precision": precision_score(yte, opt_preds_test, zero_division=0), 
                "F1": best_f1, 
                "ROC_AUC": roc_auc_score(yte, probs_test),
                "Threshold": best_thresh
            })

    res_df = pd.DataFrame(all_results)
    
    # Save the dataframe to a text file
    with open("results_summary.txt", "w") as text_file:
        text_file.write(res_df.sort_values(by=["Algorithm", "Scenario"]).to_string(index=False))
    
    sns.set_theme(style="whitegrid")
    metrics = [("ROC_AUC", "chart_1_roc_auc.png"), 
               ("F1", "chart_2_f1_score.png"), 
               ("Precision", "chart_3_precision.png"), 
               ("Recall", "chart_4_recall.png")]
               
    for metric, filename in metrics:
        plt.figure(figsize=(14, 12)) 
        sns.barplot(data=res_df, x=metric, y="Algorithm", hue="Scenario")
        plt.xlim(0, 1.0)
        plt.title(f"{metric} Comparison across all Scenarios and Thresholds")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    return X_train_raw_global_sc, y_train_raw, X_train_corr_sc, y_train_res, models_dict

if __name__ == "__main__":
    X_raw, y_raw, X_corr, y_corr, models = main()
    generate_custom_plots(X_raw, y_raw, X_corr, y_corr, models)