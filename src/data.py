"""
Data loading, feature specification, and data partitioning pipelines (Leaky vs Corrected).
"""

import os
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler, SMOTENC

TARGET_COL = "default.payment.next.month"
CATEGORICAL_FEATURES = [
    "SEX", "EDUCATION", "MARRIAGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
]


def load_credit_data(filepath: str = "UCI_Credit_Card.csv") -> pd.DataFrame:
    """
    Loads the UCI Credit Card Default dataset from local path or KaggleHub fallback.
    """
    if os.path.exists(filepath):
        print(f"[*] Loading local dataset from '{filepath}'...")
        df = pd.read_csv(filepath)
    else:
        print("[*] Local dataset not found. Downloading via kagglehub...")
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "uciml/default-of-credit-card-clients-dataset",
            "UCI_Credit_Card.csv"
        )

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    print(f"[*] Loaded dataset successfully. Shape: {df.shape}")
    return df


def get_feature_lists(X: pd.DataFrame) -> Tuple[List[str], List[str], List[int]]:
    """
    Returns categorical feature names, numerical feature names, and categorical column indices.
    """
    cat_cols = [col for col in CATEGORICAL_FEATURES if col in X.columns]
    num_cols = [col for col in X.columns if col not in cat_cols]
    cat_indices = [X.columns.get_loc(col) for col in cat_cols]
    return cat_cols, num_cols, cat_indices


def prepare_leaky_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Replicates the flawed pipeline from Xu et al. (2024):
    1. Global feature scaling on the entire dataset
    2. Random Oversampling on the entire dataset
    3. Train/test split after oversampling (leaking test rows from train)
    """
    print("[*] Building LEAKY pipeline (replicating Xu et al., 2024 methodology)...")
    scaler_global = StandardScaler()
    X_scaled_global = pd.DataFrame(scaler_global.fit_transform(X), columns=X.columns)

    ros = RandomOverSampler(random_state=random_state)
    X_res_leaky, y_res_leaky = ros.fit_resample(X_scaled_global, y)

    X_train_leaky, X_test_leaky, y_train_leaky, y_test_leaky = train_test_split(
        X_res_leaky, y_res_leaky, test_size=test_size, random_state=random_state, shuffle=True
    )

    return {
        "X_train": X_train_leaky,
        "X_test": X_test_leaky,
        "y_train": y_train_leaky,
        "y_test": y_test_leaky,
        "scaler": scaler_global
    }


def prepare_corrected_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Builds the methodologically sound, zero-leakage pipeline:
    1. Stratified train/test split on raw data first (75/25)
    2. SMOTENC applied strictly to the training split
    3. ColumnTransformer (StandardScaler on continuous, passthrough on categorical)
       fitted strictly on the training partition
    """
    print("[*] Building CORRECTED pipeline (honest zero-leakage methodology)...")
    cat_cols, num_cols, cat_indices = get_feature_lists(X)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y
    )

    print(f"[*] Applying SMOTENC strictly to training partition ({len(X_train_raw)} samples)...")
    smote_nc = SMOTENC(categorical_features=cat_indices, random_state=random_state)
    X_train_res, y_train_res = smote_nc.fit_resample(X_train_raw, y_train_raw)

    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), num_cols)],
        remainder='passthrough'
    )
    X_train_corr_sc = preprocessor.fit_transform(X_train_res)
    X_test_corr_sc = preprocessor.transform(X_test_raw)
    X_train_raw_sc = preprocessor.transform(X_train_raw)

    feature_names_transformed = [f"num__{c}" for c in num_cols] + cat_cols

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train_raw": y_train_raw,
        "y_test_raw": y_test_raw,
        "X_train_res": X_train_res,
        "y_train_res": y_train_res,
        "X_train_sc": X_train_corr_sc,
        "X_test_sc": X_test_corr_sc,
        "X_train_raw_sc": X_train_raw_sc,
        "preprocessor": preprocessor,
        "feature_names": feature_names_transformed,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_indices": cat_indices
    }


# =====================================================================
# NAIVE NOTATION ALIASES & HELPERS FOR EVALUATORS
# =====================================================================

def prepare_data_normal_leaky(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25, random_state: int = 42) -> Dict[str, Any]:
    """Naive alias: Flawed oversampling before split (replicates paper leakage)."""
    return prepare_leaky_pipeline(X, y, test_size=test_size, random_state=random_state)


def prepare_data_normal_corrected(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25, random_state: int = 42) -> Dict[str, Any]:
    """Naive alias: Honest oversampling strictly on training split (no leakage)."""
    return prepare_corrected_pipeline(X, y, test_size=test_size, random_state=random_state)
