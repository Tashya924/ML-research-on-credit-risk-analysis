"""
Dedicated GAN (CTGAN) Tabular Data Generator.
Provides zero-leakage training on training partition or leaky full-dataset training,
with user-selectable default / non-default ratios and domain bounds clipping.
"""

import os
from typing import Tuple, Optional, Dict, List
import numpy as np
import pandas as pd

TARGET_COL = "default.payment.next.month"


def clip_to_bounds(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Clips features strictly within the observed training bounds and rounds integer columns.
    """
    out_df = df.copy()
    for col, (b_min, b_max) in bounds.items():
        if col in out_df.columns:
            out_df[col] = out_df[col].round().clip(lower=b_min, upper=b_max)
    return out_df


def generate_ctgan_synthetic_data(
    X_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    categorical_features: list,
    num_defaults: Optional[int] = None,
    num_non_defaults: Optional[int] = None,
    total_samples: int = 120000,
    default_ratio: Optional[float] = None,
    epochs: int = 20,
    batch_size: int = 500,
    cache_path: Optional[str] = "data/ctgan_synthetic_120000.parquet",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic tabular data using CTGAN with user-specified class distribution.
    
    Args:
        X_train_raw: Training features DataFrame
        y_train_raw: Training target Series
        categorical_features: List of categorical feature column names
        num_defaults: Target count of default (target=1) instances
        num_non_defaults: Target count of non-default (target=0) instances
        total_samples: Total rows to generate if counts not specified
        default_ratio: Target proportion of defaults (e.g. 0.5 for 50:50 balance)
        epochs: Training epochs for CTGAN
        batch_size: Batch size for CTGAN
        cache_path: Path to pre-generated parquet/csv cache
        random_state: Random seed for reproducibility
    """
    if default_ratio is not None:
        num_defaults = int(total_samples * default_ratio)
        num_non_defaults = total_samples - num_defaults
    elif num_defaults is not None and num_non_defaults is not None:
        total_samples = num_defaults + num_non_defaults

    # Check if cache is available and covers our needs
    if cache_path and os.path.exists(cache_path):
        print(f"[*] Loading cached CTGAN dataset from '{cache_path}'...")
        if cache_path.endswith(".parquet"):
            cached_df = pd.read_parquet(cache_path)
        else:
            cached_df = pd.read_csv(cache_path)

        if num_defaults is not None and num_non_defaults is not None:
            c1 = cached_df[cached_df[TARGET_COL] == 1]
            c0 = cached_df[cached_df[TARGET_COL] == 0]
            if len(c1) >= num_defaults and len(c0) >= num_non_defaults:
                print(f"[*] Sampling {num_defaults} defaults & {num_non_defaults} non-defaults from cached dataset...")
                s1 = c1.sample(n=num_defaults, random_state=random_state)
                s0 = c0.sample(n=num_non_defaults, random_state=random_state)
                res = pd.concat([s1, s0], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
                return res
        elif len(cached_df) >= total_samples:
            print(f"[*] Sampling {total_samples} records from cached CTGAN dataset...")
            return cached_df.sample(n=total_samples, random_state=random_state).reset_index(drop=True)

    # Otherwise train CTGAN from scratch
    print(f"[*] Training CTGAN on input partition ({len(X_train_raw)} rows, {epochs} epochs)...")
    from ctgan import CTGAN
    train_df = X_train_raw.copy()
    train_df[TARGET_COL] = y_train_raw.values
    discrete_cols = [c for c in categorical_features if c in train_df.columns] + [TARGET_COL]

    ctgan = CTGAN(epochs=epochs, batch_size=batch_size, verbose=False)
    ctgan.fit(train_df, discrete_cols)

    bounds = {col: (train_df[col].min(), train_df[col].max()) for col in X_train_raw.columns}

    if num_defaults is not None and num_non_defaults is not None:
        print(f"[*] Sampling target distribution: {num_defaults} Defaults, {num_non_defaults} Non-defaults...")
        defaults_collected = []
        non_defaults_collected = []

        while len(defaults_collected) < num_defaults or len(non_defaults_collected) < num_non_defaults:
            sample_size = max(5000, (num_defaults - len(defaults_collected) + num_non_defaults - len(non_defaults_collected)) * 2)
            batch = ctgan.sample(sample_size)
            batch = clip_to_bounds(batch, bounds)

            b1 = batch[batch[TARGET_COL] == 1]
            b0 = batch[batch[TARGET_COL] == 0]

            if len(defaults_collected) < num_defaults and len(b1) > 0:
                defaults_collected.append(b1.iloc[:num_defaults - len(defaults_collected)])
            if len(non_defaults_collected) < num_non_defaults and len(b0) > 0:
                non_defaults_collected.append(b0.iloc[:num_non_defaults - len(non_defaults_collected)])

        synthetic_df = pd.concat(defaults_collected + non_defaults_collected, axis=0)
    else:
        print(f"[*] Sampling {total_samples} synthetic rows from CTGAN...")
        synthetic_df = ctgan.sample(total_samples)
        synthetic_df = clip_to_bounds(synthetic_df, bounds)

    synthetic_df = synthetic_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return synthetic_df
