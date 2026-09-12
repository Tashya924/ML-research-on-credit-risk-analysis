"""
Generative Tabular Synthesis: Conditional Tabular GAN (CTGAN) and
Tabular Denoising Diffusion Probabilistic Models (TabDDPM).
Guarantees strict zero-leakage training exclusively on the training partition,
with user-selectable default / non-default ratios and domain bounds restoration.
"""

import os
import sys
import subprocess
import tempfile
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

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
    cache_path: Optional[str] = "results/ctgan_synthetic_120000.parquet",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic tabular data using CTGAN with user-specified class distribution.
    """
    # If ratio is provided, compute counts
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
    print(f"[*] Training CTGAN on clean training partition ({len(X_train_raw)} rows, {epochs} epochs)...")
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


def generate_tabddpm_synthetic_data(
    X_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    num_defaults: int = 10000,
    num_non_defaults: int = 10000,
    cache_dir: str = "data",
    n_iter: int = 2000,
    batch_size: int = 256,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic tabular data using TabDDPM (Diffusion Model) per class,
    applying the All-Gaussian Bypass and domain bounds restoration.
    Uses uv environment running Synthcity under Python 3.10.
    """
    os.makedirs(cache_dir, exist_ok=True)
    temp_train_path = os.path.join(cache_dir, "_train_for_ddpm.csv")
    output_csv = os.path.join(cache_dir, f"synthetic_tabddpm_d{num_defaults}_nd{num_non_defaults}.csv")

    if os.path.exists(output_csv):
        print(f"[*] Found existing TabDDPM synthetic data at '{output_csv}'. Loading...")
        return pd.read_csv(output_csv)

    train_data = pd.concat([X_train_raw, y_train_raw], axis=1)
    train_data.to_csv(temp_train_path, index=False)

    generator_script = f"""
import os
import numpy as np
import torch
import pandas as pd
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.serialization import save_to_file, load_from_file

TARGET_COL = '{TARGET_COL}'
train_data = pd.read_csv('{temp_train_path}')
FEATURES = [c for c in train_data.columns if c != TARGET_COL]
bounds = {{col: (train_data[col].min(), train_data[col].max()) for col in FEATURES}}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_generate(target_label, total_needed):
    if total_needed <= 0:
        return pd.DataFrame()
    model_path = os.path.join('{cache_dir}', f'tabddpm_model_class_{{target_label}}.pkl')
    class_data = train_data[train_data[TARGET_COL] == target_label].copy()
    
    # All-Gaussian Bypass
    for col in FEATURES:
        class_data[col] = class_data[col].astype(float) + np.random.uniform(-1e-5, 1e-5, size=len(class_data))
        
    if os.path.exists(model_path):
        print(f'[*] Loading TabDDPM model from {{model_path}}')
        model = load_from_file(model_path)
    else:
        print(f'[*] Training TabDDPM for class {{target_label}} ({n_iter} iterations)...')
        loader = GenericDataLoader(class_data, target_column=TARGET_COL)
        model = Plugins().get('ddpm', n_iter={n_iter}, batch_size={batch_size}, device=device)
        model.fit(loader)
        save_to_file(model_path, model)
        
    print(f'[*] Sampling {{total_needed}} rows...')
    syn_loader = model.generate(count=total_needed)
    syn_df = syn_loader.dataframe()
    
    # Restore bounds
    for col in FEATURES:
        b_min, b_max = bounds[col]
        syn_df[col] = syn_df[col].round().astype(int).clip(lower=b_min, upper=b_max)
    syn_df[TARGET_COL] = target_label
    return syn_df

df1 = train_and_generate(1, {num_defaults})
df0 = train_and_generate(0, {num_non_defaults})
out = pd.concat([df1, df0], ignore_index=True).sample(frac=1.0, random_state={random_state}).reset_index(drop=True)
out.to_csv('{output_csv}', index=False)
print('TabDDPM synthetic data generated successfully.')
"""

    script_path = os.path.join(cache_dir, "_run_ddpm.py")
    with open(script_path, "w") as f:
        f.write(generator_script)

    print(f"[*] Executing TabDDPM via uv (Python 3.10 with Synthcity) to generate {num_defaults} defaults and {num_non_defaults} non-defaults...")
    cmd = [
        "uv", "run",
        "--with", "synthcity",
        "--with", "opacus<1.4.1",
        "--with", "pyarrow<15.0.0",
        "--with", "torch",
        "--with", "scikit-learn",
        "--with", "pandas",
        "--with", "numpy<2",
        "--python", "3.10",
        "python", script_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("[!] Warning: Synthcity TabDDPM returned error:", res.stderr)
        raise RuntimeError(f"TabDDPM generation failed: {res.stderr}")

    # Clean up temp script
    if os.path.exists(script_path):
        os.remove(script_path)
    if os.path.exists(temp_train_path):
        os.remove(temp_train_path)

    return pd.read_csv(output_csv)
