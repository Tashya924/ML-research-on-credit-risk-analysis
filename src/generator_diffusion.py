"""
Dedicated Tabular Diffusion (TabDDPM) Tabular Data Generator.
Provides zero-leakage training on training partition or leaky full-dataset training,
with user-selectable default / non-default ratios, All-Gaussian Bypass, and domain bounds clipping.
"""

import os
import subprocess
from typing import Optional
import pandas as pd

TARGET_COL = "default.payment.next.month"


def generate_tabddpm_synthetic_data(
    X_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    num_defaults: Optional[int] = None,
    num_non_defaults: Optional[int] = None,
    default_ratio: Optional[float] = None,
    total_samples: int = 20000,
    cache_dir: str = "data",
    n_iter: int = 2000,
    batch_size: int = 256,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic tabular data using TabDDPM (Diffusion Model) per class,
    applying the All-Gaussian Bypass and domain bounds restoration.
    Uses uv environment running Synthcity under Python 3.10.

    Args:
        X_train_raw: Training features DataFrame
        y_train_raw: Training target Series
        num_defaults: Target count of default (target=1) instances
        num_non_defaults: Target count of non-default (target=0) instances
        default_ratio: Target proportion of defaults (e.g. 0.5 for 50:50 balance)
        total_samples: Total rows to generate when using default_ratio
        cache_dir: Cache directory for models and temporary files
        n_iter: Training iterations for DDPM per class
        batch_size: Batch size for DDPM
        random_state: Random seed
    """
    if default_ratio is not None:
        num_defaults = int(total_samples * default_ratio)
        num_non_defaults = total_samples - num_defaults
    elif num_defaults is not None and num_non_defaults is not None:
        total_samples = num_defaults + num_non_defaults
    else:
        num_defaults = int(total_samples * 0.5)
        num_non_defaults = total_samples - num_defaults

    os.makedirs(cache_dir, exist_ok=True)
    temp_train_path = os.path.join(cache_dir, "_train_for_ddpm.csv")
    output_csv = os.path.join(cache_dir, f"synthetic_tabddpm_d{num_defaults}_nd{num_non_defaults}.csv")
    pool_path = os.path.join(cache_dir, "synthetic_tabddpm_pool.parquet")

    if os.path.exists(output_csv):
        print(f"[*] Found existing TabDDPM synthetic data at '{output_csv}'. Loading...")
        return pd.read_csv(output_csv)

    # Use a pre-generated TabDDPM pool only if it holds enough distinct rows per class;
    # sampling with replacement would inflate "size" with duplicated rows.
    if os.path.exists(pool_path):
        pool_df = pd.read_parquet(pool_path)
        p1 = pool_df[pool_df[TARGET_COL] == 1]
        p0 = pool_df[pool_df[TARGET_COL] == 0]
        if len(p1) >= num_defaults and len(p0) >= num_non_defaults:
            print(f"[*] Sampling {num_defaults} defaults & {num_non_defaults} non-defaults from TabDDPM pool...")
            s1 = p1.sample(n=num_defaults, random_state=random_state)
            s0 = p0.sample(n=num_non_defaults, random_state=random_state)
            res = pd.concat([s1, s0], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            return res
        print(f"[!] TabDDPM pool too small for {num_defaults}/{num_non_defaults} rows; generating fresh samples.")

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
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        for tmp in (script_path, temp_train_path):
            if os.path.exists(tmp):
                os.remove(tmp)
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"TabDDPM generation failed: {res.stderr}")

    return pd.read_csv(output_csv)
