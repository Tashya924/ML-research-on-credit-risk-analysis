# /// script
# requires-python = "==3.10.*"
# dependencies = [
#      "kagglehub",
#      "pandas",
#      "numpy",
#      "scikit-learn",
#      "torch",
#      "opacus<1.4.1",
#      "synthcity",
#      "pyarrow<15.0.0"
# ]
# ///

"""
Synthetic Data Generator using TabDDPM (Diffusion Models)
---------------------------------------------------------
This script downloads the UCI Default of Credit Card Clients Dataset, 
partitions it, and trains a TabDDPM on user-specified classes to generate 
purely synthetic data in memory-safe chunks.

Features:
- Interactive prompts to define the exact class distribution (Defaults vs. Non-defaults).
- "All-Gaussian Bypass": Injects micro-noise to artificially force Synthcity 
  to strictly use continuous Gaussian diffusion. This bypasses internal 
  cardinality heuristics that cause CUDA out-of-bounds assertions on large distributions.
- Exact bounds restoration: Erases micro-noise and clips generated values 
  strictly to the original dataset's min/max ranges.

Usage:
    uv run generate_synthetic_data.py
"""

import os
import numpy as np
import torch
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.serialization import save_to_file, load_from_file

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_COL = "default.payment.next.month"
CHUNK_SIZE = 25000  # Safe batch size for 16GB VRAM GPUs (e.g., Colab T4)
OUTPUT_CSV = "data/synthetic_credit_data.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Initializing hardware execution device: {device}")

os.makedirs("data", exist_ok=True)

# ==========================================
# 1. USER INPUT
# ==========================================
print("\n--- Synthetic Data Configuration ---")
try:
    num_defaults = int(input("Enter number of synthetic Defaults (Target=1, Minority) to generate: "))
    num_non_defaults = int(input("Enter number of synthetic Non-defaults (Target=0, Majority) to generate: "))
except ValueError:
    print("[!] Invalid input. Please enter valid integers.")
    exit(1)

if num_defaults == 0 and num_non_defaults == 0:
    print("[!] Both quantities are zero. Exiting.")
    exit(0)

# ==========================================
# 2. DOWNLOAD & PARTITION DATA
# ==========================================
print("\n[*] Downloading dataset from Kaggle...")
cache_path = kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset")
csv_path = os.path.join(cache_path, "UCI_Credit_Card.csv")
df = pd.read_csv(csv_path)

# Dynamically map feature columns (ignoring ID and Target)
FEATURES = [col for col in df.columns if col not in ["ID", TARGET_COL]]

X = df.drop(columns=["ID", TARGET_COL])
y = df[TARGET_COL]

print("[*] Partitioning original data (75/25 split) for training baseline...")
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
train_data = pd.concat([X_train_raw, y_train_raw], axis=1)

# Cache original bounds for exact post-generation restoration
bounds = {col: (train_data[col].min(), train_data[col].max()) for col in FEATURES}

# ==========================================
# 3. CORE GENERATION LOGIC
# ==========================================
def train_and_generate(target_label, total_needed):
    """
    Isolates specific class data, trains a DDPM model, and generates synthetic rows.
    """
    if total_needed <= 0:
        return pd.DataFrame()

    class_name = "Defaults (1)" if target_label == 1 else "Non-defaults (0)"
    model_path = f"data/tabddpm_model_class_{target_label}.pkl"
    
    print(f"\n--- Processing {class_name} ---")
    class_data = train_data[train_data[TARGET_COL] == target_label].copy()
    print(f"[*] Extracted {len(class_data)} real samples for training.")

    # THE BYPASS: Inject microscopic noise into ALL features.
    # This guarantees Synthcity bypasses categorical heuristics and routes 
    # 100% of the data through the highly stable Gaussian diffusion pipeline.
    print("[*] Applying All-Gaussian Bypass (injecting micro-noise)...")
    for col in FEATURES:
        class_data[col] = class_data[col].astype(float) + np.random.uniform(-1e-5, 1e-5, size=len(class_data))

    # TRAIN OR LOAD
    if os.path.exists(model_path):
        print(f"[*] Found existing model at {model_path}. Loading instantly...")
        model = load_from_file(model_path)
    else:
        print(f"[*] Initiating TabDDPM training for {class_name}...")
        loader = GenericDataLoader(class_data, target_column=TARGET_COL)
        model = Plugins().get("ddpm", n_iter=2000, batch_size=256, device=device)
        model.fit(loader)
        save_to_file(model_path, model)
        print(f"[*] Model successfully saved to {model_path}")

    # CHUNKED GENERATION
    chunks = []
    total_chunks = (total_needed // CHUNK_SIZE) + (1 if total_needed % CHUNK_SIZE != 0 else 0)
    
    print(f"[*] Starting chunked generation of {total_needed} synthetic samples...")
    for i in range(1, total_chunks + 1):
        # Prevent over-generation on the final chunk
        current_chunk_size = min(CHUNK_SIZE, total_needed - sum(len(c) for c in chunks))
        print(f"    -> Generating chunk {i}/{total_chunks} ({current_chunk_size} rows)...")
        syn_loader = model.generate(count=current_chunk_size)
        chunks.append(syn_loader.dataframe())

    synthetic_df = pd.concat(chunks, ignore_index=True)

    # RESTORE DOMAIN BOUNDS
    print("[*] Erasing micro-noise and restoring strict domain bounds...")
    for col in FEATURES:
        min_val, max_val = bounds[col]
        # Round back to integers and strictly clip to the original min/max bounds
        synthetic_df[col] = synthetic_df[col].round().astype(int).clip(lower=min_val, upper=max_val)

    # Force strict target label just in case the model drifted
    synthetic_df[TARGET_COL] = target_label

    return synthetic_df

# ==========================================
# 4. EXECUTION & EXPORT
# ==========================================
# Generate both classes based on user input
df_defaults = train_and_generate(target_label=1, total_needed=num_defaults)
df_non_defaults = train_and_generate(target_label=0, total_needed=num_non_defaults)

print("\n--- Finalizing Dataset ---")
# Combine whatever was generated
final_synthetic_df = pd.concat([df_defaults, df_non_defaults], ignore_index=True)

# Shuffle the combined dataset so classes are randomly distributed
final_synthetic_df = final_synthetic_df.sample(frac=1, random_state=42).reset_index(drop=True)

final_synthetic_df.to_csv(OUTPUT_CSV, index=False)
print(f"[*] Success! Saved a total of {len(final_synthetic_df)} purely synthetic rows to '{OUTPUT_CSV}'.")