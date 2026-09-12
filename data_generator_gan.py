"""
GAN Data Generator (Dedicated CLI for Evaluators)
-------------------------------------------------
Generates synthetic tabular credit card datasets using Conditional Tabular GAN (CTGAN).
Supports user-selectable ratios of Default (Target=1) vs Non-Default (Target=0).

Output datasets:
  - data/data_gan_corrected.csv  (Zero-leakage: trained on 75% train split only)
  - data/data_gan_leaky.csv      (Flawed baseline: trained on 100% full dataset)

Usage:
    # 1. Generate 50:50 balanced corrected dataset (30,000 defaults, 30,000 non-defaults)
    python data_generator_gan.py --mode corrected --defaults 30000 --non-defaults 30000

    # 2. Generate with target default ratio (e.g. 0.3 for 30% defaults out of 60,000)
    python data_generator_gan.py --mode corrected --ratio 0.3 --total 60000

    # 3. Generate leaky dataset (trained on entire dataset)
    python data_generator_gan.py --mode leaky --defaults 30000 --non-defaults 30000

    # 4. Interactive mode (guided step-by-step prompts)
    python data_generator_gan.py --interactive
"""

import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_credit_data, get_feature_lists, TARGET_COL
from src.generator_gan import generate_ctgan_synthetic_data


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic tabular credit datasets using CTGAN.")
    parser.add_argument("--mode", type=str, choices=["corrected", "leaky"], default="corrected",
                        help="'corrected' (trained strictly on train split, no leakage) or 'leaky' (trained on full data)")
    parser.add_argument("--defaults", type=int, default=None,
                        help="Number of synthetic Default (target=1) rows")
    parser.add_argument("--non-defaults", type=int, default=None,
                        help="Number of synthetic Non-Default (target=0) rows")
    parser.add_argument("--ratio", type=float, default=None,
                        help="Desired ratio of defaults (e.g. 0.5 for 50:50 balance, 0.3 for 30:70)")
    parser.add_argument("--total", type=int, default=60000,
                        help="Total rows to generate when using --ratio (default: 60,000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (.csv or .parquet). Defaults to data/data_gan_<mode>.csv")
    parser.add_argument("--epochs", type=int, default=20,
                        help="CTGAN training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="CTGAN training batch size (default: 500)")
    parser.add_argument("--cache-path", type=str, default="data/ctgan_synthetic_120000.parquet",
                        help="Path to pre-trained CTGAN parquet cache if available")
    parser.add_argument("--data-path", type=str, default="UCI_Credit_Card.csv",
                        help="Path to raw credit card dataset")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive prompt mode")
    return parser.parse_args()


def run_interactive():
    print("\n=======================================================")
    print("      GAN (CTGAN) Dataset Generator - Interactive      ")
    print("=======================================================")
    print("Choose Leakage Mode:")
    print("  1. Corrected (Zero-leakage: trained on 75% train split only) [Recommended]")
    print("  2. Leaky (Flawed: trained on 100% full dataset)")
    c_mode = input("Select mode (1-2) [default: 1]: ").strip() or "1"
    mode = "corrected" if c_mode == "1" else "leaky"

    try:
        n_def_input = input("\nEnter number of synthetic Defaults (Target=1) [default: 30000]: ").strip()
        n_def = int(n_def_input) if n_def_input else 30000
        n_non_def_input = input("Enter number of synthetic Non-Defaults (Target=0) [default: 30000]: ").strip()
        n_non_def = int(n_non_def_input) if n_non_def_input else 30000
    except ValueError:
        print("[!] Invalid integer. Exiting.")
        sys.exit(1)

    default_out = f"data/data_gan_{mode}.csv"
    out_path = input(f"Enter destination path [default: '{default_out}']: ").strip() or default_out
    return mode, n_def, n_non_def, out_path


def main():
    args = parse_args()

    if args.interactive:
        mode, n_def, n_non_def, out_path = run_interactive()
        epochs = args.epochs
        batch_size = args.batch_size
        cache_path = args.cache_path
        data_path = args.data_path
    else:
        mode = args.mode
        n_def = args.defaults
        n_non_def = args.non_defaults
        epochs = args.epochs
        batch_size = args.batch_size
        cache_path = args.cache_path
        data_path = args.data_path

        if args.ratio is not None:
            n_def = int(args.total * args.ratio)
            n_non_def = args.total - n_def
        elif n_def is None and n_non_def is None:
            # Default to 50:50 balanced (30,000 of each)
            n_def = 30000
            n_non_def = 30000

        out_path = args.output if args.output else f"data/data_gan_{mode}.csv"

    print("===============================================================================")
    print(f"             GENERATING GAN DATASET (Mode: {mode.upper()})                     ")
    print("===============================================================================")
    print(f"[*] Target distribution: {n_def:,} Defaults (1), {n_non_def:,} Non-defaults (0)")
    print(f"[*] Total synthetic rows: {n_def + n_non_def:,}")
    print(f"[*] Destination: {out_path}")

    # Load base dataset
    df = load_credit_data(data_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    cat_cols, num_cols, _ = get_feature_lists(X)

    if mode == "corrected":
        # Strict zero-leakage: fit only on training partition
        print("[*] Partitioning 75/25 train/test split (zero-leakage)...")
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        # Leaky mode: fit on entire dataset
        print("[!] Warning: Leaky mode trains CTGAN on the entire dataset.")
        X_train, y_train = X, y

    # Generate synthetic data
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    syn_df = generate_ctgan_synthetic_data(
        X_train_raw=X_train,
        y_train_raw=y_train,
        categorical_features=cat_cols,
        num_defaults=n_def,
        num_non_defaults=n_non_def,
        epochs=epochs,
        batch_size=batch_size,
        cache_path=cache_path if mode == "corrected" else None
    )

    if out_path.endswith(".parquet"):
        syn_df.to_parquet(out_path, index=False)
    else:
        syn_df.to_csv(out_path, index=False)

    print(f"\n[✓] GAN synthetic dataset saved successfully to: '{out_path}'")
    print(f"    - Rows: {len(syn_df):,}")
    print(f"    - Columns: {len(syn_df.columns)}")
    print(f"    - Defaults (1): {(syn_df[TARGET_COL] == 1).sum():,} ({syn_df[TARGET_COL].mean():.1%})")
    print(f"    - Non-Defaults (0): {(syn_df[TARGET_COL] == 0).sum():,} ({(1 - syn_df[TARGET_COL].mean()):.1%})")


if __name__ == "__main__":
    main()
