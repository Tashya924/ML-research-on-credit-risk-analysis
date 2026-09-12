"""
Diffusion Data Generator (Dedicated CLI for Evaluators)
-------------------------------------------------------
Generates synthetic tabular credit card datasets using Tabular Denoising Diffusion
Probabilistic Models (TabDDPM) with All-Gaussian Bypass and domain bounds clipping.
Supports user-selectable ratios of Default (Target=1) vs Non-Default (Target=0).

Output datasets:
  - data/data_diffusion_corrected.csv  (Zero-leakage: trained on 75% train split only)
  - data/data_diffusion_leaky.csv      (Flawed baseline: trained on 100% full dataset)

Usage:
    # 1. Generate balanced diffusion dataset (1,000 defaults, 1,000 non-defaults)
    python data_generator_diffusion.py --mode corrected --defaults 1000 --non-defaults 1000

    # 2. Generate with target default ratio (e.g. 0.3 for 30% defaults out of 5,000)
    python data_generator_diffusion.py --mode corrected --ratio 0.3 --total 5000

    # 3. Generate leaky dataset (trained on entire dataset)
    python data_generator_diffusion.py --mode leaky --defaults 1000 --non-defaults 1000

    # 4. Interactive mode (guided step-by-step prompts)
    python data_generator_diffusion.py --interactive
"""

import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_credit_data, TARGET_COL
from src.generator_diffusion import generate_tabddpm_synthetic_data


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic tabular credit datasets using TabDDPM diffusion.")
    parser.add_argument("--mode", type=str, choices=["corrected", "leaky"], default="corrected",
                        help="'corrected' (trained strictly on train split, no leakage) or 'leaky' (trained on full data)")
    parser.add_argument("--defaults", type=int, default=None,
                        help="Number of synthetic Default (target=1) rows")
    parser.add_argument("--non-defaults", type=int, default=None,
                        help="Number of synthetic Non-Default (target=0) rows")
    parser.add_argument("--ratio", type=float, default=None,
                        help="Desired ratio of defaults (e.g. 0.5 for 50:50 balance, 0.3 for 30:70)")
    parser.add_argument("--total", type=int, default=2000,
                        help="Total rows to generate when using --ratio (default: 2,000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (.csv). Defaults to data/data_diffusion_<mode>.csv")
    parser.add_argument("--iterations", type=int, default=2000,
                        help="TabDDPM training iterations per class (default: 2000)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="TabDDPM batch size (default: 256)")
    parser.add_argument("--data-path", type=str, default="UCI_Credit_Card.csv",
                        help="Path to raw credit card dataset")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive prompt mode")
    return parser.parse_args()


def run_interactive():
    print("\n=======================================================")
    print("   Diffusion (TabDDPM) Dataset Generator - Interactive ")
    print("=======================================================")
    print("Choose Leakage Mode:")
    print("  1. Corrected (Zero-leakage: trained on 75% train split only) [Recommended]")
    print("  2. Leaky (Flawed: trained on 100% full dataset)")
    c_mode = input("Select mode (1-2) [default: 1]: ").strip() or "1"
    mode = "corrected" if c_mode == "1" else "leaky"

    try:
        n_def_input = input("\nEnter number of synthetic Defaults (Target=1) [default: 500]: ").strip()
        n_def = int(n_def_input) if n_def_input else 500
        n_non_def_input = input("Enter number of synthetic Non-Defaults (Target=0) [default: 500]: ").strip()
        n_non_def = int(n_non_def_input) if n_non_def_input else 500
    except ValueError:
        print("[!] Invalid integer. Exiting.")
        sys.exit(1)

    default_out = f"data/data_diffusion_{mode}.csv"
    out_path = input(f"Enter destination path [default: '{default_out}']: ").strip() or default_out
    return mode, n_def, n_non_def, out_path


def main():
    args = parse_args()

    if args.interactive:
        mode, n_def, n_non_def, out_path = run_interactive()
        iterations = args.iterations
        batch_size = args.batch_size
        data_path = args.data_path
    else:
        mode = args.mode
        n_def = args.defaults
        n_non_def = args.non_defaults
        iterations = args.iterations
        batch_size = args.batch_size
        data_path = args.data_path

        if args.ratio is not None:
            n_def = int(args.total * args.ratio)
            n_non_def = args.total - n_def
        elif n_def is None and n_non_def is None:
            # Default to 500 defaults, 500 non-defaults
            n_def = 500
            n_non_def = 500

        out_path = args.output if args.output else f"data/data_diffusion_{mode}.csv"

    print("===============================================================================")
    print(f"          GENERATING DIFFUSION DATASET (Mode: {mode.upper()})                  ")
    print("===============================================================================")
    print(f"[*] Target distribution: {n_def:,} Defaults (1), {n_non_def:,} Non-defaults (0)")
    print(f"[*] Total synthetic rows: {n_def + n_non_def:,}")
    print(f"[*] Destination: {out_path}")

    # Load base dataset
    df = load_credit_data(data_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    if mode == "corrected":
        # Strict zero-leakage: fit only on training partition
        print("[*] Partitioning 75/25 train/test split (zero-leakage)...")
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        # Leaky mode: fit on entire dataset
        print("[!] Warning: Leaky mode trains TabDDPM on the entire dataset.")
        X_train, y_train = X, y

    # Generate synthetic diffusion data
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    syn_df = generate_tabddpm_synthetic_data(
        X_train_raw=X_train,
        y_train_raw=y_train,
        num_defaults=n_def,
        num_non_defaults=n_non_def,
        cache_dir="data",
        n_iter=iterations,
        batch_size=batch_size
    )

    syn_df.to_csv(out_path, index=False)

    print(f"\n[✓] Diffusion synthetic dataset saved successfully to: '{out_path}'")
    print(f"    - Rows: {len(syn_df):,}")
    print(f"    - Columns: {len(syn_df.columns)}")
    print(f"    - Defaults (1): {(syn_df[TARGET_COL] == 1).sum():,} ({syn_df[TARGET_COL].mean():.1%})")
    print(f"    - Non-Defaults (0): {(syn_df[TARGET_COL] == 0).sum():,} ({(1 - syn_df[TARGET_COL].mean()):.1%})")


if __name__ == "__main__":
    main()
