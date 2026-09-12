"""
Data Generator (Naive CLI for Evaluators)
-----------------------------------------
Generates synthetic tabular credit card datasets using GAN and Diffusion models,
with selectable ratios of Default (Target=1) vs Non-Default (Target=0).

Supports both 'corrected' (zero-leakage, train only) and 'leaky' (trained on full data) modes:
- data_gan_corrected
- data_gan_leaky
- data_diffusion_corrected
- data_diffusion_leaky

Usage:
    # 1. Generate GAN Corrected (50:50 balanced: 30k defaults, 30k non-defaults)
    python data_generator.py --model gan --mode corrected --defaults 30000 --non-defaults 30000 --output data/data_gan_corrected.csv

    # 2. Generate GAN Leaky (trained on full dataset)
    python data_generator.py --model gan --mode leaky --defaults 30000 --non-defaults 30000 --output data/data_gan_leaky.csv

    # 3. Generate Diffusion Corrected (TabDDPM)
    python data_generator.py --model diffusion --mode corrected --defaults 10000 --non-defaults 10000 --output data/data_diffusion_corrected.csv

    # 4. Generate by specifying target ratio (e.g. 0.5 for 50% defaults)
    python data_generator.py --model gan --ratio 0.5 --total 60000 --output data/data_gan_corrected.csv

    # 5. Interactive prompt mode
    python data_generator.py --interactive
"""

import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_credit_data, get_feature_lists, TARGET_COL
from src.generators import generate_ctgan_synthetic_data, generate_tabddpm_synthetic_data


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic tabular credit datasets (GAN / Diffusion).")
    parser.add_argument("--model", type=str, choices=["gan", "diffusion", "both"], default="gan",
                        help="Generative model: 'gan' (CTGAN) or 'diffusion' (TabDDPM) or 'both'")
    parser.add_argument("--mode", type=str, choices=["corrected", "leaky"], default="corrected",
                        help="'corrected' (trained strictly on train split, no leakage) or 'leaky' (trained on full data)")
    parser.add_argument("--defaults", type=int, default=None,
                        help="Number of synthetic Default (target=1) rows")
    parser.add_argument("--non-defaults", type=int, default=None,
                        help="Number of synthetic Non-Default (target=0) rows")
    parser.add_argument("--ratio", type=float, default=None,
                        help="Desired ratio of defaults (e.g. 0.5 for 50:50, 0.3 for 30:70)")
    parser.add_argument("--total", type=int, default=60000,
                        help="Total rows to generate when using --ratio (default: 60,000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (.csv or .parquet)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="CTGAN training epochs (default: 20)")
    parser.add_argument("--ddpm-iter", type=int, default=2000,
                        help="TabDDPM training iterations (default: 2000)")
    parser.add_argument("--data-path", type=str, default="UCI_Credit_Card.csv",
                        help="Path to UCI Credit Card CSV")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive prompt mode")
    return parser.parse_args()


def run_interactive():
    print("\n=======================================================")
    print("        Synthetic Data Generator (Interactive)         ")
    print("=======================================================")
    print("Choose Generative Model:")
    print("  1. GAN (CTGAN)")
    print("  2. Diffusion (TabDDPM)")
    print("  3. Both Models")
    c_model = input("Select model (1-3) [default: 1]: ").strip() or "1"
    model_map = {"1": "gan", "2": "diffusion", "3": "both"}
    model = model_map.get(c_model, "gan")

    print("\nChoose Leakage Mode:")
    print("  1. Corrected (Zero-leakage: trained on train split only)")
    print("  2. Leaky (Flawed: trained on entire dataset)")
    c_mode = input("Select mode (1-2) [default: 1]: ").strip() or "1"
    mode = "corrected" if c_mode == "1" else "leaky"

    try:
        n_def_input = input("\nEnter number of synthetic Defaults (Target=1) [e.g. 30000]: ").strip()
        n_def = int(n_def_input) if n_def_input else 30000
        n_non_def_input = input("Enter number of synthetic Non-Defaults (Target=0) [e.g. 30000]: ").strip()
        n_non_def = int(n_non_def_input) if n_non_def_input else 30000
    except ValueError:
        print("[!] Invalid integer. Exiting.")
        sys.exit(1)

    default_out = f"data/data_{model}_{mode}.csv"
    out_path = input(f"Enter destination path [default: '{default_out}']: ").strip() or default_out
    return model, mode, n_def, n_non_def, out_path


def main():
    args = parse_args()

    if args.interactive or (len(sys.argv) == 1):
        model_choice, mode, num_defaults, num_non_defaults, output_path = run_interactive()
        epochs = 20
        ddpm_iter = 2000
        data_path = "UCI_Credit_Card.csv"
    else:
        model_choice = args.model
        mode = args.mode
        data_path = args.data_path
        epochs = args.epochs
        ddpm_iter = args.ddpm_iter

        if args.ratio is not None:
            num_defaults = int(args.total * args.ratio)
            num_non_defaults = args.total - num_defaults
        else:
            num_defaults = args.defaults if args.defaults is not None else 30000
            num_non_defaults = args.non_defaults if args.non_defaults is not None else 30000

        if args.output is not None:
            output_path = args.output
        else:
            output_path = f"data/data_{model_choice}_{mode}.csv"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 1. Load data
    df = load_credit_data(data_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    cat_cols, num_cols, _ = get_feature_lists(X)

    if mode == "corrected":
        print("[*] MODE: CORRECTED (zero-leakage, using 75% clean train partition)...")
        X_train_to_use, _, y_train_to_use, _ = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        print("[*] MODE: LEAKY (using 100% full dataset for generator training)...")
        X_train_to_use, y_train_to_use = X, y

    total_requested = num_defaults + num_non_defaults
    def_ratio = (num_defaults / total_requested) * 100
    print(f"\n[*] Target Class Distribution: {num_defaults} Defaults ({def_ratio:.1f}%), {num_non_defaults} Non-Defaults ({100-def_ratio:.1f}%)")

    models_to_run = ["gan", "diffusion"] if model_choice == "both" else [model_choice]

    for m in models_to_run:
        print(f"\n>>> Running {m.upper()} Generation ({mode.upper()}) <<<")
        if m == "gan":
            cache_file = "results/ctgan_synthetic_120000.parquet" if mode == "corrected" else None
            syn_df = generate_ctgan_synthetic_data(
                X_train_raw=X_train_to_use,
                y_train_raw=y_train_to_use,
                categorical_features=cat_cols,
                num_defaults=num_defaults,
                num_non_defaults=num_non_defaults,
                epochs=epochs,
                cache_path=cache_file
            )
        elif m == "diffusion":
            syn_df = generate_tabddpm_synthetic_data(
                X_train_raw=X_train_to_use,
                y_train_raw=y_train_to_use,
                num_defaults=num_defaults,
                num_non_defaults=num_non_defaults,
                n_iter=ddpm_iter,
                cache_dir="data"
            )

        cur_out = output_path
        if model_choice == "both":
            base, ext = os.path.splitext(output_path)
            cur_out = f"{base}_{m}{ext or '.csv'}"

        if cur_out.endswith(".parquet"):
            syn_df.to_parquet(cur_out, index=False)
        else:
            syn_df.to_csv(cur_out, index=False)

        print(f"[✓] {m.upper()} ({mode}) saved to '{cur_out}'! Total rows: {len(syn_df):,}")
        print(f"    Class Counts: {syn_df[TARGET_COL].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
