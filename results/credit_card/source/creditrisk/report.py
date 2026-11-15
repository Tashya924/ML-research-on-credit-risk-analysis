"""Report only synthetic-data generation and prediction performance on the original CSV."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .artifacts import write_json
from .classifiers import MODEL_NAMES
from .experiment import METHODS

LABELS = {
    "real": "Original training data",
    "agss_interpolation": "AGSS interpolation (DBSCAN-guided)",
    "diffusion_mlp": "Diffusion with MLP",
    "diffusion_transformer": "Diffusion with Transformer",
    "agss_diffusion_mlp": "AGSS + diffusion with MLP",
    "agss_diffusion_transformer": "AGSS + diffusion with Transformer",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "hist_gradient_boosting": "Gradient boosting",
    "random_forest": "Random forest",
    "extra_trees": "Extra trees",
}


def table(frame):
    def display(value):
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    return "\n".join(
        ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
        + [
            "| " + " | ".join(display(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
    )


def build(root, output):
    root, output = Path(root), Path(output)
    run = json.loads((root / "run.json").read_text())
    if run["status"] not in {"complete", "complete_with_failures"} or "dataset" not in run:
        raise ValueError("Use a completed credit-card comparison run")
    if set(run["methods"]) - set(METHODS):
        raise ValueError("Run contains methods outside the credit-card comparison")
    output.parent.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(root / "metrics.csv")
    if set(results.policy) != {"raw_0.5", "validation_f1"}:
        raise ValueError("Unexpected comparison policies")
    chosen = results[results.policy == "validation_f1"]
    rows = []
    for (method, model), group in chosen.groupby(["method", "model"], sort=False):
        row = dict(method=method, model=model)
        for metric in ["average_precision", "roc_auc", "f1", "precision", "recall", "accuracy"]:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "comparison_summary.csv", index=False)
    baseline = chosen[chosen.method == "real"].set_index(["seed", "model"])
    differences = chosen[chosen.method != "real"].copy()
    for metric in ["average_precision", "f1"]:
        differences[f"delta_{metric}"] = [
            r[metric] - baseline.loc[(r["seed"], r["model"]), metric] for _, r in differences.iterrows()
        ]
    differences.to_csv(root / "paired_differences.csv", index=False)
    average_deltas = differences.groupby(["method", "model"]).delta_average_precision.mean()
    inventory = []
    for seed in run["configuration"]["seeds"]:
        for method in run["methods"]:
            path = root / f"seed_{seed}" / method
            if not (path / "complete.json").exists():
                continue
            meta = json.loads((path / "method.json").read_text())
            inventory.append(
                dict(
                    seed=seed,
                    method=method,
                    generated_rows=meta["synthetic_rows"],
                    total_training_rows=int(
                        results[(results.seed == seed) & (results.method == method)].iloc[0].train_rows
                    ),
                    selected_epoch=meta.get("selected_epoch"),
                    synthetic_file=str(path / "synthetic.csv.gz") if meta["synthetic_rows"] else "",
                    training_file=str(path / "training.csv.gz"),
                )
            )
    generated = pd.DataFrame(inventory)
    generated.to_csv(root / "dataset_inventory.csv", index=False)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/creditrisk-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(15, 5), sharey=True)
    for axis, model in zip(axes, MODEL_NAMES, strict=True):
        group = summary[summary.model == model].set_index("method").reindex(run["methods"])
        axis.barh(
            range(len(group)),
            group.average_precision_mean,
            xerr=group.average_precision_sd,
            capsize=3,
            color=["#555555" if m == "real" else "#347c98" for m in group.index],
        )
        axis.set_yticks(range(len(group)), [LABELS[m] for m in group.index], fontsize=8)
        axis.set_title(MODEL_LABELS[model], fontsize=10)
        axis.set_xlabel("Average precision (mean ± split SD)", fontsize=8)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.tight_layout()
    image = root / "comparison.png"
    fig.savefig(image, dpi=160)
    plt.close(fig)
    full = []
    for row in summary.itertuples(index=False):
        full.append(
            dict(
                Method=LABELS[row.method],
                Model=MODEL_LABELS[row.model],
                AP=f"{row.average_precision_mean:.4f} ± {row.average_precision_sd:.4f}",
                AUC=row.roc_auc_mean,
                F1=row.f1_mean,
                Precision=row.precision_mean,
                Recall=row.recall_mean,
                Accuracy=row.accuracy_mean,
            )
        )
    full = pd.DataFrame(full)
    main = full[full.Model == "Gradient boosting"].drop(columns="Model")
    count_table = (
        generated.groupby("method", sort=False)
        .agg(
            Datasets=("seed", "size"),
            Minimum_generated_rows=("generated_rows", "min"),
            Maximum_generated_rows=("generated_rows", "max"),
        )
        .reset_index()
    )
    count_table["method"] = count_table.method.map(LABELS)
    count_table = count_table.rename(
        columns={
            "method": "Method",
            "Minimum_generated_rows": "Generated rows (min)",
            "Maximum_generated_rows": "Generated rows (max)",
        }
    )
    best_real = (
        summary[summary.method == "real"].sort_values("average_precision_mean", ascending=False).iloc[0]
    )
    verification = (
        json.loads((root / "verification.json").read_text())
        if (root / "verification.json").exists()
        else None
    )
    verified = (
        f"Verification replayed {verification['serialized_classifiers_verified']} saved classifiers and recomputed "
        f"{verification['metric_records_verified']} result rows."
        if verification
        else "Run the saved-model verifier before relying on the results."
    )
    note = (
        "The existing credit-card fits were reused and reverified; no new training is implied."
        if run.get("provenance")
        else "This run trained the models from the original credit-card CSV."
    )
    report = f"""# Synthetic credit-card data: generation and model comparison

## What was done

We used only `UCI_Credit_Card.csv`: {run["dataset"]["rows"]:,} customer records, 23 features and the next-month-default label. We generated training data using AGSS interpolation and diffusion, with DBSCAN guidance and MLP/Transformer backbones, then compared models trained on those datasets against models trained on the original data.

{note}

The comparison contains **{len(run["methods"]) - 1} generation methods**, **{len(run["configuration"]["seeds"])} train/validation/test splits**, and **four prediction models**. There are **{len(chosen)} model fits** and **{len(results)} result rows** covering two decision thresholds.

## How the datasets were generated

- **AGSS interpolation:** DBSCAN groups training defaults; nearby records within retained groups are interpolated. Categorical values come from the source records.
- **Diffusion + MLP:** a diffusion model learns to remove noise using an MLP.
- **Diffusion + Transformer:** the denoiser uses a separate token for each feature.
- **AGSS + diffusion + MLP:** the generator learns from DBSCAN-retained defaults and receives cluster labels.
- **AGSS + diffusion + Transformer:** the same cluster guidance with a Transformer denoiser.

For each split, we kept all original training records and generated extra defaults to balance defaults and non-defaults. `synthetic.csv.gz` contains the generated default rows; `training.csv.gz` contains the complete two-class training dataset used for the comparison. The original-data baseline has no generated rows.

{table(count_table)}

The code uses the TabDDPM diffusion loss core with project-specific sampling/backbone choices. AGSS's mixed-feature interpolation is also a stated adaptation; these are not claims of reproducing published results exactly. Transformer was chosen from the requested Transformer-or-Mamba option.

## How performance was checked

The same real data partitions were used for every method in a split: approximately 60% training, 20% validation and 20% testing. Identical feature rows stay in one partition. Generated records enter training only. Preprocessing and generation learn from training records; validation chooses density settings, the diffusion checkpoint and the F1 cutoff. Test records are used for the reported scores.

The prediction models are logistic regression, gradient boosting, random forest and extra trees. **Average precision (AP)** measures how well defaults rise to the top of the risk ranking. **Precision** measures how many flagged cases really default; **recall** measures how many defaults are caught; **F1** balances precision and recall. Higher is better for these scores. Results at cutoff 0.5 are also saved in the CSV.

## Results

The table below keeps the classifier fixed at gradient boosting so that it compares the training datasets directly. Values are means across the splits. The ± value for AP is the split standard deviation, not a confidence interval.

{table(main)}

Across the four classifiers, **{int((average_deltas > 0).sum())} of {len(average_deltas)} generation-method/classifier combinations improved mean AP** over their matching original-data baseline. The strongest original-data result was {MODEL_LABELS[best_real.model]} at AP **{best_real.average_precision_mean:.4f}**. Small F1 changes should be read alongside AP: changing the decision cutoff can improve F1 without improving the ranking.

![Performance comparison]({os.path.relpath(image, output.parent)})

All four classifiers are listed in [the full comparison](COMPARISON.md). Per-split values are in `metrics.csv`, and `paired_differences.csv` subtracts each classifier's original-data result on the same split.

## Files and checks

The results are in `{root}`. `dataset_inventory.csv` lists the generated and complete training files. Each method directory also contains its fitted classifier files, generator weights/schema where applicable, selected-checkpoint information and saved validation/test predictions.

{verified} The fitting source is preserved separately from later code cleanup, so reusing an existing fit does not misrepresent it as newly trained.

These findings apply to this dataset and these settings. Repeated splits overlap, so their standard deviation is descriptive. The comparison does not establish that any method will improve every credit-risk dataset.
"""
    output.write_text(report)
    (output.parent / "COMPARISON.md").write_text(
        "# Full credit-card generation comparison\n\nAll methods use the original CSV and matching real test splits. Scores below use validation-selected F1 cutoffs. AP shows mean ± split standard deviation; other scores show means.\n\n"
        + table(full)
        + "\n\nFor individual splits and cutoff 0.5, use `results/credit_card/metrics.csv`.\n"
    )
    write_json(
        root / "analysis.json",
        dict(status="complete", model_fits=len(chosen), comparison_rows=len(results), report=str(output)),
    )
    print(f"Rebuilt {output}: {len(chosen)} model fits, {len(results)} comparison rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/credit_card")
    parser.add_argument("--output", default="docs/REPORT.md")
    args = parser.parse_args()
    build(args.input, args.output)
