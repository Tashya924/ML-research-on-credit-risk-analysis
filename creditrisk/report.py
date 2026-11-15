"""Report only synthetic-data generation and prediction performance on the original CSV."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import write_json
from .classifiers import MODEL_NAMES
from .data import file_hash
from .evidence import COUNTS, METRIC_GUIDE, SCORES
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
    "hist_gradient_boosting": "Histogram gradient boosting",
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


def write_grounding_docs(root, destination, run, results, evidence):
    def link(path):
        return os.path.relpath(path, destination)

    guide = pd.DataFrame(
        [
            {
                "CSV field": key,
                "Meaning": METRIC_GUIDE[key][0],
                "Formula": METRIC_GUIDE[key][1],
                "Why included": METRIC_GUIDE[key][2],
                "Uses cutoff?": "Yes" if METRIC_GUIDE[key][3] else "No",
            }
            for key in (*SCORES, *COUNTS)
        ]
    )
    seed = run["configuration"]["seeds"][0]
    job = root / f"seed_{seed}" / "real"
    prediction = job / "hist_gradient_boosting_predictions.npz"
    values = np.load(prediction)
    cutoff = float(values["threshold"])
    local = root / "evidence" / f"seed_{seed}" / "real" / "hist_gradient_boosting"
    ledger = pd.read_csv(root / "metric_evidence.csv")
    example = ledger[
        (ledger.seed == seed)
        & (ledger.method == "real")
        & (ledger.model == "hist_gradient_boosting")
        & (ledger.policy == "validation_f1")
    ]
    example_table = example[["metric", "value", "calculation"]].rename(
        columns={"metric": "Metric", "value": "Value", "calculation": "Actual calculation"}
    )
    split = pd.read_csv(root / "split_evidence.csv")
    split_table = split[["seed", "partition", "rows", "defaults", "nondefaults", "default_share"]]
    metrics_doc = f"""# What every metric means, why it is used, and where it comes from

## Common basis

Every prediction-performance score uses **real test records**, with `default.payment.next.month=1` as the positive class. A predicted default means `score >= threshold`. Scores are binary class-1 measures, not weighted or macro F1. For precision, recall and F1, a zero denominator is reported as zero. That convention is explicit in the frozen metric code.

The two policies are `raw_0.5` (fixed cutoff .5) and `validation_f1` (cutoff selected on real validation records). AP and ROC-AUC use the same unthresholded score vector under both policies, so their repetition is intentional. Two policy rows do not represent two model fits.

{table(guide)}

The implementation is in [the exact fitting-time metric code]({link(root / "source/creditrisk/metrics.py")}). Definitions agree with scikit-learn's [AP](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.average_precision_score.html), [ROC-AUC](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.roc_auc_score.html), [precision/recall/F1](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.precision_recall_fscore_support.html), and [balanced accuracy](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.balanced_accuracy_score.html) documentation. All seven score values are on a 0–1 scale, confusion counts are integers, and durations are seconds. Those links explain definitions; the local run's library versions and predictions establish its numerical results. AP is the precision-weighted sum of recall increments, **not trapezoidal PR-AUC**. ROC-AUC uses ROC trapezoids and cannot be reconstructed from one confusion matrix.

## Where to check each number

[metric_evidence.csv]({link(root / "metric_evidence.csv")}) has one row per individual metric value. Search by `seed/method/model/policy/metric`. It gives the substituted-count formula or ranking-curve sum, the exact source prediction/model/training files and hashes, the real test size/default count, the code file, and the physical line in `metrics.csv`. The authoritative full-precision values are in NPZ arrays; the CSVs are readable derivatives.

`ranking_curve.csv.gz` groups tied scores at their common threshold. Sum `ap_contribution` to recover AP; sum `roc_area_contribution` to recover ROC-AUC. The curve also shows cumulative TP/FP, precision, recall and FPR. Both sums are checked against scikit-learn. [A worked example is here](EVIDENCE.md).

## Cutoff selection and generator-setting scores

For `validation_f1`, 91 cutoffs from .05 through .95 in .01 steps are evaluated on validation predictions. The first maximum is chosen, so ties select the lowest cutoff. Each model's `validation_thresholds.csv` contains all 91 confusion counts and F1 values. This selects an operating cutoff, not a better ranking model.

[selection_evidence.csv]({link(root / "selection_evidence.csv")}) covers {evidence["validation_scores"]} saved validation AP scores used for DBSCAN-rule and generator-checkpoint selection. They are **not test results**. A fixed small gradient-boosting proxy is fitted on real training plus candidate-generated records and scored on real validation records. Its exact parameters are in each seed's `selection_model.json`; the final four classifiers have separate parameters in `classifier_parameters.json`. Maximum validation AP selects the density rule and checkpoint; test scores do not select either. Checkpoint candidates and failures are never substituted with test scores.

## Means, standard deviations and differences

For each method/classifier, the report uses the unweighted arithmetic mean of the {len(run["configuration"]["seeds"])} split scores, taking only one policy at a time. Sample SD is `sqrt(sum((score-mean)^2)/(n-1))`, with `n={len(run["configuration"]["seeds"])}`; it is not a confidence interval or standard error. We do not pool predictions across overlapping splits or count the two policy rows as additional runs.

[summary_evidence.csv]({link(root / "summary_evidence.csv")}) lists every reported mean, SD, sample count, and the {len(run["configuration"]["seeds"])} individual evidence IDs from which it was computed. `comparison_summary.csv` is the wide version. `paired_differences.csv` subtracts the same classifier's real-only score from each generated-data score **within the same seed/test partition**. Positive AP/F1 differences favor augmentation. These are descriptive differences, not significance tests.

## Counts, time and configuration fields

[metadata_evidence.csv]({link(root / "metadata_evidence.csv")}) explains and locates each non-score numeric field in `metrics.csv`:

- `threshold`: the stated cutoff; its justification is fixed .5 or maximum validation F1.
- `real_train_rows`, `synthetic_rows`, `train_rows`, `test_rows`: counts of the corresponding data tables/indices. Complete training rows equal real training plus generated defaults.
- `generation_seconds`: the observed method-build wall-time interval. It includes generator fitting, sampling, validation proxy selection, logging and generator-file exports. It is **not pure sampling speed**.
- `fit_seconds`: the observed classifier-fit interval including its start-event logging. It excludes preprocessing, score prediction and cutoff search.

Times are clock differences from `events.jsonl`, checked against the CSV. Parallel workers share the machine, so timings are operational observations, not isolated hardware benchmarks. The same fit/build time appears in both policy rows and must not be summed twice. `seed`, `method`, `model` and `policy` are identifiers, not performance metrics. `run.json` records the actual command, versions, worker/thread counts and start/end times.

## Generator training loss is not test performance

`training_loss.csv` records each epoch's optimization objective. It is the example-weighted mean of batch `numeric_mse + categorical_vb`; `batch_loss.csv` records the actual batch components and example counts so the epoch totals can be recomputed. Numeric loss is Gaussian noise-prediction MSE; categorical loss is the official TabDDPM stochastic multinomial variational loss, normalized by the number of categorical features. It is not the earlier CE-only shortcut.

The exact computation is in [the frozen diffusion adapter]({link(root / "source/creditrisk/diffusion.py")}) and [the frozen upstream diffusion core]({link(root / "source/vendor/tab_ddpm/gaussian_multinomial_diffsuion.py")}). Random corruption is part of training; batch logs establish the recorded optimization trace, not a replay from final weights alone. Lower training loss is not proof of useful synthetic records and does not select the checkpoint: validation AP does. A selected epoch-50 model can coexist with a loss history through epoch 150 because both candidate budgets were trained and compared.

DBSCAN epsilon, distance quantile, cluster counts/noise, PCA dimensions, generator epochs/timesteps and model hyperparameters describe the generation recipe. They are not prediction-performance scores. Their exact values and selection evidence are saved with the corresponding seed/method. Fixed constants such as PCA .85 and curvature .05 are project choices, not empirically proven optima.
"""
    (destination / "METRICS.md").write_text(metrics_doc)
    evidence_doc = f"""# Evidence: from the original CSV to every reported score

## Exact input and splits

The sole input is `UCI_Credit_Card.csv`; SHA-256 is `{run["dataset"]["sha256"]}`. ID is excluded from model features. Customer IDs appear only in the readable evidence export to locate the original record.

{table(split_table)}

Each seed's `split_indices.npz` gives original zero-based CSV row positions for train, validation and test. The readable export adds `csv_line_number = csv_row_index + 2` because the original CSV has one header line. Duplicate feature groups cannot cross partitions. Generated rows never enter validation or test. The verifier checks these mappings against the original CSV and separately refits the training-only preprocessor to check its parameters.

## Worked example selected in advance

This example uses seed **{seed}**, **original training data**, **histogram gradient boosting**, and **validation-selected cutoff {cutoff:.2f}**. It is a concrete row of evidence, not a best-performing example selected from the test results.

{table(example_table)}

The count-based expressions above show exactly why each rate takes its value. AP and ROC-AUC instead come from the complete ordering of the scores; one cutoff's TP/FP/FN/TN cannot explain them alone.

The key `hist_gradient_boosting` means scikit-learn's `HistGradientBoostingClassifier`, not `GradientBoostingClassifier`. Its final model uses 150 iterations, learning rate .07 and at most 20 leaves. The selection proxy uses 80 iterations, .08 and 15 leaves; its validation AP must not be confused with the final classifier's test AP. Logistic regression allows up to 1,000 iterations; random forest and extra trees each use 150 trees with minimum leaf size 3. All defaults and seeds, not just these headline settings, are in the saved parameter JSON and frozen classifier code.

Follow this example's files:

1. [Original-row split mapping]({link(root / f"seed_{seed}/split_indices.npz")}), key `test`, identifies the real records.
2. [Complete training table]({link(job / "training.csv.gz")}), [fitted preprocessor]({link(root / f"seed_{seed}/preprocessor.joblib")}), [classifier parameters]({link(job / "classifier_parameters.json")}) and [fitted classifier]({link(job / "hist_gradient_boosting.joblib")}) establish what was trained.
3. [Full-precision prediction arrays]({link(prediction)}), keys `test_y` and `test_probability`, are the score source.
4. [Readable per-record decisions]({link(local / "test_predictions.csv.gz")}) show source CSV row/ID, truth, score, and decisions at both cutoffs.
5. [All validation cutoff candidates]({link(local / "validation_thresholds.csv")}) explain why this cutoff was selected.
6. [Ranking contributions]({link(local / "ranking_curve.csv.gz")}) explain the AP and ROC-AUC sums.
7. [Timing events]({link(job / "events.jsonl")}) explain the recorded build/fit durations.

The same structure exists for every other method, classifier and split. [The complete metric ledger]({link(root / "metric_evidence.csv")}) contains {evidence["metric_values"]} individual metric values and their paths/hashes. [The summary ledger]({link(root / "summary_evidence.csv")}) maps all {evidence["summary_cells"]} aggregate score entries to their {len(run["configuration"]["seeds"])} source evidence IDs. [The validation-selection ledger]({link(root / "selection_evidence.csv")}) provides the candidate-setting AP values and their raw validation predictions.

To independently reproduce the example's F1 from the repository root:

```python
import numpy as np
p = np.load({str(prediction)!r})
y, score, cutoff = p['test_y'], p['test_probability'], float(p['threshold'])
predicted = score >= cutoff
tp = ((y == 1) & predicted).sum()
fp = ((y == 0) & predicted).sum()
fn = ((y == 1) & ~predicted).sum()
print(2 * tp / (2 * tp + fp + fn))
```

To reconstruct the ranking measures from their saved contributions:

```python
import pandas as pd
curve = pd.read_csv({str(local / "ranking_curve.csv.gz")!r})
print(curve['ap_contribution'].sum())
print(curve['roc_area_contribution'].sum())
```

## What was rerun and checked

This accepted run regenerated all five synthetic-data variants on the original dataset's five splits, retrained all four classifiers on each complete training dataset and the real-only baseline, and regenerated the validation selection scores. Nothing in the current report is copied from the broader archived comparison. A preliminary logging-only attempt was excluded after a variable-name collision; its files are archived and no scores from it enter this report.

[verification.json]({link(root / "verification.json")}) records model replay, independent scikit-learn checks of our count formulas, raw-label/split checks, complete training-table checks, generator batch-to-epoch loss reconstruction, validation-selection checks and timing checks. [run.json]({link(root / "run.json")}) records command/configuration/versions and hashes of the exact fitting source under `source/`. Later reporting code is hashed in [evidence.json]({link(root / "evidence.json")}).

Labels, prediction arrays and source-row mappings are checked, not inferred from old rounded tables. The report builder refuses changed metric/evidence inputs. File integrity and reproducibility do not by themselves prove a generator is better; that conclusion must follow the matched test results, within the stated settings and exploratory limits.
"""
    (destination / "EVIDENCE.md").write_text(evidence_doc)


def build(root, output):
    root, output = Path(root), Path(output)
    run = json.loads((root / "run.json").read_text())
    if run["status"] not in {"complete", "complete_with_failures"} or "dataset" not in run:
        raise ValueError("Use a completed credit-card comparison run")
    if set(run["methods"]) - set(METHODS):
        raise ValueError("Run contains methods outside the credit-card comparison")
    evidence_path = root / "evidence.json"
    if not evidence_path.exists():
        raise ValueError("Run model verification and creditrisk.evidence before building the report")
    evidence = json.loads(evidence_path.read_text())
    for name, digest in {**evidence["input_hashes"], **evidence["derived_hashes"]}.items():
        if file_hash(root / name) != digest:
            raise ValueError(f"Evidence input or derivative changed: {name}")
    for path, digest in evidence.get("analysis_source_sha256", {}).items():
        if file_hash(path) != digest:
            raise ValueError(f"Analysis source changed after evidence export: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(root / "metrics.csv")
    if set(results.policy) != {"raw_0.5", "validation_f1"}:
        raise ValueError("Unexpected comparison policies")
    chosen = results[results.policy == "validation_f1"]
    rows = []
    for (method, model), group in chosen.groupby(["method", "model"], sort=False):
        row = dict(method=method, model=model)
        for metric in SCORES:
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
                Balanced_accuracy=row.balanced_accuracy_mean,
            )
        )
    full = pd.DataFrame(full)
    main = full[full.Model == MODEL_LABELS["hist_gradient_boosting"]].drop(columns="Model")
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

The prediction models are logistic regression, gradient boosting, random forest and extra trees. **Average precision (AP)** measures how well defaults rise to the top of the risk ranking. **Precision** measures how many flagged cases really default; **recall** measures how many defaults are caught; **F1** balances precision and recall. Higher is better for these scores. ROC-AUC is a complementary ranking measure; accuracy counts overall correct decisions; balanced accuracy gives the two classes equal weight. Results at cutoff 0.5 are also saved in the CSV.

[Metric definitions and reasons](METRICS.md) give the formula, positive class, cutoff dependence and exact source of every measure. [Evidence and a worked example](EVIDENCE.md) trace the numbers back to individual real test records.

## Results

The table below keeps the classifier fixed at histogram gradient boosting so that it compares the training datasets directly. Values are means across the splits. The ± value for AP is the sample standard deviation over the {len(run["configuration"]["seeds"])} split scores (ddof=1), not a confidence interval. Other displayed scores are means; their standard deviations are also in `comparison_summary.csv`. The {len(run["configuration"]["seeds"])} source evidence IDs behind every reported mean and SD are listed in `summary_evidence.csv`.

{table(main)}

Across the four classifiers, **{int((average_deltas > 0).sum())} of {len(average_deltas)} generation-method/classifier combinations improved mean AP** over their matching original-data baseline. The strongest original-data result was {MODEL_LABELS[best_real.model]} at AP **{best_real.average_precision_mean:.4f}**. Small F1 changes should be read alongside AP: changing the decision cutoff can improve F1 without improving the ranking.

![Performance comparison]({os.path.relpath(image, output.parent)})

All four classifiers are listed in [the full comparison](COMPARISON.md). Per-split values are in `metrics.csv`, and `paired_differences.csv` subtracts each classifier's original-data result on the same split. `metric_evidence.csv` provides the exact formula, substituted counts, source prediction/model/training files, hashes and CSV line for each individual score.

## Files and checks

The results are in `{root}`. `dataset_inventory.csv` lists the generated and complete training files. Each method directory also contains its fitted classifier files, generator weights/schema where applicable, selected-checkpoint information and saved validation/test predictions.

{verified} Validation selection scores, the batch-to-epoch training-loss totals, observed timing intervals and fitted classifier parameters were also checked. `run.json` records the command, settings, library versions, start/end times and frozen fitting source.

This accepted run contains {evidence["metric_values"]} individual metric values, {evidence["validation_scores"]} validation-selection scores and {evidence["summary_cells"]} aggregate score entries with evidence links. Counts and time fields are explained separately in `metadata_evidence.csv`. Training loss is an optimization trace, not a test-performance score.

These findings apply to this dataset and these settings. The real test rows were excluded from fitting and setting selection in this run, but the same dataset and split scheme have been examined during earlier iterations; this is exploratory evidence, not a new untouched confirmation cohort. Repeated splits overlap, so their standard deviation is descriptive. The comparison does not establish that any method will improve every credit-risk dataset.
"""
    output.write_text(report)
    write_grounding_docs(root, output.parent, run, results, evidence)
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
