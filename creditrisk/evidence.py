"""Connect each reported score to raw predictions, formulas, and immutable fit artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import write_json
from .classifiers import MODEL_NAMES
from .data import TARGET, file_hash, load_dataset
from .metrics import THRESHOLDS, metrics

SCORES = ("average_precision", "roc_auc", "f1", "precision", "recall", "accuracy", "balanced_accuracy")
COUNTS = ("tn", "fp", "fn", "tp")
METRIC_GUIDE = {
    "average_precision": (
        "AP",
        "Sum of precision × increase in recall at each distinct descending score.",
        "Primary ranking measure: gives attention to the default class when it is the minority.",
        False,
    ),
    "roc_auc": (
        "ROC-AUC",
        "Sum of trapezoid areas under TPR versus FPR; tied positive/negative scores count half.",
        "Complementary ranking measure; does not replace precision under class imbalance.",
        False,
    ),
    "precision": (
        "Precision",
        "TP / (TP + FP)",
        "Of the flagged defaults, how many actually defaulted?",
        True,
    ),
    "recall": ("Recall", "TP / (TP + FN)", "Of the actual defaults, how many were caught?", True),
    "f1": (
        "Default-class F1",
        "2 TP / (2 TP + FP + FN)",
        "Balances default precision and recall at a stated decision cutoff.",
        True,
    ),
    "accuracy": (
        "Accuracy",
        "(TP + TN) / N",
        "Overall correctness; can conceal weak default detection when non-defaults dominate.",
        True,
    ),
    "balanced_accuracy": (
        "Balanced accuracy",
        "0.5 × [TP/(TP+FN) + TN/(TN+FP)]",
        "Weights the two classes equally rather than by their frequency.",
        True,
    ),
    "tp": ("True positives", "Count(y=1 and score >= cutoff)", "Defaults correctly flagged.", True),
    "fn": ("False negatives", "Count(y=1 and score < cutoff)", "Defaults missed.", True),
    "fp": ("False positives", "Count(y=0 and score >= cutoff)", "Non-defaults incorrectly flagged.", True),
    "tn": ("True negatives", "Count(y=0 and score < cutoff)", "Non-defaults correctly left unflagged.", True),
}


def ranking_curve(y, probability):
    """Ties are grouped. Summing contributions reconstructs sklearn AP and ROC-AUC."""
    frame = pd.DataFrame({"score": probability, "positive": y})
    groups = frame.groupby("score", sort=True).positive.agg(["sum", "size"]).sort_index(ascending=False)
    tp = groups["sum"].cumsum().to_numpy()
    seen = groups["size"].cumsum().to_numpy()
    fp = seen - tp
    positives = int(np.sum(y))
    negatives = len(y) - positives
    precision = tp / seen
    recall = tp / positives
    fpr = fp / negatives
    delta_recall = np.diff(np.r_[0.0, recall])
    roc_piece = np.diff(np.r_[0.0, fpr]) * (recall + np.r_[0.0, recall[:-1]]) / 2
    return pd.DataFrame(
        dict(
            score=groups.index.to_numpy(),
            tp=tp,
            fp=fp,
            precision=precision,
            recall=recall,
            false_positive_rate=fpr,
            delta_recall=delta_recall,
            ap_contribution=delta_recall * precision,
            roc_area_contribution=roc_piece,
        )
    )


def threshold_curve(y, probability):
    rows = []
    for cutoff in THRESHOLDS:
        values = metrics(y, probability, cutoff)
        rows.append(
            dict(threshold=cutoff, **{key: values[key] for key in (*COUNTS, "precision", "recall", "f1")})
        )
    return pd.DataFrame(rows)


def calculation(metric, values):
    tp, fp, fn, tn = (int(values[k]) for k in ["tp", "fp", "fn", "tn"])
    n = tp + fp + fn + tn
    expressions = {
        "tp": str(tp),
        "fp": str(fp),
        "fn": str(fn),
        "tn": str(tn),
        "precision": f"{tp}/({tp}+{fp})",
        "recall": f"{tp}/({tp}+{fn})",
        "f1": f"2*{tp}/(2*{tp}+{fp}+{fn})",
        "accuracy": f"({tp}+{tn})/{n}",
        "balanced_accuracy": f"0.5*({tp}/({tp}+{fn})+{tn}/({tn}+{fp}))",
        "average_precision": "sum(ranking_curve.ap_contribution)",
        "roc_auc": "sum(ranking_curve.roc_area_contribution)",
    }
    return expressions[metric]


def build(root):
    root = Path(root)
    run = json.loads((root / "run.json").read_text())
    verified = json.loads((root / "verification.json").read_text())
    if (
        run["status"] != "complete"
        or verified["status"] != "passed"
        or not verified["serialized_classifiers_verified"]
    ):
        raise ValueError("Full successful model verification is required before evidence export")
    dataset = load_dataset()
    frame = dataset.frame
    raw = pd.read_csv(dataset.path)
    if run.get("subset_indices") is not None:
        subset = np.asarray(run["subset_indices"])
        frame = frame.iloc[subset].reset_index(drop=True)
    else:
        subset = np.arange(len(frame))
    result = pd.read_csv(root / "metrics.csv")
    destination = root / "evidence"
    destination.mkdir(exist_ok=True)
    source_root = (root / run.get("fit_source_root", "source")).resolve()
    data_hash = file_hash(dataset.path)
    evidence_rows = []
    validation_rows = []
    split_rows = []
    for seed in run["configuration"]["seeds"]:
        folder = root / f"seed_{seed}"
        index = np.load(folder / "split_indices.npz")
        index_hash = file_hash(folder / "split_indices.npz")
        pre_hash = file_hash(folder / "preprocessor.joblib")
        for partition in ["train", "validation", "test"]:
            y = frame.iloc[index[partition]][TARGET]
            split_rows.append(
                dict(
                    seed=seed,
                    partition=partition,
                    rows=len(y),
                    defaults=int(y.sum()),
                    nondefaults=int((y == 0).sum()),
                    default_share=float(y.mean()),
                    row_indices_file=str(folder / "split_indices.npz"),
                    row_indices_key=partition,
                )
            )
        density = json.loads((folder / "density_selection.json").read_text())
        for item in density["candidates"]:
            if item["status"] != "ok":
                continue
            prediction = folder / item["predictions_file"]
            p = np.load(prediction)
            score = average_precision_score(p["validation_y"], p["validation_probability"])
            validation_rows.append(
                dict(
                    seed=seed,
                    stage="density_selection",
                    method="agss_interpolation",
                    candidate=item["quantile"],
                    validation_ap=score,
                    selected=item["quantile"] == density["selected"]["distance_quantile"],
                    predictions_file=str(prediction),
                    predictions_sha256=file_hash(prediction),
                    metadata_file=str(folder / "density_selection.json"),
                    truth_key="validation_y",
                    score_key="validation_probability",
                )
            )
        for method in run["methods"]:
            job = folder / method
            training_hash = file_hash(job / "training.csv.gz")
            if (job / "checkpoint_selection.csv").exists():
                selected_epoch = json.loads((job / "method.json").read_text())["selected_epoch"]
                for item in pd.read_csv(job / "checkpoint_selection.csv").itertuples(index=False):
                    if item.status != "ok":
                        continue
                    prediction = job / item.predictions_file
                    p = np.load(prediction)
                    validation_rows.append(
                        dict(
                            seed=seed,
                            stage="generator_checkpoint",
                            method=method,
                            candidate=item.epoch,
                            validation_ap=average_precision_score(
                                p["validation_y"], p["validation_probability"]
                            ),
                            selected=item.epoch == selected_epoch,
                            predictions_file=str(prediction),
                            predictions_sha256=file_hash(prediction),
                            metadata_file=str(job / "checkpoint_selection.csv"),
                            truth_key="validation_y",
                            score_key="validation_probability",
                        )
                    )
            for model in MODEL_NAMES:
                prediction_file = job / f"{model}_predictions.npz"
                p = np.load(prediction_file)
                cutoff = float(p["threshold"])
                path = destination / f"seed_{seed}" / method / model
                path.mkdir(parents=True, exist_ok=True)
                scores = p["test_probability"]
                y = p["test_y"]
                original_rows = subset[index["test"]]
                observations = pd.DataFrame(
                    dict(
                        csv_row_index=original_rows,
                        csv_line_number=original_rows + 2,
                        customer_ID=raw.iloc[original_rows]["ID"].to_numpy(),
                        actual_default=y,
                        default_score=scores,
                        decision_at_0_5=(scores >= 0.5).astype(int),
                        decision_at_validation_threshold=(scores >= cutoff).astype(int),
                    )
                )
                observations.to_csv(path / "test_predictions.csv.gz", index=False, float_format="%.17g")
                curve = ranking_curve(y, scores)
                if not np.isclose(
                    curve.ap_contribution.sum(), average_precision_score(y, scores)
                ) or not np.isclose(curve.roc_area_contribution.sum(), roc_auc_score(y, scores)):
                    raise ValueError("Ranking-curve contribution mismatch")
                curve.to_csv(path / "ranking_curve.csv.gz", index=False, float_format="%.17g")
                thresholds = threshold_curve(p["validation_y"], p["validation_probability"])
                thresholds.to_csv(path / "validation_thresholds.csv", index=False, float_format="%.17g")
                common = dict(
                    seed=seed,
                    method=method,
                    model=model,
                    raw_csv=dataset.path,
                    raw_csv_sha256=data_hash,
                    split_indices_file=str(folder / "split_indices.npz"),
                    split_indices_sha256=index_hash,
                    evaluated_partition="test",
                    truth_key="test_y",
                    score_key="test_probability",
                    predictions_file=str(prediction_file),
                    predictions_sha256=file_hash(prediction_file),
                    readable_predictions=str(path / "test_predictions.csv.gz"),
                    ranking_curve=str(path / "ranking_curve.csv.gz"),
                    threshold_curve=str(path / "validation_thresholds.csv"),
                    training_file=str(job / "training.csv.gz"),
                    training_sha256=training_hash,
                    model_file=str(job / f"{model}.joblib"),
                    model_sha256=file_hash(job / f"{model}.joblib"),
                    preprocessor_file=str(folder / "preprocessor.joblib"),
                    preprocessor_sha256=pre_hash,
                    classifier_parameters=str(job / "classifier_parameters.json"),
                    timing_events=str(job / "events.jsonl"),
                    fit_metric_code=str(source_root / "creditrisk/metrics.py"),
                    test_rows=len(y),
                    test_defaults=int(y.sum()),
                    test_nondefaults=int((y == 0).sum()),
                )
                for policy, t in [("raw_0.5", 0.5), ("validation_f1", cutoff)]:
                    rows = result[
                        (result.seed == seed)
                        & (result.method == method)
                        & (result.model == model)
                        & (result.policy == policy)
                    ]
                    if len(rows) != 1:
                        raise ValueError("Missing or repeated result row")
                    values = metrics(y, scores, t)
                    for metric, value in values.items():
                        if not np.isclose(value, rows.iloc[0][metric], rtol=1e-10, atol=1e-12):
                            raise ValueError(f"Result/array mismatch: {seed} {method} {model} {metric}")
                        title, formula, reason, dependent = METRIC_GUIDE[metric]
                        evidence_rows.append(
                            dict(
                                evidence_id=f"{seed}/{method}/{model}/{policy}/{metric}",
                                policy=policy,
                                metric=metric,
                                value=value,
                                threshold=t,
                                threshold_dependent=dependent,
                                formula=formula,
                                calculation=calculation(metric, values),
                                why=reason,
                                metrics_csv_line=int(rows.index[0]) + 2,
                                **common,
                            )
                        )
    metadata_definitions = {
        "threshold": (
            "Decision cutoff, not a model performance score.",
            "0.5 or lowest validation-F1-maximizing grid point.",
        ),
        "real_train_rows": ("Real training observations.", "len(split_indices[train])"),
        "synthetic_rows": (
            "Additional default observations generated.",
            "len(synthetic.csv.gz); zero for real-only",
        ),
        "train_rows": ("Complete training observations.", "real_train_rows + synthetic_rows"),
        "test_rows": (
            "Real test observations used for all classifiers in this split.",
            "len(split_indices[test])",
        ),
        "generation_seconds": (
            "Observed method-build wall time, including generator training, sampling, validation proxy selection, logging and generator-file writes.",
            "events[generation_end] - events[generation_start]",
        ),
        "fit_seconds": (
            "Observed classifier-fit wall time including start-event logging; excludes preprocessing, predictions and threshold selection.",
            "events[fit_end,model] - events[fit_start,model]",
        ),
    }
    metadata_rows = []
    for index, row in result.iterrows():
        job = root / f"seed_{int(row.seed)}" / row.method
        for column, (why, formula) in metadata_definitions.items():
            metadata_rows.append(
                dict(
                    seed=int(row.seed),
                    method=row.method,
                    model=row.model,
                    policy=row.policy,
                    field=column,
                    value=row[column],
                    why=why,
                    calculation=formula,
                    metrics_csv_line=int(index) + 2,
                    training_file=str(job / "training.csv.gz"),
                    split_indices=str(job.parent / "split_indices.npz"),
                    events_file=str(job / "events.jsonl"),
                    predictions_file=str(job / f"{row.model}_predictions.npz"),
                )
            )
    pd.DataFrame(metadata_rows).to_csv(root / "metadata_evidence.csv", index=False)
    ledger = pd.DataFrame(evidence_rows)
    ledger.to_csv(root / "metric_evidence.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(root / "selection_evidence.csv", index=False)
    pd.DataFrame(split_rows).to_csv(root / "split_evidence.csv", index=False)
    score_rows = result[result.policy == "validation_f1"]
    summary = []
    for (method, model), group in score_rows.groupby(["method", "model"], sort=False):
        for metric in SCORES:
            ids = [f"{int(seed)}/{method}/{model}/validation_f1/{metric}" for seed in group.seed]
            summary.append(
                dict(
                    summary_id=f"{method}/{model}/{metric}",
                    method=method,
                    model=model,
                    metric=metric,
                    mean=group[metric].mean(),
                    sample_sd=group[metric].std(ddof=1),
                    n=len(group),
                    aggregation="unweighted arithmetic mean; sample SD with ddof=1; no pooling",
                    evidence_ids=json.dumps(ids),
                )
            )
    pd.DataFrame(summary).to_csv(root / "summary_evidence.csv", index=False)
    write_json(
        root / "evidence.json",
        dict(
            status="complete",
            metric_values=len(ledger),
            validation_scores=len(validation_rows),
            summary_cells=len(summary),
            data_sha256=data_hash,
            input_hashes={
                name: file_hash(root / name) for name in ["run.json", "metrics.csv", "verification.json"]
            },
            derived_hashes={
                name: file_hash(root / name)
                for name in [
                    "metric_evidence.csv",
                    "metadata_evidence.csv",
                    "selection_evidence.csv",
                    "split_evidence.csv",
                    "summary_evidence.csv",
                ]
            },
            analysis_source_sha256={
                str(p): file_hash(p)
                for p in [Path(__file__), Path("creditrisk/verify.py"), Path("creditrisk/report.py")]
            },
            note="NPZ arrays are the full-precision source. CSV files are readable derivatives. Scores reconstructed from counts and ranking-curve contributions.",
        ),
    )
    print(
        f"Evidence exported: {len(ledger)} metric values, {len(validation_rows)} validation scores, {len(summary)} summary cells."
    )
    return ledger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/credit_card")
    args = parser.parse_args()
    build(args.input)
