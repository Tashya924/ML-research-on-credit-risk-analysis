"""Replay saved credit-card classifiers and verify the generation/comparison outputs."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics
from threadpoolctl import threadpool_limits

from .artifacts import write_json
from .classifiers import MODEL_NAMES
from .data import TARGET, classifier_preprocessor, feature_groups, file_hash, fingerprint, load_dataset
from .metrics import metrics, threshold


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_validation_score(path, expected_y, expected_score):
    values = np.load(path)
    np.testing.assert_array_equal(values["validation_y"], expected_y)
    require(
        np.isclose(
            skmetrics.average_precision_score(expected_y, values["validation_probability"]), expected_score
        ),
        f"Validation AP mismatch: {path}",
    )


def audit_selection(folder, run, validation_y, train):
    checked = 0
    density = json.loads((folder / "density_selection.json").read_text())
    valid = [item for item in density["candidates"] if item["status"] == "ok"]
    for item in valid:
        check_validation_score(folder / item["predictions_file"], validation_y, item["validation_ap"])
        checked += 1
    winner = max(valid, key=lambda item: item["validation_ap"])
    require(winner["quantile"] == density["selected"]["distance_quantile"], "Wrong density selected")
    require(
        sum(density["selected"]["cluster_counts"]) + density["selected"]["noise_count"]
        == int(train[TARGET].sum()),
        "Density counts do not cover real training defaults",
    )
    loss_epochs = batches = 0
    for method in run["methods"]:
        job = folder / method
        if not (job / "checkpoint_selection.csv").exists():
            continue
        candidates = pd.read_csv(job / "checkpoint_selection.csv")
        valid = candidates[candidates.status == "ok"]
        require(set(candidates.epoch) == set(run["configuration"]["epochs"]), "Missing checkpoint candidate")
        require(
            len(valid) == len(candidates),
            "Failed checkpoint candidate: this run is not accepted as complete evidence",
        )
        for item in valid.itertuples(index=False):
            check_validation_score(job / item.predictions_file, validation_y, item.validation_ap)
            checked += 1
        selected = (
            valid.sort_values("epoch").sort_values("validation_ap", ascending=False, kind="stable").iloc[0]
        )
        meta = json.loads((job / "method.json").read_text())
        require(int(selected.epoch) == meta["selected_epoch"], "Wrong checkpoint selected")
        require(np.isclose(selected.validation_ap, meta["validation_selection_ap"]), "Selected AP mismatch")
        history = pd.read_csv(job / "training_loss.csv")
        batch = pd.read_csv(job / "batch_loss.csv")
        require(
            set(history.epoch) == set(range(1, max(run["configuration"]["epochs"]) + 1)),
            "Incomplete generator epoch history",
        )
        for epoch in history.itertuples(index=False):
            group = batch[batch.epoch == epoch.epoch]
            require(int(group.examples.sum()) == meta["training_rows"], "Batch observation count mismatch")
            for column in ["training_loss", "numeric_mse", "categorical_vb"]:
                require(
                    np.isclose(np.average(group[column], weights=group.examples), getattr(epoch, column)),
                    "Epoch loss mismatch",
                )
        require(
            np.allclose(batch.training_loss, batch.numeric_mse + batch.categorical_vb, atol=1e-6),
            "Loss components mismatch",
        )
        loss_epochs += len(history)
        batches += len(batch)
    return checked, loss_epochs, batches


def independent_metrics(y, probability, cutoff):
    predicted = probability >= cutoff
    counts = skmetrics.confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(
        average_precision=skmetrics.average_precision_score(y, probability),
        roc_auc=skmetrics.roc_auc_score(y, probability),
        precision=skmetrics.precision_score(y, predicted, zero_division=0),
        recall=skmetrics.recall_score(y, predicted, zero_division=0),
        f1=skmetrics.f1_score(y, predicted, zero_division=0),
        accuracy=skmetrics.accuracy_score(y, predicted),
        balanced_accuracy=skmetrics.balanced_accuracy_score(y, predicted),
        **dict(zip(["tn", "fp", "fn", "tp"], counts, strict=True)),
    )


def verify(root, model_predictions=True):
    root = Path(root)
    run = json.loads((root / "run.json").read_text())
    require(run["status"] in {"complete", "complete_with_failures"}, "Run is incomplete")
    source = root / run.get("fit_source_root", "source")
    for path, digest in run["source_hashes"].items():
        require(file_hash(source / path) == digest, f"Fit-time source changed: {path}")
    dataset = load_dataset()
    require(file_hash(dataset.path) == run["dataset"]["sha256"], "Input CSV changed")
    frame = dataset.frame
    if run.get("subset_indices") is not None:
        frame = frame.iloc[run["subset_indices"]].reset_index(drop=True)
    results = pd.read_csv(root / "metrics.csv")
    require(set(results.policy) == {"raw_0.5", "validation_f1"}, "Unexpected evaluation policies")
    record_count = model_count = synthetic_count = failed_count = 0
    validation_scores = loss_epochs = loss_batches = timing_intervals = 0
    all_groups = feature_groups(frame)
    with threadpool_limits(limits=4):
        for seed in run["configuration"]["seeds"]:
            folder = root / f"seed_{seed}"
            indices = np.load(folder / "split_indices.npz")
            combined = np.concatenate([indices[key] for key in ["train", "validation", "test"]])
            require(
                np.array_equal(np.sort(combined), np.arange(len(frame))),
                "Splits must partition the input rows exactly once",
            )
            groups = [set(all_groups[indices[key]]) for key in ["train", "validation", "test"]]
            require(
                all(not groups[i] & groups[j] for i, j in [(0, 1), (0, 2), (1, 2)]),
                "Duplicate groups crossed splits",
            )
            parts = [
                frame.iloc[indices[key]].reset_index(drop=True) for key in ["train", "validation", "test"]
            ]
            split = json.loads((folder / "split.json").read_text())
            for key, part in zip(["train", "validation", "test"], parts, strict=True):
                require(fingerprint(part) == split[key]["sha256"], f"Partition changed: {key}")
            train, validation, test = parts
            checked, epochs, batches = audit_selection(folder, run, validation[TARGET].to_numpy(), train)
            validation_scores += checked
            loss_epochs += epochs
            loss_batches += batches
            pre = joblib.load(folder / "preprocessor.joblib")
            expected_pre = classifier_preprocessor(train.drop(columns=TARGET), dataset.categorical).fit(
                train.drop(columns=TARGET)
            )
            np.testing.assert_allclose(
                pre.transform(train.drop(columns=TARGET)),
                expected_pre.transform(train.drop(columns=TARGET)),
                rtol=1e-12,
                atol=1e-12,
            )
            test_x = pre.transform(test.drop(columns=TARGET)) if model_predictions else None
            for method in run["methods"]:
                job = folder / method
                if not (job / "complete.json").exists():
                    require(
                        (job / "failure.json").exists(), f"Missing method without a failure record: {job}"
                    )
                    failed_count += 1
                    continue
                for item in json.loads((job / "complete.json").read_text())["artifacts"]:
                    require(
                        file_hash(job / item["path"]) == item["sha256"],
                        f"Artifact changed: {job / item['path']}",
                    )
                meta = json.loads((job / "method.json").read_text())
                events = [json.loads(line) for line in (job / "events.jsonl").read_text().splitlines()]
                clock = {(item["event"], item["model"]): item["monotonic_seconds"] for item in events}
                require(
                    np.isclose(
                        meta["generation_seconds"],
                        clock[("generation_end", None)] - clock[("generation_start", None)],
                    ),
                    "Generation timing mismatch",
                )
                timing_intervals += 1
                parameters = json.loads((job / "classifier_parameters.json").read_text())
                if meta["synthetic_rows"]:
                    generated = pd.read_csv(job / "synthetic.csv.gz")
                    require(len(generated) == meta["synthetic_rows"], "Generated row count mismatch")
                    require(list(generated.columns) == list(train.columns), "Generated schema mismatch")
                    require(
                        (generated[TARGET] == 1).all() and not generated.isna().any().any(),
                        "Invalid generated rows",
                    )
                    for column in dataset.categorical:
                        require(generated[column].isin(train[column]).all(), f"Invalid category: {column}")
                    numeric = [c for c in train if c not in dataset.categorical and c != TARGET]
                    require(np.isfinite(generated[numeric].to_numpy()).all(), "Nonfinite generated values")
                    for column in numeric:
                        require(
                            generated[column].between(train[column].min(), train[column].max()).all(),
                            f"Generated bound violation: {column}",
                        )
                    synthetic_count += len(generated)
                training = pd.read_csv(job / "training.csv.gz")
                require(
                    len(training) == len(train) + meta["synthetic_rows"],
                    "Complete training row count mismatch",
                )
                pd.testing.assert_frame_equal(
                    training.iloc[: len(train)].reset_index(drop=True), train, check_dtype=False
                )
                if meta["synthetic_rows"]:
                    pd.testing.assert_frame_equal(
                        training.iloc[len(train) :].reset_index(drop=True), generated, check_dtype=False
                    )
                for model_name in MODEL_NAMES:
                    p = np.load(job / f"{model_name}_predictions.npz")
                    require(
                        set(p.files)
                        == {
                            "validation_y",
                            "validation_probability",
                            "test_y",
                            "test_probability",
                            "threshold",
                        },
                        "Unexpected saved prediction fields",
                    )
                    np.testing.assert_array_equal(p["validation_y"], validation[TARGET])
                    np.testing.assert_array_equal(p["test_y"], test[TARGET])
                    cutoff = threshold(p["validation_y"], p["validation_probability"])
                    require(
                        np.isclose(cutoff, p["threshold"]), "Threshold does not match validation predictions"
                    )
                    if model_predictions:
                        model = joblib.load(job / f"{model_name}.joblib")
                        require(model.get_params() == parameters[model_name], "Classifier parameter mismatch")
                        np.testing.assert_array_equal(model.classes_, [0, 1])
                        np.testing.assert_allclose(
                            model.predict_proba(test_x)[:, 1], p["test_probability"], rtol=1e-9, atol=1e-10
                        )
                        model_count += 1
                    for policy, cut in [("raw_0.5", 0.5), ("validation_f1", cutoff)]:
                        selected = results[
                            (results.seed == seed)
                            & (results.method == method)
                            & (results.model == model_name)
                            & (results.policy == policy)
                        ]
                        require(len(selected) == 1, "Missing or duplicated comparison row")
                        require(
                            np.isclose(
                                selected.iloc[0].fit_seconds,
                                clock[("fit_end", model_name)] - clock[("fit_start", model_name)],
                            ),
                            "Classifier timing mismatch",
                        )
                        require(
                            np.isclose(selected.iloc[0].generation_seconds, meta["generation_seconds"]),
                            "Timing metadata mismatch",
                        )
                        for field, expected in [
                            ("real_train_rows", len(train)),
                            ("synthetic_rows", meta["synthetic_rows"]),
                            ("train_rows", len(training)),
                            ("test_rows", len(test)),
                        ]:
                            require(
                                int(selected.iloc[0][field]) == expected, f"Count metadata mismatch: {field}"
                            )
                        require(np.isclose(selected.iloc[0].threshold, cut), "Reported threshold mismatch")
                        calculated = metrics(p["test_y"], p["test_probability"], cut)
                        independent = independent_metrics(p["test_y"], p["test_probability"], cut)
                        for metric, value in independent.items():
                            require(
                                np.isclose(calculated[metric], value),
                                f"Independent metric check failed: {metric}",
                            )
                        for metric, value in calculated.items():
                            require(np.isclose(selected.iloc[0][metric], value), f"Metric mismatch: {metric}")
                        record_count += 1
                    timing_intervals += 1
    require(record_count == len(results), "Unexpected aggregate rows")
    summary = dict(
        status="passed",
        metric_records_verified=record_count,
        serialized_classifiers_verified=model_count,
        synthetic_rows_verified=synthetic_count,
        failed_methods=failed_count,
        validation_scores_verified=validation_scores,
        training_epochs_verified=loss_epochs,
        training_batches_verified=loss_batches,
        timing_intervals_verified=timing_intervals,
        independently_checked_metric_rows=record_count,
        real_training_preprocessor_verified=True,
        source_snapshot_verified=True,
        dataset_and_split_hashes_verified=True,
        duplicate_group_isolation_verified=True,
    )
    write_json(root / "verification.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/credit_card")
    parser.add_argument("--skip-model-replay", action="store_true")
    args = parser.parse_args()
    verify(args.input, not args.skip_model_replay)
