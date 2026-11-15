"""Train neural default classifiers on the saved real and augmented training tables."""

import argparse
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .artifacts import inventory, write_json
from .data import TARGET, classifier_preprocessor, feature_groups, file_hash, fingerprint, load_dataset
from .metrics import metrics, threshold

MODELS = ("mlp_classifier", "transformer_classifier")
SCORES = ("average_precision", "roc_auc", "f1", "precision", "recall", "accuracy", "balanced_accuracy")
SOURCE_FILES = (
    "creditrisk/deep_learning.py",
    "creditrisk/data.py",
    "creditrisk/metrics.py",
    "creditrisk/artifacts.py",
)


class MLPClassifier(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class TransformerClassifier(nn.Module):
    """One learned token per scaled numeric or one-hot encoded input component."""

    def __init__(self, inputs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(inputs, 32) * 0.02)
        self.bias = nn.Parameter(torch.randn(inputs, 32) * 0.02)
        layer = nn.TransformerEncoderLayer(
            32, 4, dim_feedforward=64, dropout=0.2, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 1))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        return self.head(self.encoder(tokens).mean(dim=1)).squeeze(-1)


def make_model(name, inputs):
    return {"mlp_classifier": MLPClassifier, "transformer_classifier": TransformerClassifier}[name](inputs)


def predict(model, x, device, batch_size):
    model.eval()
    with torch.inference_mode():
        batches = [
            model(torch.from_numpy(x[start : start + batch_size]).to(device)).sigmoid().cpu().numpy()
            for start in range(0, len(x), batch_size)
        ]
    probabilities = np.concatenate(batches)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Nonfinite classifier predictions")
    return probabilities


def checked_inputs(root, seed, dataset, methods):
    folder = root / f"seed_{seed}"
    with np.load(folder / "split_indices.npz", allow_pickle=False) as saved:
        indices = [saved[name] for name in ("train", "validation", "test")]
    combined = np.concatenate(indices)
    if sorted(combined.tolist()) != list(range(len(dataset.frame))):
        raise ValueError("Saved splits must partition every original CSV row exactly once")
    parts = [dataset.frame.iloc[index].reset_index(drop=True) for index in indices]
    split = json.loads((folder / "split.json").read_text())
    groups = feature_groups(dataset.frame)
    for i, name in enumerate(("train", "validation", "test")):
        if fingerprint(parts[i]) != split[name]["sha256"]:
            raise ValueError(f"Saved {name} split differs from the original CSV")
        for j in range(i):
            if set(groups[indices[i]]) & set(groups[indices[j]]):
                raise ValueError("Duplicate features cross saved partitions")
    tables = {}
    for method in methods:
        job = folder / method
        completed = json.loads((job / "complete.json").read_text())
        expected = next(
            item["sha256"] for item in completed["artifacts"] if item["path"] == "training.csv.gz"
        )
        if file_hash(job / "training.csv.gz") != expected:
            raise ValueError(f"Saved training table changed: {job}")
        table = pd.read_csv(job / "training.csv.gz")
        real = parts[0]
        if list(table.columns) != list(real.columns):
            raise ValueError("Training feature schema differs")
        pd.testing.assert_frame_equal(table.iloc[: len(real)].reset_index(drop=True), real, check_dtype=False)
        extra = table.iloc[len(real) :]
        expected_rows = 0 if method == "real" else int((real[TARGET] == 0).sum() - real[TARGET].sum())
        if len(extra) != expected_rows or (len(extra) and not extra[TARGET].eq(1).all()):
            raise ValueError("Augmented row count or target differs from the generation recipe")
        if not np.isfinite(table.to_numpy()).all():
            raise ValueError("Training table contains missing or nonfinite values")
        tables[method] = table
    return parts, indices, tables


def fit_job(name, x, y, validation_x, validation_y, job, args, device, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = make_model(name, x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    best_ap, best_epoch, stale, history = -1.0, 0, 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch_y)
        validation_p = predict(model, validation_x, device, args.batch_size)
        ap = float(average_precision_score(validation_y, validation_p))
        history.append(dict(epoch=epoch, training_bce=total / len(y), validation_ap=ap))
        pd.DataFrame(history).to_csv(job / "history.csv", index=False)
        if ap > best_ap:
            best_ap, best_epoch, stale = ap, epoch, 0
            checkpoint = dict(
                model=name,
                inputs=x.shape[1],
                state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                selected_epoch=epoch,
                validation_ap=ap,
            )
            temporary = job / "classifier.tmp.pt"
            torch.save(checkpoint, temporary)
            temporary.replace(job / "classifier.pt")
            np.savez_compressed(job / "selected_validation.npz", y=validation_y, probability=validation_p)
        else:
            stale += 1
        print(f"{job}: epoch={epoch} validation_AP={ap:.6f}", flush=True)
        if stale >= args.patience:
            break
    duration = time.perf_counter() - started
    saved = torch.load(job / "classifier.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    model.to(device)
    return model, dict(
        selected_epoch=best_epoch,
        selected_validation_ap=best_ap,
        epochs_completed=len(history),
        fit_seconds=duration,
        fit_seconds_definition="training, per-epoch validation and checkpoint I/O",
        parameters=sum(parameter.numel() for parameter in model.parameters()),
    )


def summarize(output, expected_jobs):
    completed = sorted(output.glob("seed_*/*/*/complete.json"))
    if len(completed) != expected_jobs:
        raise RuntimeError("Incomplete classifier matrix")
    results = pd.concat([pd.read_csv(path.parent / "metrics.csv") for path in completed], ignore_index=True)
    results.to_csv(output / "metrics.csv", index=False)
    selected = results[results.policy == "validation_f1"]
    selected.groupby(["method", "model"])[list(SCORES)].agg(["mean", "std", "count"]).to_csv(
        output / "comparison_summary.csv"
    )
    baseline = selected[selected.method == "real"][["seed", "model", *SCORES]]
    paired = selected.merge(baseline, on=["seed", "model"], suffixes=("", "_real"), validate="many_to_one")
    for score in SCORES:
        paired[f"delta_{score}"] = paired[score] - paired[f"{score}_real"]
    paired.to_csv(output / "paired_differences.csv", index=False)
    summary = selected.groupby(["method", "model"])[list(SCORES)].mean().reset_index()
    lines = [
        "# Deep learning on enlarged credit-card training datasets",
        "",
        f"Completed {expected_jobs} neural classifier fits on the saved generation datasets.",
        "",
        "The MLP uses hidden layers of 128 and 64 units with ReLU and dropout 0.2. "
        "The Transformer uses one token per scaled numeric or one-hot encoded component, "
        "width 32, four attention heads and two encoder layers with dropout 0.2. "
        "Mean pooling feeds the binary output head.",
        "",
        "Both use unweighted binary cross-entropy, AdamW and the same recipe across datasets. "
        "Real validation AP selects the epoch. Real validation F1 selects the decision cutoff. "
        "The following scores are means over saved splits at that cutoff.",
        "",
        "| Training method | Classifier | AP | ROC-AUC | F1 | Precision | Recall | Accuracy | Balanced accuracy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(str(value) if isinstance(value, str) else f"{value:.4f}" for value in row)
            + " |"
        )
    lines += [
        "",
        "The original-data row is the baseline for the same classifier. "
        "Paired differences subtract that baseline on the same split. "
        "Sample standard deviations and split counts appear in comparison_summary.csv. "
        "Repeated splits overlap, so these summaries describe this experiment.",
        "",
        "Every score comes from a saved predictions.npz file in seed_SEED/METHOD/CLASSIFIER. "
        "It contains real labels, full-precision scores, the selected cutoff and original CSV row positions. "
        "history.csv records all validation APs and training losses. classifier.pt stores the selected weights. "
        "run.json records exact input hashes, training settings and runtime versions. "
        "Frozen source is in source/. The repository docs/METRICS.md defines the seven scores and confusion counts.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/credit_card")
    parser.add_argument("--output", default="results/credit_card_deep_learning")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only", action="store_true", help="Check inputs without fitting models or writing results"
    )
    args = parser.parse_args()
    if (
        min(args.epochs, args.patience, args.batch_size, args.threads) < 1
        or not np.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("Training budgets and learning rate must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        parser.error("Seeds and models must be distinct")
    root, output = Path(args.input).resolve(), Path(args.output).resolve()
    if output == root or root in output.parents:
        parser.error("Choose an output directory outside the saved generation run")
    source = json.loads((root / "run.json").read_text())
    dataset = load_dataset()
    if source["status"] != "complete" or source["subset_indices"] is not None:
        parser.error("A complete full-dataset generation run is required")
    if file_hash(dataset.path) != source["dataset"]["sha256"]:
        parser.error("Original CSV hash differs from the generation run")
    if not set(args.seeds) <= set(source["configuration"]["seeds"]):
        parser.error("Requested seed absent from the saved generation run")
    methods = source["methods"]
    if "real" not in methods:
        parser.error("Original-data baseline is required")
    source_hashes = {path: file_hash(path) for path in SOURCE_FILES}
    input_hashes = {"run.json": file_hash(root / "run.json")}
    for seed in args.seeds:
        checked_inputs(root, seed, dataset, methods)
        folder = f"seed_{seed}"
        paths = [f"{folder}/split_indices.npz", f"{folder}/split.json"]
        paths += [
            f"{folder}/{method}/{name}" for method in methods for name in ("training.csv.gz", "complete.json")
        ]
        input_hashes.update({path: file_hash(root / path) for path in paths})
    if args.validate_only:
        print(
            f"Input checks passed for {len(args.seeds)} splits and {len(methods)} training datasets per split. No models fitted."
        )
        return
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    configuration = {
        key: value for key, value in vars(args).items() if key not in {"resume", "validate_only"}
    }
    configuration["input"], configuration["output"] = str(root), str(output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error("Output exists. Choose a fresh directory or --resume")
    if args.resume:
        manifest = json.loads((output / "run.json").read_text())
        for key, current in (
            ("configuration", configuration),
            ("source_hashes", source_hashes),
            ("input_hashes", input_hashes),
        ):
            if manifest[key] != current:
                parser.error(f"Resume {key} differs")
        if any(
            manifest[key] != value
            for key, value in (
                ("torch", str(torch.__version__)),
                ("sklearn", sklearn.__version__),
                ("numpy", np.__version__),
                ("pandas", pd.__version__),
                ("device", str(device)),
            )
        ):
            parser.error("Resume runtime differs. Use the original runtime or a fresh output")
    else:
        output.mkdir(parents=True, exist_ok=True)
        manifest = dict(
            status="running",
            configuration=configuration,
            input_hashes=input_hashes,
            source_hashes=source_hashes,
            source_dataset_sha256=source["dataset"]["sha256"],
            started_utc=datetime.now(timezone.utc).isoformat(),
            command=[sys.executable, *sys.argv],
            python=platform.python_version(),
            torch=str(torch.__version__),
            sklearn=sklearn.__version__,
            numpy=np.__version__,
            pandas=pd.__version__,
            device=str(device),
            accelerator=torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
            selection="Highest real-validation AP, earliest exact tie. Patience counts epochs without improvement.",
            optimizer="AdamW, weight_decay=0.0001",
            loss="unweighted BCEWithLogitsLoss",
            gradient_clip=1.0,
            note="Fixed model recipes across methods. Repeated splits overlap. GPU results can depend on runtime versions.",
        )
        for path in SOURCE_FILES:
            destination = output / "source" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(path).read_bytes())
        write_json(output / "run.json", manifest)
    for seed in args.seeds:
        (real, validation, test), indices, tables = checked_inputs(root, seed, dataset, methods)
        pre = classifier_preprocessor(real.drop(columns=TARGET)).fit(real.drop(columns=TARGET))
        folder = output / f"seed_{seed}"
        folder.mkdir(exist_ok=True)
        joblib.dump(pre, folder / "preprocessor.joblib", compress=3)
        np.savez_compressed(
            folder / "split_indices.npz", train=indices[0], validation=indices[1], test=indices[2]
        )
        validation_x = pre.transform(validation.drop(columns=TARGET)).astype(np.float32)
        test_x = pre.transform(test.drop(columns=TARGET)).astype(np.float32)
        for method, table in tables.items():
            x = pre.transform(table.drop(columns=TARGET)).astype(np.float32)
            for name in args.models:
                job = folder / method / name
                if (job / "complete.json").exists():
                    completed = json.loads((job / "complete.json").read_text())
                    for artifact in completed["artifacts"]:
                        if file_hash(job / artifact["path"]) != artifact["sha256"]:
                            raise ValueError(f"Completed classifier artifact changed: {job}")
                    print(f"Verified completed job {job}", flush=True)
                    continue
                job.mkdir(parents=True, exist_ok=True)
                model, metadata = fit_job(
                    name,
                    x,
                    table[TARGET].to_numpy(),
                    validation_x,
                    validation[TARGET].to_numpy(),
                    job,
                    args,
                    device,
                    seed,
                )
                validation_p = predict(model, validation_x, device, args.batch_size)
                test_p = predict(model, test_x, device, args.batch_size)
                # Replay saved weights before accepting a completed classifier.
                replay = make_model(name, x.shape[1]).to(device)
                replay.load_state_dict(
                    torch.load(job / "classifier.pt", map_location=device, weights_only=True)["state_dict"]
                )
                np.testing.assert_allclose(
                    predict(replay, test_x, device, args.batch_size), test_p, rtol=1e-6, atol=1e-7
                )
                cutoff = threshold(validation[TARGET], validation_p)
                np.savez_compressed(
                    job / "predictions.npz",
                    validation_y=validation[TARGET].to_numpy(),
                    validation_probability=validation_p,
                    test_y=test[TARGET].to_numpy(),
                    test_probability=test_p,
                    threshold=cutoff,
                    test_csv_row=indices[2],
                )
                rows = [
                    dict(
                        seed=seed,
                        method=method,
                        model=name,
                        policy=policy,
                        threshold=value,
                        real_train_rows=len(real),
                        synthetic_rows=len(table) - len(real),
                        train_rows=len(table),
                        test_rows=len(test),
                        **metadata,
                        **metrics(test[TARGET], test_p, value),
                    )
                    for policy, value in (("raw_0.5", 0.5), ("validation_f1", cutoff))
                ]
                pd.DataFrame(rows).to_csv(job / "metrics.csv", index=False)
                write_json(
                    job / "complete.json",
                    dict(status="complete", replay_passed=True, artifacts=inventory(job)),
                )
                del model, replay
    summarize(output, len(args.seeds) * len(methods) * len(args.models))
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "run.json", manifest)
    print(f"Completed deep learning comparison in {output}")


if __name__ == "__main__":
    main()
