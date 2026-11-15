"""Credit-card synthetic-data comparison with validation selection and resumable method jobs."""

import argparse
import json
import multiprocessing
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from .agss import discover, interpolate
from .artifacts import inventory, snapshot_sources, write_json
from .classifiers import make_models, selection_model
from .data import TARGET, classifier_preprocessor, file_hash, fingerprint, load_dataset, partitions
from .diffusion import MixedDiffusion
from .metrics import metrics, threshold

METHODS = (
    "real",
    "agss_interpolation",
    "diffusion_mlp",
    "diffusion_transformer",
    "agss_diffusion_mlp",
    "agss_diffusion_transformer",
)


def log_event(job, event, model=None):
    moment = time.monotonic()
    with (job / "events.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                dict(
                    event=event,
                    model=model,
                    monotonic_seconds=moment,
                    utc=datetime.now(timezone.utc).isoformat(),
                )
            )
            + "\n"
        )
    return moment


def proxy_score(train_x, train_y, synthetic, validation_x, validation_y, seed):
    model = selection_model(seed)
    x = np.concatenate([train_x, synthetic]) if len(synthetic) else train_x
    y = np.r_[train_y, np.ones(len(synthetic), dtype=int)]
    model.fit(x, y)
    probabilities = model.predict_proba(validation_x)[:, 1]
    return float(average_precision_score(validation_y, probabilities)), probabilities


def select_density(
    minority, categorical, count, seed, pre, train_x, train_y, validation_x, validation_y, folder
):
    candidates = []
    choices = []
    for quantile in [0.65, 0.8, 0.95]:
        labels, weights, info = discover(minority, seed, categorical, quantile=quantile)
        if len(weights) == 0:
            candidates.append(dict(quantile=quantile, status="no_clusters"))
            continue
        synthetic = interpolate(minority, categorical, labels, count, seed)
        score, probabilities = proxy_score(
            train_x, train_y, pre.transform(synthetic), validation_x, validation_y, seed
        )
        evidence_file = f"density_q{quantile:.2f}_validation.npz"
        np.savez_compressed(
            folder / evidence_file, validation_y=validation_y, validation_probability=probabilities
        )
        candidates.append(
            dict(quantile=quantile, validation_ap=score, predictions_file=evidence_file, status="ok", **info)
        )
        choices.append((score, labels, weights, info))
    if not choices:
        raise RuntimeError("All prespecified density rules found no clusters")
    return max(choices, key=lambda x: x[0])[1:], candidates


def generate(method, train, dataset, seed, args, pre, arrays, density, folder):
    x, y = train.drop(columns=TARGET), train[TARGET]
    minority = x[y == 1].reset_index(drop=True)
    needed = int((y == 0).sum() - (y == 1).sum())
    metadata = dict(method=method, requested_synthetic_rows=needed)
    if method == "real":
        return minority.iloc[:0], metadata
    labels, weights, info = density
    if method == "agss_interpolation":
        metadata["density"] = info
        return interpolate(minority, dataset.categorical, labels, needed, seed), metadata
    clustered = method.startswith("agss_")
    if clustered:
        keep = labels >= 0
        fitting, cluster_labels, probabilities = (
            minority.loc[keep].reset_index(drop=True),
            labels[keep],
            weights,
        )
    else:
        fitting, cluster_labels, probabilities = minority, None, None
    backbone = "transformer" if method.endswith("transformer") else "mlp"
    generator = MixedDiffusion(backbone, args.steps, args.batch_size, seed).initialize(
        fitting, dataset.categorical, cluster_labels
    )
    candidate_rows, best = [], None
    for epoch in args.epochs:
        generator.train_until(epoch)
        try:
            synthetic = generator.sample(needed, probabilities)
            encoded = pre.transform(synthetic)
            score, validation_probabilities = proxy_score(*arrays[:2], encoded, *arrays[2:4], seed)
            evidence_file = f"checkpoint_{epoch}_validation.npz"
            np.savez_compressed(
                folder / evidence_file,
                validation_y=arrays[3],
                validation_probability=validation_probabilities,
            )
            candidate_rows.append(
                dict(epoch=epoch, validation_ap=score, predictions_file=evidence_file, status="ok")
            )
            if best is None or score > best[0]:
                best = score, epoch, synthetic, generator.state()
        except (ValueError, FloatingPointError) as error:
            candidate_rows.append(dict(epoch=epoch, status="sampling_failed", error=str(error)))
    pd.DataFrame(generator.history).to_csv(folder / "training_loss.csv", index=False)
    pd.DataFrame(generator.batch_history).to_csv(folder / "batch_loss.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(folder / "checkpoint_selection.csv", index=False)
    if best is None:
        raise RuntimeError("No valid diffusion candidate; see checkpoint_selection.csv")
    score, epoch, synthetic, state = best
    torch.save(state, folder / "generator.pt")
    joblib.dump(generator.schema, folder / "schema.joblib")
    metadata.update(
        selected_epoch=epoch,
        validation_selection_ap=score,
        backbone=backbone,
        training_rows=len(fitting),
        density=info if clustered else None,
    )
    return synthetic, metadata


def run_seed(dataset, seed, args, output):
    folder = output / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    (train, validation, test), indices = partitions(dataset.frame, seed)
    np.savez_compressed(
        folder / "split_indices.npz", train=indices[0], validation=indices[1], test=indices[2]
    )
    write_json(
        folder / "split.json",
        {
            name: dict(rows=len(part), defaults=int(part[TARGET].sum()), sha256=fingerprint(part))
            for name, part in zip(["train", "validation", "test"], [train, validation, test], strict=False)
        },
    )
    pre = classifier_preprocessor(dataset.features, dataset.categorical).fit(train.drop(columns=TARGET))
    joblib.dump(pre, folder / "preprocessor.joblib", compress=3)
    arrays = (
        pre.transform(train.drop(columns=TARGET)),
        train[TARGET].to_numpy(),
        pre.transform(validation.drop(columns=TARGET)),
        validation[TARGET].to_numpy(),
        pre.transform(test.drop(columns=TARGET)),
        test[TARGET].to_numpy(),
    )
    needed = int((train[TARGET] == 0).sum() - (train[TARGET] == 1).sum())
    minority = train[train[TARGET] == 1].drop(columns=TARGET).reset_index(drop=True)
    density, candidates = select_density(
        minority, dataset.categorical, needed, seed, pre, *arrays[:4], folder
    )
    write_json(folder / "density_selection.json", dict(candidates=candidates, selected=density[2]))
    write_json(folder / "selection_model.json", selection_model(seed).get_params())
    for method in METHODS:
        job = folder / method
        if (job / "complete.json").exists():
            print(f"{dataset.name} seed={seed} {method}: resume verified job", flush=True)
            saved = json.loads((job / "complete.json").read_text())
            if any(file_hash(job / item["path"]) != item["sha256"] for item in saved["artifacts"]):
                raise RuntimeError("Completed job artifacts changed")
            continue
        job.mkdir(exist_ok=True)
        print(f"{dataset.name} seed={seed} {method}: start", flush=True)
        (job / "events.jsonl").write_text("")
        start = log_event(job, "generation_start")
        try:
            synthetic, metadata = generate(method, train, dataset, seed, args, pre, arrays, density, job)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            write_json(job / "failure.json", dict(error=str(error), method=method))
            print(f"  FAILED: {error}", flush=True)
            continue
        generation_seconds = log_event(job, "generation_end") - start
        metadata.update(synthetic_rows=len(synthetic), generation_seconds=generation_seconds)
        if len(synthetic):
            generated = synthetic.copy()
            generated[TARGET] = 1
            generated.to_csv(job / "synthetic.csv.gz", index=False)
            train_x = np.concatenate([arrays[0], pre.transform(synthetic)])
            train_y = np.r_[arrays[1], np.ones(len(synthetic), dtype=int)]
        else:
            train_x, train_y = arrays[:2]
        training_table = pd.concat([train, generated], ignore_index=True) if len(synthetic) else train
        training_table.to_csv(job / "training.csv.gz", index=False)
        records = []
        models = make_models(seed, args.threads)
        write_json(
            job / "classifier_parameters.json", {name: model.get_params() for name, model in models.items()}
        )
        for name, model in models.items():
            start = log_event(job, "fit_start", name)
            model.fit(train_x, train_y)
            fit_seconds = log_event(job, "fit_end", name) - start
            validation_p = model.predict_proba(arrays[2])[:, 1]
            test_p = model.predict_proba(arrays[4])[:, 1]
            cutoff = threshold(arrays[3], validation_p)
            np.savez_compressed(
                job / f"{name}_predictions.npz",
                validation_y=arrays[3],
                validation_probability=validation_p,
                test_y=arrays[5],
                test_probability=test_p,
                threshold=cutoff,
            )
            joblib.dump(model, job / f"{name}.joblib", compress=3)
            for policy, probabilities, t in [
                ("raw_0.5", test_p, 0.5),
                ("validation_f1", test_p, cutoff),
            ]:
                records.append(
                    dict(
                        seed=seed,
                        method=method,
                        model=name,
                        policy=policy,
                        threshold=t,
                        real_train_rows=len(train),
                        synthetic_rows=len(synthetic),
                        train_rows=len(train_y),
                        test_rows=len(test),
                        generation_seconds=generation_seconds,
                        fit_seconds=fit_seconds,
                        **metrics(arrays[5], probabilities, t),
                    )
                )
            print(
                f"  seed={seed} method={method} model={name}: AP={average_precision_score(arrays[5], test_p):.4f}",
                flush=True,
            )
        pd.DataFrame(records).to_csv(job / "metrics.csv", index=False)
        write_json(job / "method.json", metadata)
        write_json(job / "complete.json", dict(status="complete", artifacts=inventory(job)))


def run_worker(dataset, seed, args, output):
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        run_seed(dataset, seed, args, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/credit_card")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument("--epochs", nargs="+", type=int, default=[50, 150])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use a small subset for a software check")
    args = parser.parse_args()
    if (
        args.epochs != sorted(set(args.epochs))
        or min(args.epochs) < 1
        or args.steps < 2
        or args.threads < 1
        or args.workers < 1
        or args.batch_size < 2
    ):
        parser.error("Invalid epoch checkpoints, steps, batch size or threads")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    dataset = load_dataset()
    subset_indices = None
    if args.smoke:
        subset_indices = (
            dataset.frame.groupby(TARGET, group_keys=False)
            .sample(frac=0.04, random_state=909)
            .index.to_numpy()
        )
        dataset = type(dataset)(
            dataset.name,
            dataset.frame.iloc[subset_indices].reset_index(drop=True),
            dataset.categorical,
            dataset.source,
            dataset.path,
        )
        args.seeds = args.seeds[:1]
        args.epochs = [2, 4]
        args.steps = 10
    output = Path(args.output)
    configuration = {key: value for key, value in vars(args).items() if key != "resume"}
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error("Output exists; choose a fresh path or --resume")
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        manifest = json.loads((output / "run.json").read_text())
        if manifest["configuration"] != configuration:
            parser.error("Resume configuration differs from recorded run")
        for path, digest in manifest.get("fitting_hashes", manifest["source_hashes"]).items():
            if file_hash(Path(path)) != digest:
                parser.error(f"Fitting source changed: {path}")
    else:
        sources = snapshot_sources(output)
        manifest = dict(
            configuration=configuration,
            status="running",
            source_hashes=sources,
            fitting_hashes={
                p: h
                for p, h in sources.items()
                if not p.startswith("tests/")
                and p not in {"creditrisk/report.py", "creditrisk/verify.py", "creditrisk/compact.py"}
            },
            command=[sys.executable, *sys.argv],
            python=platform.python_version(),
            platform=platform.platform(),
            sklearn=sklearn.__version__,
            torch=torch.__version__,
            numpy=np.__version__,
            pandas=pd.__version__,
            joblib=joblib.__version__,
            started_utc=datetime.now(timezone.utc).isoformat(),
            dataset=dict(
                path=dataset.path,
                source=dataset.source,
                rows=len(dataset.frame),
                defaults=int(dataset.frame[TARGET].sum()),
                categorical=list(dataset.categorical),
                sha256=file_hash(dataset.path),
            ),
            methods=list(METHODS),
            subset_indices=None if subset_indices is None else subset_indices.tolist(),
        )
        write_json(output / "run.json", manifest)
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(args.seeds)), mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(run_worker, dataset, seed, args, output) for seed in args.seeds]
        for future in futures:
            future.result()
    metrics_paths = list(output.glob("seed_*/*/metrics.csv"))
    pd.concat([pd.read_csv(path) for path in metrics_paths], ignore_index=True).to_csv(
        output / "metrics.csv", index=False
    )
    failures = [
        dict(path=str(path.relative_to(output)), **json.loads(path.read_text()))
        for path in output.glob("seed_*/*/failure.json")
    ]
    manifest.update(
        status="complete_with_failures" if failures else "complete",
        failures=failures,
        finished_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(output / "run.json", manifest)
    print(f"Finished: {output}; failed methods={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
