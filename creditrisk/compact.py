"""Losslessly compress large fitted-model files with recoverable manifest updates."""

import argparse
import json
from pathlib import Path

import joblib

from .artifacts import write_json
from .data import file_hash
from .verify import require


def compact_job(job):
    pending = job / ".repacking_pending.json"
    manifest_path = job / "complete.json"
    if pending.exists():
        journal = json.loads(pending.read_text())
    elif (job / "model_storage.json").exists():
        return
    else:
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["artifacts"]:
            require(
                file_hash(job / item["path"]) == item["sha256"],
                f"Artifact changed before compression: {job / item['path']}",
            )
        changes = []
        for path in sorted(job.glob("*.joblib")):
            if path.stat().st_size < 1024 * 1024:
                continue
            temporary = path.with_name(path.name + ".repacking")
            model = joblib.load(path)
            joblib.dump(model, temporary, compress=3)
            del model
            if temporary.stat().st_size >= path.stat().st_size:
                temporary.unlink()
                continue
            changes.append(
                dict(
                    path=path.name,
                    temporary=temporary.name,
                    before_sha256=file_hash(path),
                    after_sha256=file_hash(temporary),
                    before_bytes=path.stat().st_size,
                    after_bytes=temporary.stat().st_size,
                )
            )
        journal = dict(previous_manifest_sha256=file_hash(manifest_path), changes=changes)
        write_json(pending, journal)
    for item in journal["changes"]:
        path, temporary = job / item["path"], job / item["temporary"]
        current_hash = file_hash(path)
        if current_hash == item["before_sha256"]:
            require(
                temporary.exists() and file_hash(temporary) == item["after_sha256"],
                "Missing/corrupt staged compressed model",
            )
            temporary.replace(path)
        else:
            require(
                current_hash == item["after_sha256"], "Model differs from both recorded serialization states"
            )
    write_json(job / "model_storage.json", dict(codec="joblib zlib level 3", **journal))
    manifest = json.loads(manifest_path.read_text())
    manifest["prior_manifest_sha256"] = journal["previous_manifest_sha256"]
    manifest["artifacts"] = [
        dict(path=path.name, bytes=path.stat().st_size, sha256=file_hash(path))
        for path in sorted(job.iterdir())
        if path.is_file()
        and path.name not in {"complete.json", ".repacking_pending.json"}
        and not path.name.endswith(".repacking")
    ]
    write_json(manifest_path, manifest)
    pending.unlink()


def compact(root):
    root = Path(root)
    run = json.loads((root / "run.json").read_text())
    require(
        run["status"] in {"complete", "complete_with_failures"}, "Only completed studies may be compacted"
    )
    for index, manifest in enumerate(sorted(root.glob("seed_*/*/complete.json")), 1):
        compact_job(manifest.parent)
        if index % 10 == 0:
            print(f"Compacted {index} method jobs", flush=True)
    receipts = [json.loads(path.read_text()) for path in root.glob("seed_*/*/model_storage.json")]
    changes = [item for receipt in receipts for item in receipt["changes"]]
    result = dict(
        status="complete",
        compressed_model_files=len(changes),
        before_bytes=sum(item["before_bytes"] for item in changes),
        after_bytes=sum(item["after_bytes"] for item in changes),
        saved_bytes=sum(item["before_bytes"] - item["after_bytes"] for item in changes),
        note="Serialization changed losslessly after fitting; original/new hashes are retained in per-method receipts. Model replay must follow.",
    )
    write_json(root / "storage.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/credit_card")
    args = parser.parse_args()
    compact(args.input)
