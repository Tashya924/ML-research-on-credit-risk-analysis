"""Package current source and saved training inputs for upload to Colab. No fitting."""

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from creditrisk.data import file_hash


def main():
    root = Path(__file__).resolve().parents[1]
    inputs = root / "results/credit_card"
    run = json.loads((inputs / "run.json").read_text())
    if run["status"] != "complete" or run["subset_indices"] is not None:
        raise ValueError("Bundle requires a completed full-data generation run")
    files = []
    for folder in ("creditrisk", "vendor", "colab", "docs"):
        files.extend(
            path
            for path in (root / folder).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".md", ".json", ".ipynb"}
        )
    files.extend(root / name for name in ("UCI_Credit_Card.csv", "requirements.txt", "train.py", "README.md"))
    files.extend(
        inputs / name
        for name in (
            "run.json",
            "verification.json",
            "metrics.csv",
            "comparison_summary.csv",
            "comparison.png",
            "paired_differences.csv",
        )
    )
    for seed in run["configuration"]["seeds"]:
        folder = inputs / f"seed_{seed}"
        files.extend(folder / name for name in ("split_indices.npz", "split.json"))
        for method in run["methods"]:
            job = folder / method
            completed = json.loads((job / "complete.json").read_text())
            expected = next(
                item["sha256"] for item in completed["artifacts"] if item["path"] == "training.csv.gz"
            )
            if file_hash(job / "training.csv.gz") != expected:
                raise ValueError(f"Training table changed: {job}")
            files.extend(job / name for name in ("training.csv.gz", "complete.json"))
    manifest = {str(path.relative_to(root)): file_hash(path) for path in sorted(set(files))}
    output = root / "colab/credit_risk_colab.zip"
    temporary = output.with_suffix(".tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=3) as archive:
        for name in manifest:
            archive.write(root / name, name)
        archive.writestr("COLAB_BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2))
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Bundle CRC check failed")
    temporary.replace(output)
    print(f"Packaged {len(manifest)} files in {output} ({output.stat().st_size / 1024**2:.1f} MiB)")
    print(f"SHA-256 {file_hash(output)}")


if __name__ == "__main__":
    main()
