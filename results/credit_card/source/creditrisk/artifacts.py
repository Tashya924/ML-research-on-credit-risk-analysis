"""Atomic metadata, portable source snapshots, and content fingerprints."""

import json
import shutil
from pathlib import Path

from .data import file_hash


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def snapshot_sources(output):
    root = Path(output) / "source"
    for name in ["creditrisk", "vendor", "tests"]:
        shutil.copytree(name, root / name, ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    for name in ["train.py", "requirements.txt", "pyproject.toml"]:
        shutil.copy(name, root / name)
    return {
        str(path.relative_to(root)): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".toml", ".txt", ".json", ".md"}
    }


def inventory(root):
    root = Path(root)
    return [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": file_hash(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"artifact_hashes.json", "analysis.json", "complete.json"}
        and "__pycache__" not in path.parts
    ]
