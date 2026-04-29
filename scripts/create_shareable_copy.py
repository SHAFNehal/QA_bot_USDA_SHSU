"""
Create a shareable project copy for GitHub/public distribution.

This keeps source code and documentation, while excluding local artifacts:
- datasets / outputs / logs
- model weights / checkpoints
- virtual environments / caches
- secret files (.env, env.hf, tokens)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


EXCLUDE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "env",
    "logs",
    "pretrained_models",
    "data_output",
    "checkpoints",
}

EXCLUDE_FILE_NAMES = {
    "env.hf",
    ".env",
}

EXCLUDE_SUFFIXES = {
    ".out",
    ".err",
    ".log",
    ".jsonl",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
}

EXCLUDE_PREFIXES = (
    ".env.",
    "secrets.",
)


def should_skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = set(rel.parts)
    if parts & EXCLUDE_DIRS:
        return True

    name = path.name
    if name in EXCLUDE_FILE_NAMES:
        return True
    if any(name.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True

    lower_name = name.lower()
    if lower_name.startswith(("run2_", "run3_", "run4_", "fine_tuned_weights")):
        return True

    return False


def copy_project(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source path does not exist: {src}")

    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    def ignore(directory: str, names: list[str]) -> set[str]:
        directory_path = Path(directory)
        ignored: set[str] = set()
        for name in names:
            p = directory_path / name
            if should_skip(p, src):
                ignored.add(name)
        return ignored

    shutil.copytree(src, dst, ignore=ignore)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a clean, shareable copy of the repository."
    )
    parser.add_argument(
        "--source",
        default=".",
        help="Source project directory (default: current directory).",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination directory for the shareable copy (must not exist).",
    )
    args = parser.parse_args()

    src = Path(args.source).resolve()
    dst = Path(args.dest).resolve()

    copy_project(src, dst)
    print(f"Shareable copy created at: {dst}")


if __name__ == "__main__":
    main()
