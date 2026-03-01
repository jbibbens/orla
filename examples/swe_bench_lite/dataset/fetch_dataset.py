#!/usr/bin/env python3
"""
Fetch princeton-nlp/SWE-bench_Lite from Hugging Face and save each instance
as a JSON file in dataset/<split>/<instance_id>.json.
Format: instance_id, repo, base_commit, problem_statement (as expected by the baseline).
Run from repo root: uv run python examples/swe_bench_lite/fetch_dataset.py
"""
import json
import re
from pathlib import Path

from datasets import load_dataset

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
REPO = "princeton-nlp/SWE-bench_Lite"


def safe_filename(instance_id: str) -> str:
    """Use instance_id as filename; replace any path-unsafe chars."""
    return re.sub(r'[<>:"/\\|?*]', "_", instance_id) + ".json"


def main() -> None:
    print(f"Loading {REPO}...")
    ds = load_dataset(REPO)
    for split in ("dev", "test"):
        out_dir = DATASET_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for row in ds[split]:
            instance = {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "problem_statement": row["problem_statement"],
            }
            path = out_dir / safe_filename(row["instance_id"])
            path.write_text(json.dumps(instance, indent=2), encoding="utf-8")
            count += 1
        print(f"  {split}: {count} instances -> {out_dir}")
    print(f"Done. Total in {DATASET_DIR}")


if __name__ == "__main__":
    main()
