#!/usr/bin/env python3
"""Download DAG-MATH-Formatted-CoT from HuggingFace and write per-problem JSON files.

Usage:
    pip install datasets
    python prepare_dataset.py

Outputs:
    ../dataset/test/<problem_id>.json   (one per problem)
    ../dataset.zip                      (zipped dataset ready for Docker)
"""

import json
import zipfile
from pathlib import Path

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "dataset" / "test"


def main():
    print("Downloading liminho123/DAG-MATH-Formatted-CoT ...")
    ds = load_dataset("liminho123/DAG-MATH-Formatted-CoT", split="train")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for existing in DATASET_DIR.glob("*.json"):
        existing.unlink()

    count = 0
    for row in ds:
        steps = []
        for step in row["steps"]:
            steps.append({
                "step_id": step["step_id"],
                "edge": step["edge"],
                "direct_dependent_steps": step["direct_dependent_steps"],
                "node": step["node"],
            })

        problem = {
            "problem_id": row["problem_id"],
            "problem_text": row["problem_text"],
            "final_answer": row["final_answer"],
            "difficulty": row["difficulty"],
            "domain": row["domain"],
            "steps": steps,
        }
        out_path = DATASET_DIR / f"{row['problem_id']}.json"
        out_path.write_text(json.dumps(problem, indent=2, ensure_ascii=False) + "\n")
        count += 1

    print(f"Wrote {count} problem files to {DATASET_DIR}")

    zip_path = SCRIPT_DIR.parent / "dataset.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for json_file in sorted(DATASET_DIR.glob("*.json")):
            arcname = f"test/{json_file.name}"
            zf.write(json_file, arcname)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {zip_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
