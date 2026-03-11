#!/usr/bin/env python3
"""Download SWE-bench from HuggingFace and write per-instance JSON files
with the gold patch included (needed for oracle context gathering).

Set FULL_SWE_BENCH=1 for full SWE-bench (~2,294 instances, chunked zips).
Default: SWE-bench Lite (~300 instances, single dataset.zip).

Usage:
    pip install datasets
    python prepare_dataset.py                    # Lite (default)
    FULL_SWE_BENCH=1 python prepare_dataset.py   # Full

Outputs (Lite):
    ../dataset/test/<instance_id>.json   (one per instance)
    ../dataset.zip                       (zipped dataset ready for Docker)

Outputs (Full):
    ../dataset/test/<instance_id>.json   (one per instance)
    ../dataset_full_001.zip .. dataset_full_008.zip   (chunked, ~300 instances each)
"""

import json
import os
import zipfile
from pathlib import Path

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "dataset" / "test"
CHUNK_SIZE = 300  # instances per zip for full SWE-bench


def main():
    full_swe_bench = os.environ.get("FULL_SWE_BENCH", "0") == "1"

    if full_swe_bench:
        print("Downloading princeton-nlp/SWE-bench (full) ...")
        ds = load_dataset("princeton-nlp/SWE-bench", split="test")
    else:
        print("Downloading princeton-nlp/SWE-bench_Lite ...")
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for existing in DATASET_DIR.glob("*.json"):
        existing.unlink()

    count = 0
    for row in ds:
        instance = {
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "problem_statement": row["problem_statement"],
            "patch": row["patch"],
        }
        out_path = DATASET_DIR / f"{row['instance_id']}.json"
        out_path.write_text(json.dumps(instance, indent=2, ensure_ascii=False) + "\n")
        count += 1

    print(f"Wrote {count} instance files to {DATASET_DIR}")

    if full_swe_bench:
        # Write chunked zips
        json_files = sorted(DATASET_DIR.glob("*.json"))
        total_size_mb = 0
        for chunk_idx in range(0, len(json_files), CHUNK_SIZE):
            chunk_files = json_files[chunk_idx : chunk_idx + CHUNK_SIZE]
            zip_num = (chunk_idx // CHUNK_SIZE) + 1
            zip_path = SCRIPT_DIR.parent / f"dataset_full_{zip_num:03d}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for json_file in chunk_files:
                    arcname = f"test/{json_file.name}"
                    zf.write(json_file, arcname)
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            print(f"Wrote {zip_path} ({size_mb:.1f} MB)")
        print(f"Total: {total_size_mb:.1f} MB across {(len(json_files) + CHUNK_SIZE - 1) // CHUNK_SIZE} chunks")
    else:
        # Write single dataset.zip
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
