# scripts/01_download_bird.py
# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Download the BIRD dataset (filtered training split + dev split)
#
# What this script does:
#   1. Downloads BIRD23-train-filtered (6,601 high-quality training pairs)
#   2. Downloads BIRD dev set (1,534 evaluation pairs)
#   3. Downloads the corresponding .sqlite database files for both splits
#   4. Verifies the download is complete and prints a summary
#
# Expected output structure after running:
#   data/bird/
#     train/
#       train.json              ← 6,601 NL→SQL pairs
#       train_databases/        ← .sqlite files (one per database)
#     dev/
#       dev.json                ← 1,534 NL→SQL pairs
#       dev_databases/          ← .sqlite files for evaluation
#
# Runtime: ~5–10 minutes depending on connection speed (dataset is ~4 GB)
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import zipfile
import urllib.request
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.training_config import DATA_DIR, BIRD_DIR, TRAIN_JSON, DEV_JSON


# ── Download URLs ─────────────────────────────────────────────────────────────
# BIRD official download links (from bird-bench.github.io)
# If these URLs expire, visit https://bird-bench.github.io/ for updated links.

BIRD_URLS = {
    "train": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
    "dev":   "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
}

# Filtered training JSON (cleaner subset recommended by BIRD team for fine-tuning)
FILTERED_TRAIN_JSON_URL = (
    "https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/"
    "main/bird/data/bird23-train-filtered.json"
)


def download_with_progress(url: str, dest_path: str) -> None:
    """Download a file with a simple progress indicator."""
    print(f"  Downloading: {url}")
    print(f"  Saving to:   {dest_path}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb_done = downloaded / 1e6
            mb_total = total_size / 1e6
            print(f"\r  Progress: {pct}%  ({mb_done:.1f} / {mb_total:.1f} MB)", end="")

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    print()  # newline after progress


def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extract a zip archive."""
    print(f"  Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    os.remove(zip_path)  # remove zip after extraction to save disk space
    print(f"  Extracted to: {extract_to}")


def verify_bird_structure() -> bool:
    """Check that all expected files and directories exist."""
    required = [
        TRAIN_JSON,
        DEV_JSON,
        os.path.join(BIRD_DIR, "train", "train_databases"),
        os.path.join(BIRD_DIR, "dev", "dev_databases"),
    ]
    all_ok = True
    for path in required:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗ MISSING"
        print(f"  [{status}] {path}")
        if not exists:
            all_ok = False
    return all_ok


def print_dataset_summary() -> None:
    """Print counts from the downloaded JSON files."""
    with open(TRAIN_JSON) as f:
        train_data = json.load(f)
    with open(DEV_JSON) as f:
        dev_data = json.load(f)

    train_dbs = set(ex["db_id"] for ex in train_data)
    dev_dbs   = set(ex["db_id"] for ex in dev_data)

    difficulties = {}
    for ex in train_data:
        d = ex.get("difficulty", "unknown")
        difficulties[d] = difficulties.get(d, 0) + 1

    print("\n── Dataset Summary ──────────────────────────────────────────")
    print(f"  Training examples : {len(train_data):,}")
    print(f"  Dev examples      : {len(dev_data):,}")
    print(f"  Training databases: {len(train_dbs)}")
    print(f"  Dev databases     : {len(dev_dbs)}")
    print(f"  Difficulty split  :")
    for diff, count in sorted(difficulties.items()):
        print(f"    {diff:<15}: {count:,}")
    print("─────────────────────────────────────────────────────────────\n")


def main():
    print("=" * 65)
    print("  STEP 1: Downloading BIRD Dataset")
    print("=" * 65)

    # Create directories
    os.makedirs(BIRD_DIR, exist_ok=True)
    train_dir = os.path.join(BIRD_DIR, "train")
    dev_dir   = os.path.join(BIRD_DIR, "dev")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(dev_dir, exist_ok=True)

    # ── Download training split ───────────────────────────────────────────────
    print("\n[1/3] Downloading BIRD training split (~3.5 GB)...")
    train_zip = os.path.join(BIRD_DIR, "train.zip")
    if not os.path.exists(TRAIN_JSON):
        download_with_progress(BIRD_URLS["train"], train_zip)
        extract_zip(train_zip, BIRD_DIR)
    else:
        print("  Already downloaded. Skipping.")

    # ── Replace with filtered JSON ────────────────────────────────────────────
    # The BIRD team recommends using the filtered 6,601-example subset for
    # fine-tuning as it removes ~2,800 noisy/ambiguous training examples.
    print("\n[2/3] Downloading BIRD23-train-filtered JSON (6,601 clean examples)...")
    filtered_json_path = os.path.join(train_dir, "train.json")

    # Check if the current train.json is already filtered
    with open(filtered_json_path) as f:
        existing = json.load(f)

    if len(existing) > 7000:
        # Still the original 9,428 — replace with filtered version
        print(f"  Current train.json has {len(existing):,} examples (unfiltered).")
        print("  Downloading filtered version (6,601 examples)...")
        try:
            download_with_progress(FILTERED_TRAIN_JSON_URL, filtered_json_path)
            print("  Replaced with filtered version.")
        except Exception as e:
            print(f"  Warning: Could not download filtered JSON: {e}")
            print("  Proceeding with full 9,428 examples.")
            print("  Tip: Manually download from https://bird-bench.github.io/")
    else:
        print(f"  train.json already has {len(existing):,} examples (filtered). Skipping.")

    # ── Download dev split ────────────────────────────────────────────────────
    print("\n[3/3] Downloading BIRD dev split (~0.5 GB)...")
    dev_zip = os.path.join(BIRD_DIR, "dev.zip")
    if not os.path.exists(DEV_JSON):
        download_with_progress(BIRD_URLS["dev"], dev_zip)
        extract_zip(dev_zip, dev_dir)
    else:
        print("  Already downloaded. Skipping.")

    # ── Verify and summarise ──────────────────────────────────────────────────
    print("\n── Verifying download structure ─────────────────────────────")
    ok = verify_bird_structure()

    if ok:
        print_dataset_summary()
        print("✓ STEP 1 COMPLETE — Run scripts/02_extract_schemas.py next.\n")
    else:
        print("\n✗ Some files are missing. Check URLs and try again.")
        print("  Alternatively, manually download from: https://bird-bench.github.io/\n")


if __name__ == "__main__":
    main()
