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
import shutil
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

# Filtered training JSON — URL kept for reference; falls back to full set if 404.
FILTERED_TRAIN_JSON_URL = (
    "https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/"
    "main/bird/data/bird23-train-filtered.json"
)

# Expected file/directory names to locate after extraction
_TARGETS = {
    "train.json":      TRAIN_JSON,
    "dev.json":        DEV_JSON,
    "train_databases": os.path.join(BIRD_DIR, "train", "train_databases"),
    "dev_databases":   os.path.join(BIRD_DIR, "dev",   "dev_databases"),
}


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
    """Extract a zip archive, always targeting BIRD_DIR so nested folders land correctly."""
    print(f"  Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    os.remove(zip_path)
    print(f"  Extracted to: {extract_to}")


def fix_extracted_structure() -> None:
    """
    Walk BIRD_DIR and move any misplaced files/directories to their expected
    canonical locations. This handles zip archives that ship with an extra
    top-level folder (e.g. dev.zip → dev/dev/dev.json instead of dev/dev.json).
    """
    # Build a reverse lookup: basename → canonical destination path
    pending = dict(_TARGETS)  # copy so we can pop as we resolve

    for root, dirs, files in os.walk(BIRD_DIR):
        root_path = Path(root)

        # Check files
        for fname in files:
            if fname in pending:
                src = root_path / fname
                dst = Path(pending[fname])
                if src != dst:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    print(f"  Moving {src}\n      → {dst}")
                    shutil.move(str(src), str(dst))
                pending.pop(fname, None)

        # Check directories
        for dname in list(dirs):
            if dname in pending:
                src = root_path / dname
                dst = Path(pending[dname])
                if src.resolve() != dst.resolve():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    print(f"  Moving {src}\n      → {dst}")
                    shutil.move(str(src), str(dst))
                    dirs.remove(dname)  # don't descend into moved dir
                pending.pop(dname, None)


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

    # Create base directories
    os.makedirs(BIRD_DIR, exist_ok=True)
    os.makedirs(os.path.join(BIRD_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(BIRD_DIR, "dev"),   exist_ok=True)

    # ── Download training split ───────────────────────────────────────────────
    print("\n[1/3] Downloading BIRD training split (~3.5 GB)...")
    train_zip = os.path.join(BIRD_DIR, "train.zip")
    train_db_dir = os.path.join(BIRD_DIR, "train", "train_databases")
    if not os.path.exists(TRAIN_JSON) or not os.path.exists(train_db_dir):
        download_with_progress(BIRD_URLS["train"], train_zip)
        # Always extract to BIRD_DIR; the zip ships with a top-level "train/" folder.
        extract_zip(train_zip, BIRD_DIR)
        fix_extracted_structure()
    else:
        print("  Already downloaded. Skipping.")

    # ── Optionally replace with filtered JSON ─────────────────────────────────
    # The BIRD team recommends the filtered 6,601-example subset for fine-tuning.
    # The upstream URL is sometimes unavailable; we fall back to the full set.
    print("\n[2/3] Checking BIRD23-train-filtered JSON (6,601 clean examples)...")
    if os.path.exists(TRAIN_JSON):
        with open(TRAIN_JSON) as f:
            existing = json.load(f)
        if len(existing) > 7000:
            print(f"  Current train.json has {len(existing):,} examples (unfiltered).")
            print("  Attempting to download filtered version (6,601 examples)...")
            try:
                download_with_progress(FILTERED_TRAIN_JSON_URL, TRAIN_JSON)
                with open(TRAIN_JSON) as f:
                    new_count = len(json.load(f))
                print(f"  Replaced with filtered version ({new_count:,} examples).")
            except Exception as e:
                print(f"  Warning: Could not download filtered JSON: {e}")
                print("  Proceeding with full 9,428 examples (no action needed).")
                print("  Tip: Manually download from https://bird-bench.github.io/")
        else:
            print(f"  train.json already has {len(existing):,} examples (filtered). Skipping.")
    else:
        print("  train.json not found — will be resolved after dev download.")

    # ── Download dev split ────────────────────────────────────────────────────
    print("\n[3/3] Downloading BIRD dev split (~0.5 GB)...")
    dev_zip = os.path.join(BIRD_DIR, "dev.zip")
    dev_db_dir = os.path.join(BIRD_DIR, "dev", "dev_databases")
    if not os.path.exists(DEV_JSON) or not os.path.exists(dev_db_dir):
        download_with_progress(BIRD_URLS["dev"], dev_zip)
        # Extract to BIRD_DIR — the zip ships with a top-level "dev/" folder, so
        # extracting to dev_dir would create dev/dev/ nesting.
        extract_zip(dev_zip, BIRD_DIR)
        fix_extracted_structure()
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
