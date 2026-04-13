# scripts/01_download_bird.py
# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Download the BIRD dataset (filtered training split + dev split)
#
# What this script does:
#   1. Downloads BIRD train.zip (full train databases + unfiltered train.json)
#   2. Overwrites train.json with Hugging Face birdsql/bird23-train-filtered (6,601 pairs)
#   3. Downloads BIRD dev set (1,534 evaluation pairs)
#   4. Downloads the corresponding .sqlite database files for both splits
#   5. Verifies the download is complete and prints a summary
#
# Expected output structure after running:
#   Local train bundle (default when present): Data/train/train/
#     train.json                ← 6,601 NL→SQL pairs (HF filtered)
#     train_databases/          ← .sqlite files (one per database)
#   Or after downloading: data/bird/train/ with the same files under train/
#     dev/                      ← dev.json + dev_databases/ (always under data/bird/dev/)
#
# Runtime: ~5–10 minutes depending on connection speed (dataset is ~4 GB)
# ─────────────────────────────────────────────────────────────────────────────

import glob
import os
import json
import shutil
import zipfile
import urllib.request
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.training_config import (
    DATA_DIR,
    BIRD_DIR,
    TRAIN_JSON,
    TRAIN_DB_DIR,
    DEV_JSON,
    save_filtered_train_json,
)


# ── Download URLs ─────────────────────────────────────────────────────────────
# BIRD official download links (from bird-bench.github.io)
# If these URLs expire, visit https://bird-bench.github.io/ for updated links.

BIRD_URLS = {
    "train": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
    "dev":   "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
}

# Expected file/directory names to locate after extraction
_TARGETS = {
    "train.json":      TRAIN_JSON,
    "dev.json":        DEV_JSON,
    "train_databases": TRAIN_DB_DIR,
    "dev_databases":   os.path.join(BIRD_DIR, "dev", "dev_databases"),
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


def normalize_bird_layout() -> None:
    """
    Newer BIRD dev.zip may extract under dev_20240627/ instead of dev/.
    Ensure dev.json and dev_databases/ end up under data/bird/dev/.
    Also ensure train_databases/ ends up under the expected TRAIN_DB_DIR,
    handling any extra nesting introduced by updated train.zip structures.
    """
    # ── dev layout ────────────────────────────────────────────────────────────
    dev_parent = os.path.join(BIRD_DIR, "dev")
    os.makedirs(dev_parent, exist_ok=True)

    if not os.path.isfile(DEV_JSON):
        for p in sorted(glob.glob(os.path.join(BIRD_DIR, "**", "dev.json"), recursive=True)):
            if os.path.normpath(p) == os.path.normpath(DEV_JSON):
                continue
            print(f"  Normalizing dev.json:\n      {p}\n      → {DEV_JSON}")
            shutil.move(p, DEV_JSON)
            break

    dev_db_target = os.path.join(BIRD_DIR, "dev", "dev_databases")
    if not _dir_nonempty(dev_db_target):
        for p in sorted(glob.glob(os.path.join(BIRD_DIR, "**", "dev_databases"), recursive=True)):
            if os.path.normpath(p) == os.path.normpath(dev_db_target):
                continue
            if not os.path.isdir(p) or not _dir_nonempty(p):
                continue
            print(f"  Normalizing dev_databases:\n      {p}\n      → {dev_db_target}")
            if os.path.isdir(dev_db_target):
                shutil.rmtree(dev_db_target)
            shutil.move(p, dev_db_target)
            break

    # ── train layout ──────────────────────────────────────────────────────────
    # The train.zip may ship with an unexpected top-level folder name.
    # Find train_databases/ wherever it landed and move it to TRAIN_DB_DIR.
    if not _dir_nonempty(TRAIN_DB_DIR):
        for p in sorted(glob.glob(os.path.join(BIRD_DIR, "**", "train_databases"), recursive=True)):
            if os.path.normpath(p) == os.path.normpath(TRAIN_DB_DIR):
                continue
            if not os.path.isdir(p) or not _dir_nonempty(p):
                continue
            print(f"  Normalizing train_databases:\n      {p}\n      → {TRAIN_DB_DIR}")
            os.makedirs(os.path.dirname(TRAIN_DB_DIR), exist_ok=True)
            if os.path.isdir(TRAIN_DB_DIR):
                shutil.rmtree(TRAIN_DB_DIR)
            shutil.move(p, TRAIN_DB_DIR)
            break

    # Also find and place train.json if it's missing from the standard location.
    if not os.path.isfile(TRAIN_JSON):
        for p in sorted(glob.glob(os.path.join(BIRD_DIR, "**", "train.json"), recursive=True)):
            if os.path.normpath(p) == os.path.normpath(TRAIN_JSON):
                continue
            print(f"  Normalizing train.json:\n      {p}\n      → {TRAIN_JSON}")
            os.makedirs(os.path.dirname(TRAIN_JSON), exist_ok=True)
            shutil.move(p, TRAIN_JSON)
            break


def _dir_nonempty(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        return any(os.scandir(path))
    except OSError:
        return False


def verify_bird_structure() -> bool:
    """Check that all expected files and directories exist."""
    required = [
        TRAIN_JSON,
        DEV_JSON,
        TRAIN_DB_DIR,
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
    if not os.path.exists(TRAIN_JSON) or not os.path.exists(TRAIN_DB_DIR):
        download_with_progress(BIRD_URLS["train"], train_zip)
        # Always extract to BIRD_DIR; the zip ships with a top-level "train/" folder.
        extract_zip(train_zip, BIRD_DIR)
        fix_extracted_structure()
    else:
        print("  Already downloaded. Skipping.")

    # ── Replace train.json with Hugging Face filtered split (6,601 examples) ─
    # Same dataset as: load_dataset("birdsql/bird23-train-filtered")
    print("\n[2/3] Writing BIRD23-train-filtered JSON from Hugging Face (6,601 examples)...")
    try:
        path = save_filtered_train_json(TRAIN_JSON)
        with open(path) as f:
            n = len(json.load(f))
        print(f"  Saved filtered training labels to: {path}")
        print(f"  Examples: {n:,}")
    except Exception as e:
        print(f"  Error: Could not load Hugging Face dataset: {e}")
        print("  Install: pip install datasets")
        print("  Or set HF_TOKEN if the hub requires authentication.")
        raise

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

    # ── Fix any misplaced train/dev directories after extraction ──────────────
    print("\n── Normalizing BIRD layout (train + dev folder names) ──────")
    normalize_bird_layout()

    # ── Verify and summarise ──────────────────────────────────────────────────
    print("\n── Verifying download structure ─────────────────────────────")
    ok = verify_bird_structure()

    if ok:
        print_dataset_summary()
        print("✓ STEP 1 COMPLETE — Run scripts/02_extract_schemas.py next.\n")
    else:
        print("\n✗ Some files are missing. Check URLs and try again.")
        print("  Alternatively, manually download from: https://bird-bench.github.io/")

        # ── Diagnostic: show what actually landed under BIRD_DIR ──────────────
        print(f"\n── Diagnostic: top-level contents of {BIRD_DIR} ──────────────")
        try:
            import glob as _glob
            all_paths = sorted(_glob.glob(os.path.join(BIRD_DIR, "**"), recursive=True))
            # Show dirs first, then files (limit output to 60 entries)
            shown = 0
            for p in all_paths:
                if shown >= 60:
                    print(f"  ... ({len(all_paths) - shown} more entries)")
                    break
                rel = os.path.relpath(p, BIRD_DIR)
                if os.path.isdir(p):
                    print(f"  [dir]  {rel}/")
                else:
                    size_mb = os.path.getsize(p) / 1e6
                    print(f"  [file] {rel}  ({size_mb:.1f} MB)")
                shown += 1
        except Exception as diag_err:
            print(f"  (diagnostic failed: {diag_err})")
        print()


if __name__ == "__main__":
    main()
