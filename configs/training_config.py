# configs/training_config.py
# ─────────────────────────────────────────────────────────────────────────────
# ALL hyperparameters and paths live here.
# Change values here only — never hardcode in training scripts.
# ─────────────────────────────────────────────────────────────────────────────

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR            = os.path.join(PROJECT_ROOT, "data")
BIRD_DIR            = os.path.join(DATA_DIR, "bird")

# Train bundle: prefer local `Data/train/train/` (train.json + train_databases/) if present;
# else `data/bird/train/` after scripts/01_download_bird.py. Override with BIRD_TRAIN_ROOT.
_LOCAL_TRAIN_ROOT = os.path.join(PROJECT_ROOT, "Data", "train", "train")
_STANDARD_TRAIN_ROOT = os.path.join(BIRD_DIR, "train")

def _dir_nonempty(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        return any(os.scandir(path))
    except OSError:
        return False


def _resolve_train_root() -> str:
    """
    Prefer a local Data/train/train bundle only when it actually contains train.json
    or a non-empty train_databases/. An empty train_databases/ must not win over
    data/bird/train/ (e.g. Colab after downloading train.zip).
    """
    if os.environ.get("BIRD_TRAIN_ROOT"):
        return os.environ["BIRD_TRAIN_ROOT"]

    local_json = os.path.join(_LOCAL_TRAIN_ROOT, "train.json")
    local_dbs = os.path.join(_LOCAL_TRAIN_ROOT, "train_databases")
    std_json = os.path.join(_STANDARD_TRAIN_ROOT, "train.json")
    std_dbs = os.path.join(_STANDARD_TRAIN_ROOT, "train_databases")

    local_ok = os.path.isfile(local_json) or _dir_nonempty(local_dbs)
    std_ok = os.path.isfile(std_json) or _dir_nonempty(std_dbs)

    if local_ok and not std_ok:
        return _LOCAL_TRAIN_ROOT
    if std_ok and not local_ok:
        return _STANDARD_TRAIN_ROOT
    if local_ok and std_ok:
        if os.path.isfile(local_json):
            return _LOCAL_TRAIN_ROOT
        return _STANDARD_TRAIN_ROOT
    return _STANDARD_TRAIN_ROOT

_TRAIN_ROOT = _resolve_train_root()
TRAIN_JSON          = os.path.join(_TRAIN_ROOT, "train.json")
TRAIN_DB_DIR        = os.path.join(_TRAIN_ROOT, "train_databases")
DEV_JSON            = os.path.join(BIRD_DIR, "dev", "dev.json")
DEV_DB_DIR          = os.path.join(BIRD_DIR, "dev", "dev_databases")

# Official filtered training split (6,601 examples) — use for all training / schema db_id collection.
FILTERED_TRAIN_HF_ID = "birdsql/bird23-train-filtered"

SCHEMA_CACHE        = os.path.join(DATA_DIR, "schemas.json")       # saved after Step 2
TRAIN_DATASET_PATH  = os.path.join(DATA_DIR, "train_dataset")      # HF dataset after Step 3
DEV_DATASET_PATH    = os.path.join(DATA_DIR, "dev_dataset")        # HF dataset after Step 3

OUTPUT_DIR          = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR      = os.path.join(OUTPUT_DIR, "checkpoints")
MERGED_MODEL_DIR    = os.path.join(OUTPUT_DIR, "merged_model")


# ── Model ─────────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "google/gemma-4-e4b-it"    # instruction-tuned Gemma 4 E4B
HF_REPO_ID    = "KARAN6550/gemma4-e4b-text2sql-bird"  # ← change this


# ── W&B ──────────────────────────────────────────────────────────────────────

WANDB_PROJECT = "gemma4-text2sql"
WANDB_RUN_NAME = "gemma4-e4b-bird-qlora-v1"


# ── QLoRA / Quantization ─────────────────────────────────────────────────────

QUANT_TYPE          = "nf4"     # NF4 is better than FP4 for language models
USE_DOUBLE_QUANT    = True      # nested quantization — saves ~0.4 GB extra
COMPUTE_DTYPE       = "float16" # T4 does not support bfloat16


# ── LoRA Adapter ─────────────────────────────────────────────────────────────

LORA_R              = 16        # rank — controls adapter capacity
LORA_ALPHA          = 32        # scaling = alpha/r, so 32/16 = 2.0x
LORA_DROPOUT        = 0.05

# Layers to inject LoRA into. Covering attention + MLP gives best results.
LORA_TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ── Training ──────────────────────────────────────────────────────────────────

NUM_EPOCHS                  = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 2     # max that fits on T4 with 768-token sequences
GRADIENT_ACCUMULATION_STEPS = 16   # effective batch = 2 × 16 = 32
LEARNING_RATE               = 2e-4
LR_SCHEDULER                = "cosine"
WARMUP_RATIO                = 0.05
MAX_SEQ_LENGTH              = 768   # BIRD prompts comfortably fit in 768 tokens
OPTIMIZER                   = "paged_adamw_8bit"  # 8-bit Adam saves ~2 GB VRAM
SAVE_STEPS                  = 200
LOGGING_STEPS               = 25
SEED                        = 42


# ── Inference / Generation ────────────────────────────────────────────────────

MAX_NEW_TOKENS  = 256
TEMPERATURE     = 0.1   # low = more deterministic SQL output
DO_SAMPLE       = True


# ── Prompt Template ───────────────────────────────────────────────────────────
# This exact format is used in training, evaluation, and inference.
# Do NOT change it after training begins — model learns this specific structure.

PROMPT_TEMPLATE = """### Task
Generate a valid SQL query to answer the following question using the database schema below.

### Database Schema
{schema}

### Evidence
{evidence}

### Question
{question}

### SQL Query
{sql}"""

# For inference: prompt stops before SQL so model generates it
INFERENCE_PROMPT_TEMPLATE = """### Task
Generate a valid SQL query to answer the following question using the database schema below.

### Database Schema
{schema}

### Evidence
{evidence}

### Question
{question}

### SQL Query
"""


def load_filtered_train_examples() -> list:
    """
    Load the BIRD23 filtered training split from Hugging Face (same 6,601 examples
    as birdsql/bird23-train-filtered). Matches the JSON shape used elsewhere
    (db_id, question, evidence, SQL, optional difficulty).
    """
    from datasets import load_dataset

    ds = load_dataset(FILTERED_TRAIN_HF_ID, split="train")
    examples = []
    for row in ds:
        sql = row.get("SQL")
        if sql is None:
            sql = row.get("sql")
        examples.append({
            "db_id": row["db_id"],
            "question": row["question"],
            "evidence": (row.get("evidence") or "").strip(),
            "SQL": (sql or "").strip(),
            "difficulty": row.get("difficulty", "unknown"),
        })
    return examples


def save_filtered_train_json(dest_path: Optional[str] = None) -> str:
    """Write filtered training examples to train.json (default: TRAIN_JSON)."""
    import json

    path = dest_path or TRAIN_JSON
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = load_filtered_train_examples()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def find_bird_sqlite_path(db_id: str, db_dirs: List[str]) -> Optional[str]:
    """
    Locate {db_id}.sqlite under BIRD train or dev database roots.
    Supports standard layout, nested train_databases/, and db_id/sqlite/ (HF-style).
    """
    import glob

    for base in db_dirs:
        if not base or not os.path.isdir(base):
            continue
        candidates = [
            os.path.join(base, db_id, f"{db_id}.sqlite"),
            os.path.join(base, f"{db_id}.sqlite"),
            os.path.join(base, "train_databases", db_id, f"{db_id}.sqlite"),
            os.path.join(base, db_id, "sqlite", f"{db_id}.sqlite"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        pattern = os.path.join(base, "**", f"{db_id}.sqlite")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None
