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
# Default: data/bird/train/ — override with env BIRD_TRAIN_ROOT if you placed the
# train zip contents elsewhere (e.g. a manual download under Data/train/...).
_TRAIN_ROOT = os.environ.get("BIRD_TRAIN_ROOT", os.path.join(BIRD_DIR, "train"))
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
