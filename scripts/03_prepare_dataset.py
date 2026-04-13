# scripts/03_prepare_dataset.py
# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Format BIRD examples into training prompts and save as HuggingFace datasets
#
# What this script does:
#   1. Loads training examples from Hugging Face birdsql/bird23-train-filtered and dev.json
#   2. For each example:
#      - Looks up the schema text from schemas.json (built in Step 2)
#      - Fills in the prompt template: Schema + Evidence + Question + SQL
#      - Creates a single "text" field per example
#   3. Tokenizes every example and checks it fits within MAX_SEQ_LENGTH (768 tokens)
#      - Examples that are too long are truncated (< 2% of BIRD examples)
#   4. Saves formatted datasets to disk as HuggingFace Dataset objects
#   5. Prints statistics: token length distribution, difficulty split, etc.
#
# Why does the format matter?
#   The model learns the EXACT prompt structure. At inference time you must
#   use the IDENTICAL template (minus the SQL part) or performance collapses.
#   The template is defined in configs/training_config.py — never change it
#   after training starts.
#
# Output:
#   data/train_dataset/   ← HuggingFace Dataset (6,601 examples)
#   data/dev_dataset/     ← HuggingFace Dataset (1,534 examples)
#
# Runtime: ~3–5 minutes
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import sys
from collections import Counter

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.training_config import (
    DEV_JSON,
    SCHEMA_CACHE,
    TRAIN_DATASET_PATH,
    DEV_DATASET_PATH,
    BASE_MODEL_ID,
    MAX_SEQ_LENGTH,
    PROMPT_TEMPLATE,
    load_filtered_train_examples,
)


def load_bird_json(json_path: str) -> list:
    """Load and return examples from a BIRD JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def load_schemas(cache_path: str) -> dict:
    """Load the precomputed schema cache from Step 2."""
    with open(cache_path) as f:
        return json.load(f)


def format_example(example: dict, schemas: dict) -> dict:
    """
    Convert one BIRD JSON entry into the prompt format the model will train on.

    BIRD JSON fields used:
        db_id     → look up schema text
        question  → natural language question
        evidence  → domain-specific hint (unique to BIRD)
        SQL       → the ground-truth SQL query
        difficulty → easy / medium / challenging (kept as metadata)

    Returns a dict with:
        text        → the full formatted prompt (input + SQL output)
        db_id       → for debugging
        difficulty  → for stratified analysis
        question    → kept for evaluation
        gold_sql    → kept for evaluation
    """
    db_id    = example["db_id"]
    question = example["question"].strip()
    evidence = example.get("evidence", "").strip()
    sql      = example["SQL"].strip()
    schema   = schemas.get(db_id, f"-- Schema not available for {db_id}")

    # Handle empty evidence (some BIRD examples have none)
    if not evidence:
        evidence = "No additional evidence provided."

    full_text = PROMPT_TEMPLATE.format(
        schema=schema,
        evidence=evidence,
        question=question,
        sql=sql
    )

    return {
        "text":       full_text,
        "db_id":      db_id,
        "difficulty": example.get("difficulty", "unknown"),
        "question":   question,
        "gold_sql":   sql,
    }


def check_token_lengths(examples: list, tokenizer, max_length: int) -> dict:
    """
    Tokenize all examples and compute length statistics.
    Returns a stats dict and flags examples that exceed max_length.
    """
    lengths = []
    too_long = 0

    for ex in tqdm(examples, desc="  Checking token lengths"):
        tokens = tokenizer(
            ex["text"],
            return_tensors=None,
            truncation=False
        )["input_ids"]
        length = len(tokens)
        lengths.append(length)
        if length > max_length:
            too_long += 1

    lengths = np.array(lengths)
    return {
        "min":     int(lengths.min()),
        "max":     int(lengths.max()),
        "mean":    float(lengths.mean()),
        "median":  float(np.median(lengths)),
        "p95":     float(np.percentile(lengths, 95)),
        "p99":     float(np.percentile(lengths, 99)),
        "too_long": too_long,
        "too_long_pct": 100 * too_long / len(lengths),
    }


def print_stats(split_name: str, examples: list, token_stats: dict) -> None:
    """Print a formatted summary of dataset statistics."""
    difficulties = Counter(ex["difficulty"] for ex in examples)
    db_ids = set(ex["db_id"] for ex in examples)

    print(f"\n── {split_name} Split Statistics ──────────────────────────────")
    print(f"  Examples          : {len(examples):,}")
    print(f"  Unique databases  : {len(db_ids)}")
    print(f"  Difficulty split  :")
    for diff in ["simple", "moderate", "challenging", "unknown"]:
        if diff in difficulties:
            pct = 100 * difficulties[diff] / len(examples)
            print(f"    {diff:<15}: {difficulties[diff]:,}  ({pct:.1f}%)")
    print(f"  Token lengths     :")
    print(f"    Min    : {token_stats['min']}")
    print(f"    Median : {token_stats['median']:.0f}")
    print(f"    Mean   : {token_stats['mean']:.0f}")
    print(f"    p95    : {token_stats['p95']:.0f}")
    print(f"    p99    : {token_stats['p99']:.0f}")
    print(f"    Max    : {token_stats['max']}")
    print(f"    > {MAX_SEQ_LENGTH} tokens : {token_stats['too_long']} ({token_stats['too_long_pct']:.1f}%) — will be truncated")
    print("─────────────────────────────────────────────────────────────")


def show_sample_prompt(examples: list, n: int = 1) -> None:
    """Print one formatted example so you can visually verify the format."""
    print(f"\n── Sample Formatted Prompt (example 0) ──────────────────────")
    preview = examples[0]["text"][:800]
    print(preview)
    print("  [... truncated for display ...]")
    print("─────────────────────────────────────────────────────────────\n")


def main():
    print("=" * 65)
    print("  STEP 3: Preparing Training and Dev Datasets")
    print("=" * 65)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("\n[1/4] Loading BIRD JSON files and schema cache...")
    train_raw = load_filtered_train_examples()
    dev_raw   = load_bird_json(DEV_JSON)
    schemas   = load_schemas(SCHEMA_CACHE)
    print(f"  Loaded {len(train_raw):,} training examples")
    print(f"  Loaded {len(dev_raw):,} dev examples")
    print(f"  Loaded {len(schemas):,} database schemas")

    # ── Format examples ───────────────────────────────────────────────────────
    print("\n[2/4] Formatting examples into prompt template...")
    train_formatted = [format_example(ex, schemas) for ex in tqdm(train_raw, desc="  Train")]
    dev_formatted   = [format_example(ex, schemas) for ex in tqdm(dev_raw,   desc="  Dev  ")]

    show_sample_prompt(train_formatted)

    # ── Token length analysis ─────────────────────────────────────────────────
    print("\n[3/4] Analyzing token lengths (loading tokenizer — may take a minute)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        train_token_stats = check_token_lengths(train_formatted, tokenizer, MAX_SEQ_LENGTH)
        dev_token_stats   = check_token_lengths(dev_formatted,   tokenizer, MAX_SEQ_LENGTH)
        print_stats("Training", train_formatted, train_token_stats)
        print_stats("Dev",      dev_formatted,   dev_token_stats)
    except Exception as e:
        print(f"  Warning: Could not load tokenizer for length analysis: {e}")
        print("  Gemma 4 requires transformers>=5.5.0 — upgrade with:")
        print("    pip install 'transformers>=5.5.0'")
        print("  Skipping token statistics. Datasets will still be saved.")

    # ── Save as HuggingFace Datasets ──────────────────────────────────────────
    print("\n[4/4] Saving datasets to disk...")

    # Training dataset: only needs "text" field for SFTTrainer
    # Keep extra fields (question, gold_sql, etc.) for debugging but
    # SFTTrainer only reads the "text" column during training.
    train_ds = Dataset.from_list(train_formatted)
    dev_ds   = Dataset.from_list(dev_formatted)

    os.makedirs(os.path.dirname(TRAIN_DATASET_PATH), exist_ok=True)
    train_ds.save_to_disk(TRAIN_DATASET_PATH)
    dev_ds.save_to_disk(DEV_DATASET_PATH)

    print(f"  ✓ Training dataset saved to: {TRAIN_DATASET_PATH}")
    print(f"  ✓ Dev dataset saved to:      {DEV_DATASET_PATH}")
    print(f"\n  Training dataset features: {train_ds.features}")
    print(f"  Training dataset size    : {train_ds.num_rows:,} rows")

    print("\n✓ STEP 3 COMPLETE — Run scripts/04_train.py next.\n")


if __name__ == "__main__":
    main()
