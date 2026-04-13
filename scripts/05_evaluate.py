# scripts/05_evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Evaluate fine-tuned model on BIRD dev set
#
# Metrics computed:
#   EX  (Execution Accuracy) — primary metric
#     → Run predicted SQL on .sqlite database. Run gold SQL on same database.
#       If result tables match → correct.
#
#   Valid SQL %
#     → What % of generated SQL is syntactically parseable and executes without error.
#
#   EX by Difficulty
#     → Breakdown of EX for simple / moderate / challenging questions.
#
# Results are:
#   1. Printed to terminal
#   2. Logged to W&B (continuing the same run from Step 4 if run_id is given)
#   3. Saved to outputs/eval_results.json
#
# How long does it take?
#   ~45 minutes to 1.5 hours on T4 for all 1,534 dev examples.
#   You can run on a subset first (--subset 100) to get a quick sanity check.
#
# Usage:
#   python scripts/05_evaluate.py               # full 1,534 examples
#   python scripts/05_evaluate.py --subset 100  # quick check on 100 examples
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import sqlite3
import argparse
from collections import defaultdict

import torch
import pandas as pd
import wandb
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.training_config import (
    BASE_MODEL_ID,
    CHECKPOINT_DIR,
    DEV_DATASET_PATH,
    DEV_DB_DIR,
    INFERENCE_PROMPT_TEMPLATE,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    QUANT_TYPE,
    USE_DOUBLE_QUANT,
    COMPUTE_DTYPE,
    WANDB_PROJECT,
    WANDB_RUN_NAME,
    OUTPUT_DIR,
    MAX_SEQ_LENGTH,
    find_bird_sqlite_path,
)


# ── SQL Generation ─────────────────────────────────────────────────────────────

def generate_sql(
    model,
    tokenizer,
    question: str,
    schema: str,
    evidence: str,
) -> str:
    """
    Generate a SQL query from a question, schema, and evidence.
    Uses the same prompt template as training to ensure consistency.
    """
    prompt = INFERENCE_PROMPT_TEMPLATE.format(
        schema=schema,
        evidence=evidence if evidence else "No additional evidence provided.",
        question=question,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the input prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    generated  = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Clean up: take only the first statement, strip trailing text
    sql = generated.split("###")[0].strip()   # stop at next section marker
    sql = sql.split("\n\n")[0].strip()         # stop at double newline
    if not sql.endswith(";"):
        sql += ";"
    return sql


# ── Execution Accuracy ─────────────────────────────────────────────────────────

def execute_sql(db_path: str, sql: str) -> tuple[bool, any]:
    """
    Execute SQL on a SQLite database.
    Returns (success: bool, result: DataFrame or error message).
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        result_df = pd.read_sql_query(sql, conn)
        conn.close()
        return True, result_df
    except Exception as e:
        return False, str(e)


def results_match(df_pred: pd.DataFrame, df_gold: pd.DataFrame) -> bool:
    """
    Check if two result DataFrames are equivalent (order-insensitive).
    Both DataFrames are sorted and compared as sets of rows.
    """
    if df_pred is None or df_gold is None:
        return False
    if set(df_pred.columns) != set(df_gold.columns):
        # Column names differ → not matching
        # But some queries return unnamed columns — be lenient with positional compare
        if df_pred.shape[1] != df_gold.shape[1]:
            return False
        try:
            df_pred_vals = df_pred.values.tolist()
            df_gold_vals = df_gold.values.tolist()
            return sorted(map(str, df_pred_vals)) == sorted(map(str, df_gold_vals))
        except Exception:
            return False
    try:
        # Sort both by all columns before comparing
        cols = sorted(df_pred.columns.tolist())
        pred_sorted = df_pred[cols].sort_values(by=cols).reset_index(drop=True)
        gold_sorted = df_gold[cols].sort_values(by=cols).reset_index(drop=True)
        return pred_sorted.equals(gold_sorted)
    except Exception:
        # Fall back to string comparison
        return str(df_pred.values.tolist()) == str(df_gold.values.tolist())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None,
                        help="Evaluate on first N examples only (for quick testing)")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Resume an existing W&B run ID to add eval metrics")
    args = parser.parse_args()

    print("=" * 65)
    print("  STEP 5: Evaluating on BIRD Dev Set")
    print("=" * 65)

    # ── W&B init ──────────────────────────────────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        id=args.wandb_run_id,          # None = new run, string = resume existing run
        resume="allow",
        name=f"{WANDB_RUN_NAME}_eval",
        tags=["evaluation", "bird-dev", "execution-accuracy"],
    )

    # ── Load fine-tuned model ──────────────────────────────────────────────────
    print(f"\n[1/3] Loading fine-tuned model from {CHECKPOINT_DIR}...")
    compute_dtype = torch.float16 if COMPUTE_DTYPE == "float16" else torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("  Model loaded.")

    # ── Load dev dataset ───────────────────────────────────────────────────────
    print(f"\n[2/3] Loading dev dataset from {DEV_DATASET_PATH}...")
    dev_dataset = load_from_disk(DEV_DATASET_PATH)

    if args.subset:
        dev_dataset = dev_dataset.select(range(min(args.subset, len(dev_dataset))))
        print(f"  Using subset of {len(dev_dataset)} examples.")
    else:
        print(f"  Evaluating all {len(dev_dataset):,} examples.")

    # Load schema cache for generating prompts
    from configs.training_config import SCHEMA_CACHE
    with open(SCHEMA_CACHE) as f:
        schemas = json.load(f)

    # ── Run evaluation ─────────────────────────────────────────────────────────
    print("\n[3/3] Running evaluation...")
    print("  (For each example: generate SQL → execute on .sqlite → compare results)")

    results = []
    difficulty_results = defaultdict(list)
    results_table_rows = []

    for i, ex in enumerate(tqdm(dev_dataset, desc="  Evaluating")):
        db_id      = ex["db_id"]
        question   = ex["question"]
        gold_sql   = ex["gold_sql"]
        difficulty = ex.get("difficulty", "unknown")
        evidence   = ex.get("text", "").split("### Evidence\n")
        evidence   = evidence[1].split("\n### Question")[0].strip() if len(evidence) > 1 else ""

        schema_text = schemas.get(db_id, "")
        db_path     = find_bird_sqlite_path(db_id, [DEV_DB_DIR])

        if db_path is None:
            results.append({"correct": False, "valid_sql": False, "error": "db_not_found"})
            difficulty_results[difficulty].append(False)
            continue

        # Generate SQL
        predicted_sql = generate_sql(model, tokenizer, question, schema_text, evidence)

        # Execute both predicted and gold SQL
        pred_ok, pred_result = execute_sql(db_path, predicted_sql)
        gold_ok, gold_result = execute_sql(db_path, gold_sql)

        # Check execution accuracy
        if not pred_ok:
            correct   = False
            valid_sql = False
            error     = pred_result  # error message
        elif not gold_ok:
            # Gold SQL itself failed — skip this example
            correct   = None
            valid_sql = True
            error     = "gold_sql_failed"
        else:
            correct   = results_match(pred_result, gold_result)
            valid_sql = True
            error     = None

        results.append({
            "db_id":         db_id,
            "question":      question,
            "gold_sql":      gold_sql,
            "predicted_sql": predicted_sql,
            "correct":       correct,
            "valid_sql":     valid_sql,
            "difficulty":    difficulty,
            "error":         error,
        })
        if correct is not None:
            difficulty_results[difficulty].append(correct)

        # Log to W&B table periodically
        results_table_rows.append([
            i, db_id, difficulty, question,
            gold_sql, predicted_sql,
            "✓" if correct else ("??" if correct is None else "✗"),
        ])

        # Log running accuracy every 50 steps
        if (i + 1) % 50 == 0:
            valid_results = [r for r in results if r["correct"] is not None]
            running_ex = sum(r["correct"] for r in valid_results) / len(valid_results) if valid_results else 0
            wandb.log({"running_ex": running_ex * 100}, step=i + 1)

    # ── Compute final metrics ─────────────────────────────────────────────────
    valid_results = [r for r in results if r["correct"] is not None]
    total_valid   = len(valid_results)
    num_correct   = sum(r["correct"] for r in valid_results)
    num_valid_sql = sum(r.get("valid_sql", False) for r in results)

    execution_accuracy = num_correct / total_valid if total_valid > 0 else 0
    valid_sql_pct      = num_valid_sql / len(results) if results else 0

    # Per-difficulty EX
    diff_ex = {}
    for diff, corrects in difficulty_results.items():
        diff_ex[diff] = sum(corrects) / len(corrects) if corrects else 0

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  BIRD Dev Evaluation Results")
    print("=" * 65)
    print(f"  Examples evaluated  : {len(results):,}")
    print(f"  Valid SQL generated : {num_valid_sql:,} ({valid_sql_pct*100:.1f}%)")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  Execution Accuracy  : {execution_accuracy*100:.2f}%   ← headline metric")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  EX by difficulty:")
    for diff in ["simple", "moderate", "challenging"]:
        if diff in diff_ex:
            count = len(difficulty_results[diff])
            print(f"    {diff:<15}: {diff_ex[diff]*100:.2f}%  ({count} examples)")
    print("=" * 65)

    # Context: frontier model baselines on BIRD dev (for your resume comparison)
    print("\n  Baseline comparison (BIRD dev, approximate public figures):")
    print("    GPT-4o            : ~72-75% EX")
    print("    Claude Sonnet 4.6 : ~70-73% EX")
    print("    Gemini 2.5 Pro    : ~75-78% EX")
    print("    Your fine-tuned   :", f"{execution_accuracy*100:.2f}% EX")
    print()

    # ── Log to W&B ────────────────────────────────────────────────────────────
    wandb.log({
        "eval/execution_accuracy":  execution_accuracy * 100,
        "eval/valid_sql_pct":       valid_sql_pct * 100,
        "eval/num_correct":         num_correct,
        "eval/total_evaluated":     total_valid,
    })
    for diff, ex_score in diff_ex.items():
        wandb.log({f"eval/ex_{diff}": ex_score * 100})

    # Log detailed results as W&B table
    eval_table = wandb.Table(
        columns=["idx", "db_id", "difficulty", "question", "gold_sql", "predicted_sql", "result"],
        data=results_table_rows[:200]   # cap at 200 rows for W&B table size limits
    )
    wandb.log({"eval/predictions_table": eval_table})

    # W&B bar chart: EX by difficulty
    difficulty_data = wandb.Table(
        columns=["difficulty", "execution_accuracy"],
        data=[[d, v * 100] for d, v in diff_ex.items()]
    )
    wandb.log({"eval/ex_by_difficulty": wandb.plot.bar(
        difficulty_data, "difficulty", "execution_accuracy",
        title="Execution Accuracy by Difficulty"
    )})

    wandb.finish()

    # ── Save results to disk ───────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "execution_accuracy": execution_accuracy,
            "valid_sql_pct":      valid_sql_pct,
            "by_difficulty":      diff_ex,
            "num_examples":       len(results),
            "predictions":        results[:50],  # save first 50 for inspection
        }, f, indent=2)

    print(f"  Results saved to: {output_path}")
    print("\n✓ STEP 5 COMPLETE — Run scripts/06_push_to_hub.py next.\n")


if __name__ == "__main__":
    main()
