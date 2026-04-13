# scripts/06_push_to_hub.py
# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Merge LoRA adapter into base model and publish to HuggingFace Hub
#
# What this script does:
#   1. Loads the base Gemma 4 E4B model in float16 (NOT 4-bit — we need full
#      precision to properly merge the LoRA weights)
#   2. Loads the trained LoRA adapter from outputs/checkpoints/
#   3. Merges adapter weights into the base model and unloads the adapter
#      → Result is a standalone model with no PEFT dependency
#   4. Saves the merged model locally to outputs/merged_model/
#   5. Creates a professional model card (README.md on HuggingFace)
#      → Includes benchmark scores, training details, usage examples
#   6. Pushes everything to your HuggingFace repository
#
# Before running:
#   1. Create a free HuggingFace account at huggingface.co
#   2. Create a new model repository
#   3. Set HF_REPO_ID in configs/training_config.py to "your-username/repo-name"
#   4. Login: huggingface-cli login   (or set HF_TOKEN env variable)
#
# Tip: Also push a GGUF quantized version so people can run it locally with llama.cpp
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import torch
from huggingface_hub import HfApi, login

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.training_config import (
    BASE_MODEL_ID, CHECKPOINT_DIR, MERGED_MODEL_DIR,
    HF_REPO_ID, OUTPUT_DIR,
)


def load_eval_results() -> dict:
    """Load evaluation results from Step 5 for the model card."""
    results_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return {}


def create_model_card(eval_results: dict) -> str:
    """
    Generate a professional HuggingFace model card (README.md).
    A good model card is what makes your HuggingFace repo look credible
    to recruiters and hiring managers reviewing your portfolio.
    """
    ex_score = eval_results.get("execution_accuracy", 0) * 100
    valid_sql = eval_results.get("valid_sql_pct", 0) * 100
    by_diff = eval_results.get("by_difficulty", {})

    simple_ex     = by_diff.get("simple", 0) * 100
    moderate_ex   = by_diff.get("moderate", 0) * 100
    challenging_ex = by_diff.get("challenging", 0) * 100

    model_card = f"""---
language:
  - en
license: apache-2.0
base_model: google/gemma-4-e4b-it
tags:
  - text-to-sql
  - sql
  - nlp
  - fine-tuned
  - qlora
  - lora
  - gemma
  - bird-benchmark
datasets:
  - bird-bench
metrics:
  - execution_accuracy
---

# Gemma 4 E4B — Text-to-SQL (BIRD Benchmark)

Fine-tuned [google/gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it)
for complex natural language to SQL generation using QLoRA on the
[BIRD benchmark](https://bird-bench.github.io/) dataset.

---

## Benchmark Results (BIRD Dev Set — 1,534 examples)

| Metric | Score |
|--------|-------|
| **Execution Accuracy (EX)** | **{ex_score:.2f}%** |
| Valid SQL % | {valid_sql:.2f}% |
| EX — Simple | {simple_ex:.2f}% |
| EX — Moderate | {moderate_ex:.2f}% |
| EX — Challenging | {challenging_ex:.2f}% |

### Comparison vs Frontier Models (BIRD dev, approximate public figures)

| Model | EX on BIRD Dev |
|-------|---------------|
| Gemini 2.5 Pro | ~75–78% |
| GPT-4o | ~72–75% |
| Claude Sonnet 4.6 | ~70–73% |
| **This model (Gemma 4 E4B fine-tuned)** | **{ex_score:.2f}%** |

---

## Model Details

- **Base model:** google/gemma-4-e4b-it (Gemma 4 Effective 4B, instruction-tuned)
- **Fine-tuning method:** QLoRA (4-bit NF4 quantization + LoRA adapters)
- **Training dataset:** BIRD23-train-filtered (6,601 high-quality examples)
- **Training hardware:** NVIDIA T4 GPU (16 GB VRAM)
- **LoRA rank:** 16 | Alpha: 32 | Dropout: 0.05
- **Epochs:** 3 | Effective batch size: 32 | Learning rate: 2e-4

---

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{HF_REPO_ID}"
model     = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

schema = \"\"\"
CREATE TABLE employees (
  emp_id INTEGER PRIMARY KEY,
  name VARCHAR,
  salary FLOAT,
  hire_date DATE,
  dept_id INTEGER,
  status VARCHAR
)
CREATE TABLE departments (
  dept_id INTEGER PRIMARY KEY,
  department_name VARCHAR
)
\"\"\"

question = "Which department had the highest average salary for employees hired after 2020?"
evidence = "salary is annual compensation in USD; active employees have status = 'A'"

prompt = f\"\"\"### Task
Generate a valid SQL query to answer the following question using the database schema below.

### Database Schema
{{schema}}

### Evidence
{{evidence}}

### Question
{{question}}

### SQL Query
\"\"\"

inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True)
sql = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
print(sql)
```

---

## Training Details

### Dataset: BIRD Benchmark
- **Training:** BIRD23-train-filtered — 6,601 curated text-to-SQL pairs
- **Evaluation:** BIRD dev set — 1,534 examples across 95 databases and 37 domains
- BIRD is significantly harder than Spider: databases contain up to 549K rows,
  dirty real-world values, and domain-specific evidence hints

### QLoRA Setup
The base Gemma 4 E4B model is loaded in 4-bit NF4 quantization (frozen).
LoRA adapters are injected into all attention and MLP projection layers:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
Only ~40M parameters (≈0.95% of total) are trained.

### Prompt Format
```
### Task
Generate a valid SQL query to answer the following question using the database schema below.

### Database Schema
[CREATE TABLE statements with sample values]

### Evidence
[domain-specific hints about column meanings and values]

### Question
[natural language question]

### SQL Query
[model generates from here]
```

---

## Limitations
- Optimized for SQLite dialect. May need prompting adjustments for PostgreSQL/MySQL.
- Performance on databases not represented in BIRD may vary.
- Maximum effective schema+question length: 768 tokens.

## Citation
If you use this model, please cite the BIRD benchmark:
```bibtex
@article{{li2024bird,
  title={{Can LLM Already Serve as A Database Interface? A BIG Bench for Large-Scale Database Grounded Text-to-SQLs}},
  author={{Li, Jinyang and Hui, Binyuan and others}},
  journal={{NeurIPS}},
  year={{2024}}
}}
```
"""
    return model_card


def main():
    print("=" * 65)
    print("  STEP 6: Merge LoRA Adapter and Push to HuggingFace")
    print("=" * 65)

    # Validate config
    if "YOUR_HF_USERNAME" in HF_REPO_ID:
        print("\n✗ ERROR: Please set HF_REPO_ID in configs/training_config.py")
        print("  Example: 'your-username/gemma4-e4b-text2sql-bird'")
        return

    # ── Login to HuggingFace ──────────────────────────────────────────────────
    print("\n[1/5] Authenticating with HuggingFace...")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("  Logged in via HF_TOKEN environment variable.")
    else:
        print("  No HF_TOKEN env variable found.")
        print("  Running: huggingface-cli login")
        os.system("huggingface-cli login")

    # ── Merge adapter into base model ─────────────────────────────────────────
    print(f"\n[2/5] Loading base model in float16 for merging...")
    print("  Note: Loading in float16 (not 4-bit) to ensure clean weight merge.")
    print("  This requires more RAM (~16 GB). Use CPU if GPU RAM is insufficient.")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",       # load on CPU for merging — safer for low-VRAM
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    print(f"\n[3/5] Loading LoRA adapter from {CHECKPOINT_DIR} and merging...")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
    model = model.merge_and_unload()   # merges LoRA weights into base — removes PEFT dependency

    print(f"  Saving merged model to {MERGED_MODEL_DIR}...")
    os.makedirs(MERGED_MODEL_DIR, exist_ok=True)
    model.save_pretrained(MERGED_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)
    print("  Merged model saved.")

    # ── Create model card ─────────────────────────────────────────────────────
    print("\n[4/5] Creating HuggingFace model card...")
    eval_results = load_eval_results()
    model_card = create_model_card(eval_results)

    model_card_path = os.path.join(MERGED_MODEL_DIR, "README.md")
    with open(model_card_path, "w") as f:
        f.write(model_card)
    print(f"  Model card saved to {model_card_path}")

    # ── Push to HuggingFace Hub ───────────────────────────────────────────────
    print(f"\n[5/5] Pushing to HuggingFace Hub: {HF_REPO_ID}")
    print("  This will upload ~8–10 GB. May take 10–30 minutes.")

    api = HfApi()

    # Create the repo if it doesn't exist
    try:
        api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
        print(f"  Repository: https://huggingface.co/{HF_REPO_ID}")
    except Exception as e:
        print(f"  Warning creating repo: {e}")

    # Upload entire folder
    api.upload_folder(
        folder_path=MERGED_MODEL_DIR,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Add fine-tuned Gemma 4 E4B text-to-SQL model (BIRD benchmark)",
    )

    print(f"\n── Upload Complete ───────────────────────────────────────────")
    print(f"  Model page  : https://huggingface.co/{HF_REPO_ID}")
    print(f"  Model card  : Includes benchmark scores and usage examples")
    print(f"  Local copy  : {MERGED_MODEL_DIR}")

    print("\n✓ STEP 6 COMPLETE — Your model is live on HuggingFace!\n")
    print("  Next steps for your resume:")
    print("    1. Add the HuggingFace model URL to your resume/portfolio")
    print("    2. Link the W&B training run (public sharing available on wandb.ai)")
    print("    3. Write a short blog post on Medium or LinkedIn documenting the project")
    print("    4. Consider adding GGUF quantized version for local inference:\n"
          "       pip install llama-cpp-python && python -m llama_cpp.convert_hf_to_gguf\n")


if __name__ == "__main__":
    main()
