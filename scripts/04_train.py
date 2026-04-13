# scripts/04_train.py
# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: QLoRA SFT Training with Weights & Biases
#
# What happens in this script:
#   1.  Verify GPU is available (must be run on T4 in Google Colab)
#   2.  Initialize W&B run — all metrics, hyperparams, and samples are logged there
#   3.  Load Gemma 4 E4B in 4-bit NF4 quantization (~7 GB VRAM)
#   4.  Attach LoRA adapters to attention + MLP layers (~40M trainable params)
#   5.  Load the formatted training dataset from Step 3
#   6.  Run SFT training for 3 epochs using SFTTrainer
#       - Loss logged to W&B every 25 steps
#       - Checkpoints saved every 200 steps
#       - Sample predictions logged to W&B every 100 steps
#   7.  Save the final LoRA adapter checkpoint to outputs/checkpoints/
#
# W&B Dashboard will show:
#   - Training loss curve (should decrease from ~2.5 to ~0.4 over 3 epochs)
#   - Learning rate schedule (cosine warmup)
#   - GPU memory usage
#   - Sample SQL predictions vs ground truth
#   - Hyperparameter summary table
#
# Runtime: 4–6 hours on free T4 GPU (3 epochs, 6,601 examples, batch size 32)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import torch
import wandb
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.training_config import (
    BASE_MODEL_ID, WANDB_PROJECT, WANDB_RUN_NAME,
    QUANT_TYPE, USE_DOUBLE_QUANT, COMPUTE_DTYPE,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    NUM_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, LR_SCHEDULER, WARMUP_RATIO, MAX_SEQ_LENGTH,
    OPTIMIZER, SAVE_STEPS, LOGGING_STEPS, SEED,
    TRAIN_DATASET_PATH, CHECKPOINT_DIR,
    INFERENCE_PROMPT_TEMPLATE, MAX_NEW_TOKENS, TEMPERATURE,
)


# ── W&B Sample Logging Callback ───────────────────────────────────────────────

class WandbSamplePredictionCallback(TrainerCallback):
    """
    Every N training steps, generates SQL predictions for 5 dev examples
    and logs them to W&B as a table. This lets you watch the model improve
    throughout training — you can see it go from garbled SQL to correct SQL.
    """

    def __init__(self, model, tokenizer, dev_samples: list, log_every_n_steps: int = 100):
        self.model        = model
        self.tokenizer    = tokenizer
        self.dev_samples  = dev_samples[:5]  # use first 5 dev examples
        self.log_every    = log_every_n_steps
        self._step        = 0

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._step += 1
        if self._step % self.log_every != 0:
            return

        self.model.eval()
        rows = []

        with torch.no_grad():
            for ex in self.dev_samples:
                prompt = INFERENCE_PROMPT_TEMPLATE.format(
                    schema   = ex.get("schema_text", ""),   # we'll add this below
                    evidence = ex.get("evidence", "No evidence."),
                    question = ex["question"],
                )
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                ).to(self.model.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                predicted_sql = generated_text.strip().split(";")[0].strip() + ";"

                rows.append({
                    "step":          state.global_step,
                    "question":      ex["question"],
                    "gold_sql":      ex["gold_sql"],
                    "predicted_sql": predicted_sql,
                    "match":         predicted_sql.strip().lower() == ex["gold_sql"].strip().lower()
                })

        # Log as W&B table
        table = wandb.Table(
            columns=["step", "question", "gold_sql", "predicted_sql", "match"],
            data=[[r["step"], r["question"], r["gold_sql"], r["predicted_sql"], r["match"]]
                  for r in rows]
        )
        wandb.log({"sample_predictions": table}, step=state.global_step)
        self.model.train()


# ── Helper: verify GPU ─────────────────────────────────────────────────────────

def verify_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. This script must run on a GPU.\n"
            "In Google Colab: Runtime → Change runtime type → T4 GPU"
        )
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU : {device_name}")
    print(f"  VRAM: {vram_gb:.1f} GB")
    if vram_gb < 14:
        print("  Warning: Less than 14 GB VRAM. Reduce MAX_SEQ_LENGTH or batch size if OOM.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  STEP 4: QLoRA SFT Training")
    print("=" * 65)

    # ── GPU check ─────────────────────────────────────────────────────────────
    print("\n[1/7] Checking GPU...")
    verify_gpu()

    # ── W&B init ──────────────────────────────────────────────────────────────
    print("\n[2/7] Initializing Weights & Biases...")
    # W&B will prompt for API key on first run — create free account at wandb.ai
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            # Model
            "base_model":      BASE_MODEL_ID,
            "lora_r":          LORA_R,
            "lora_alpha":      LORA_ALPHA,
            "lora_dropout":    LORA_DROPOUT,
            "target_modules":  LORA_TARGET_MODULES,
            # Training
            "epochs":             NUM_EPOCHS,
            "batch_size":         PER_DEVICE_TRAIN_BATCH_SIZE,
            "grad_accum":         GRADIENT_ACCUMULATION_STEPS,
            "effective_batch":    PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "learning_rate":      LEARNING_RATE,
            "lr_scheduler":       LR_SCHEDULER,
            "warmup_ratio":       WARMUP_RATIO,
            "max_seq_length":     MAX_SEQ_LENGTH,
            "optimizer":          OPTIMIZER,
            "quantization":       f"4-bit {QUANT_TYPE}",
            "dataset":            "BIRD23-train-filtered",
            "num_train_examples": 6601,
        },
        tags=["text-to-sql", "qlora", "gemma4", "bird"],
    )
    print(f"  W&B run: {wandb.run.url}")

    # ── Load 4-bit quantized base model ───────────────────────────────────────
    print(f"\n[3/7] Loading {BASE_MODEL_ID} in 4-bit NF4 quantization...")
    print("  This downloads ~4 GB on first run. Subsequent runs use HF cache.")

    compute_dtype = torch.float16 if COMPUTE_DTYPE == "float16" else torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for causal LM training

    used_vram = torch.cuda.memory_allocated(0) / 1e9
    print(f"  Model loaded. VRAM used: {used_vram:.1f} GB")
    wandb.log({"vram_after_model_load_gb": used_vram})

    # ── Attach LoRA adapters ──────────────────────────────────────────────────
    print("\n[4/7] Attaching LoRA adapters...")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total
    print(f"  Trainable params : {trainable:,}  ({pct:.2f}% of total)")
    print(f"  Frozen params    : {total - trainable:,}")
    wandb.log({
        "trainable_params": trainable,
        "trainable_pct":    pct,
        "total_params":     total,
    })

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\n[5/7] Loading training dataset...")
    train_dataset = load_from_disk(TRAIN_DATASET_PATH)
    print(f"  Loaded {len(train_dataset):,} training examples")

    # Load a few dev examples for the sample prediction callback
    from datasets import load_from_disk as lfd
    from configs.training_config import DEV_DATASET_PATH
    dev_dataset = lfd(DEV_DATASET_PATH)
    dev_samples = [dev_dataset[i] for i in range(5)]  # first 5 for W&B callback

    # ── Training configuration ─────────────────────────────────────────────────
    print("\n[6/7] Configuring trainer...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    training_args = SFTConfig(
        output_dir=CHECKPOINT_DIR,

        # Epochs and batching
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,         # frees activations during backward pass

        # Optimizer
        optim=OPTIMIZER,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,

        # Precision — T4 uses FP16 (no BF16 support)
        fp16=True,
        bf16=False,

        # Sequence
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",           # SFTTrainer reads from this column
        packing=False,                       # set True to pack short examples together (faster)

        # Logging and saving
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,                  # keep only 3 most recent checkpoints
        load_best_model_at_end=False,        # no eval during training (saves memory)
        seed=SEED,

        # W&B
        report_to="wandb",
        run_name=WANDB_RUN_NAME,

        # Misc
        dataloader_num_workers=2,
        remove_unused_columns=False,         # keep our custom columns (db_id, etc.)
        ddp_find_unused_parameters=False,
    )

    # CompletionOnlyLM collator: computes loss ONLY on SQL tokens, not schema/question
    # This is critical — you don't want to penalize the model for the input tokens
    response_template = "### SQL Query\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    # W&B sample prediction callback
    sample_callback = WandbSamplePredictionCallback(
        model=model,
        tokenizer=tokenizer,
        dev_samples=dev_samples,
        log_every_n_steps=100,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collator,
        args=training_args,
        callbacks=[sample_callback],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[7/7] Starting training...")
    print(f"  Epochs          : {NUM_EPOCHS}")
    print(f"  Effective batch : {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Steps per epoch : {len(train_dataset) // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}")
    print(f"  Total steps     : ~{(len(train_dataset) * NUM_EPOCHS) // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}")
    print(f"  Checkpoints in : {CHECKPOINT_DIR}")
    print()

    train_result = trainer.train()

    # ── Log final metrics ─────────────────────────────────────────────────────
    trainer.save_model(CHECKPOINT_DIR)  # save final adapter

    final_metrics = train_result.metrics
    trainer.log_metrics("train", final_metrics)
    trainer.save_metrics("train", final_metrics)

    wandb.log({
        "final/train_loss":        final_metrics.get("train_loss", 0),
        "final/train_runtime_sec": final_metrics.get("train_runtime", 0),
        "final/samples_per_sec":   final_metrics.get("train_samples_per_second", 0),
    })

    used_vram_end = torch.cuda.max_memory_allocated(0) / 1e9
    wandb.log({"vram_peak_gb": used_vram_end})

    wandb.finish()

    print("\n── Training Complete ─────────────────────────────────────────")
    print(f"  Final training loss : {final_metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Peak VRAM used      : {used_vram_end:.1f} GB")
    print(f"  Adapter saved to    : {CHECKPOINT_DIR}")
    print(f"  W&B run             : {wandb.run.url if wandb.run else 'check wandb.ai'}")
    print("\n✓ STEP 4 COMPLETE — Run scripts/05_evaluate.py next.\n")


if __name__ == "__main__":
    main()
