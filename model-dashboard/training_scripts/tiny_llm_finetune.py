"""
train_tiny_llm_finetune.py

Finetunes a genuinely tiny GPT-2 (sshleifer/tiny-gpt2 — a few MB, HF's own
testing checkpoint) on a small text dataset. Follows the EXACT SAME job
contract as train_dummy_model.py, so it plugs into training_manager.py and
the dashboard's Train tab with zero changes to either.

Install first (same venv the backend/subprocess runs in):
  pip install transformers torch datasets

--data-path here expects either:
  - a .csv with a "text" column (one training example per row), or
  - a path that doesn't exist -> falls back to a small built-in set of
    churn-support example sentences, so the demo always runs without you
    needing to prepare a real dataset first.

Honest expectations: tiny-gpt2 is a TINY, mostly-random-weight architecture.
This proves the pipeline (finetune -> log -> register -> serve), it does not
produce a genuinely capable assistant. A real finetune would swap in an
actual base model (e.g. Qwen2-0.5B) and real GPU time.
"""

import argparse
import os
import time

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


FALLBACK_TEXTS = [
    "Customers with high support ticket volume in the last 90 days are more likely to churn.",
    "A sudden drop in login frequency is an early warning sign of customer disengagement.",
    "Offering a loyalty discount can reduce churn risk for high-value accounts.",
    "Customers who never enable autopay tend to have higher payment friction and churn more.",
    "Low email engagement combined with no recent logins often precedes a cancellation.",
    "Retention outreach is most effective when targeted at customers with declining usage trends.",
    "Long contract tenure is generally associated with lower churn probability.",
    "A recent complaint filed without follow-up resolution increases churn risk significantly.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--base-model", default="sshleifer/tiny-gpt2")
    args = parser.parse_args()

    # Imported here (not top of file) so the dashboard's other endpoints don't
    # require torch/transformers installed if you're not using this script.
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline,
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("training-api-jobs")
    client = MlflowClient(tracking_uri=args.tracking_uri)

    log(f"Starting LLM finetune job {args.job_id} -> registering as '{args.model_name}'")
    log(f"Base model: {args.base_model}")

    # --- Load text data ---
    if os.path.isfile(args.data_path):
        log(f"Loading training text from {args.data_path}")
        df = pd.read_csv(args.data_path)
        if "text" not in df.columns:
            log("ERROR: CSV must have a 'text' column.")
            return
        texts = df["text"].tolist()
    else:
        log(f"No data file at {args.data_path} — using built-in churn-support example sentences.")
        texts = FALLBACK_TEXTS

    log(f"Training on {len(texts)} text examples...")

    # --- Load base model + tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    dataset = Dataset.from_dict({"text": texts})

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    with mlflow.start_run(run_name=f"llm-finetune-job-{args.job_id}") as run:
        mlflow.log_param("training_job_id", args.job_id)
        mlflow.log_param("base_model", args.base_model)
        mlflow.log_param("n_examples", len(texts))
        mlflow.log_param("method", "full finetune (tiny model, CPU-friendly)")

        training_args = TrainingArguments(
            output_dir="/tmp/tiny_llm_finetune_output",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            logging_steps=1,
            save_strategy="no",
            report_to=[],  # we're doing our own MLflow logging, not the HF auto-integration
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )

        log("Beginning training...")
        trainer.train()
        log("Training complete.")

        # --- Wrap as a text-generation pipeline for logging/serving ---
        finetuned_pipeline = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=30, device=-1,  # CPU
        )

        example_input = "Explain why a customer might be at risk of churning:"
        example_output = finetuned_pipeline(example_input)[0]["generated_text"]
        log(f"Sample generation check: {example_output[:80]}...")

        model_info = mlflow.transformers.log_model(
            transformers_model=finetuned_pipeline,
            name="model",
            task="text-generation",
            registered_model_name=args.model_name,
            input_example=example_input,
        )
        version = model_info.registered_model_version

        client.set_model_version_tag(args.model_name, version, "training_job_id", args.job_id)
        client.set_model_version_tag(args.model_name, version, "backing_runtime", "transformers")
        client.set_model_version_tag(args.model_name, version, "base_model", args.base_model)
        client.set_model_version_tag(args.model_name, version, "intended_use",
            "Demo text-generation finetune via the training API — placeholder for a real support-assistant LLM.")
        client.set_model_version_tag(args.model_name, version, "known_limitations",
            "Base model is a tiny testing checkpoint with near-random pretrained weights, not a capable LLM. "
            "Demonstrates the finetune->register->serve pipeline only; outputs are not coherent/reliable.")

        log(f"Registered '{args.model_name}' version {version} (run_id={run.info.run_id})")
        log("DONE")


if __name__ == "__main__":
    main()