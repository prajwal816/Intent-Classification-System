"""
src/training/train_intent.py
-----------------------------
Fine-tune BERT for 30-class intent classification using HuggingFace Trainer.

Usage
-----
    python src/training/train_intent.py --config configs/training_config.yaml
    python src/training/train_intent.py --config configs/training_config.yaml --epochs 1 --dry-run
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np

from src.training.dataset import IntentDataset, INTENT_LABELS
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def compute_metrics(eval_pred) -> dict:
    """HuggingFace Trainer-compatible compute_metrics function."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": round(acc, 4), "f1": round(f1, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_intent(
    config_path: str = "configs/training_config.yaml",
    epochs: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    """
    Fine-tune BERT on synthetic 30-class intent dataset.

    Parameters
    ----------
    config_path : str
    epochs : int | None
    dry_run : bool
        Use minimal data/epochs for quick smoke-test.

    Returns
    -------
    dict of evaluation metrics.
    """
    try:
        from transformers import (
            BertTokenizerFast,
            BertForSequenceClassification,
            TrainingArguments,
            Trainer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
        )
    except ImportError as e:
        raise ImportError("Install transformers: pip install transformers") from e

    cfg = load_config(config_path).intent_training
    seed = int(cfg.seed)
    set_seed(seed)

    n_per_class = 5 if dry_run else int(cfg.num_samples_per_class)
    n_epochs = 1 if dry_run else int(epochs or cfg.epochs)
    batch_size = int(cfg.batch_size) if not dry_run else 4
    lr = float(cfg.learning_rate)
    wd = float(cfg.weight_decay)
    max_len = int(cfg.max_length)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = "bert-base-uncased"
    num_labels = len(INTENT_LABELS)
    label2id = {l: i for i, l in enumerate(INTENT_LABELS)}
    id2label = {i: l for i, l in enumerate(INTENT_LABELS)}

    logger.info("Generating intent dataset (n_per_class=%d)...", n_per_class)
    intent_ds = IntentDataset(n_per_class=n_per_class, seed=seed)
    dataset_dict = intent_ds.generate_split(
        val_split=float(cfg.val_split),
        test_split=float(cfg.test_split),
    )
    logger.info(
        "Dataset splits: train=%d, val=%d, test=%d",
        len(dataset_dict["train"]),
        len(dataset_dict["validation"]),
        len(dataset_dict["test"]),
    )

    # ── Tokenise ──────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", base_model)
    tokenizer = BertTokenizerFast.from_pretrained(base_model)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding=False,  # handled by DataCollator
        )

    tokenized = dataset_dict.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Loading BERT model: %s (num_labels=%d)", base_model, num_labels)
    model = BertForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # ── Training Arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=wd,
        warmup_ratio=float(cfg.warmup_ratio),
        evaluation_strategy="steps",
        eval_steps=int(cfg.eval_steps) if not dry_run else 10,
        save_strategy="steps",
        save_steps=int(cfg.save_steps) if not dry_run else 10,
        logging_steps=int(cfg.logging_steps) if not dry_run else 5,
        load_best_model_at_end=bool(cfg.load_best_model_at_end),
        metric_for_best_model=str(cfg.metric_for_best_model),
        greater_is_better=True,
        seed=seed,
        report_to="none",          # disable wandb / mlflow
        dataloader_num_workers=0,
        fp16=False,
    )

    callbacks = []
    if not dry_run:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    logger.info("Starting BERT fine-tuning...")
    trainer.train()

    # ── Evaluate on test set ──────────────────────────────────────────────────
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized["test"])
    logger.info("Test results: %s", test_results)

    # ── Save best model ───────────────────────────────────────────────────────
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save label map
    import json
    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(id2label, f, indent=2)
    logger.info("Label map saved to %s", label_map_path)
    logger.info("Model saved to %s", output_dir)

    return test_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Intent Classification")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results = train_intent(
        config_path=args.config,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )
    print("\n=== Intent Training Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
