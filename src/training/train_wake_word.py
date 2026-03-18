"""
src/training/train_wake_word.py
--------------------------------
Full training pipeline for the CNN-GRU wake-word detection model.

Covers:
- Synthetic data generation
- Train/val/test split
- Training loop with early stopping
- Evaluation: TPR, FAR, ROC-AUC
- Checkpoint saving

Usage
-----
    python src/training/train_wake_word.py --config configs/training_config.yaml
    python src/training/train_wake_word.py --config configs/training_config.yaml --epochs 2 --dry-run
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.models.wake_word_model import WakeWordCNNGRU
from src.training.dataset import WakeWordDataset
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.metrics import compute_tpr_far, compute_roc_auc

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    model: WakeWordCNNGRU,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Run evaluation and return metrics dict."""
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            probs = model(inputs).squeeze(1)
            loss = criterion(probs, labels)
            total_loss += loss.item()
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_scores = np.array(all_probs)

    metrics = compute_tpr_far(y_true, y_scores, threshold)
    try:
        roc = compute_roc_auc(y_true, y_scores)
        metrics["roc_auc"] = roc["auc"]
    except Exception:
        metrics["roc_auc"] = 0.0

    metrics["loss"] = round(total_loss / max(len(loader), 1), 4)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_wake_word(
    config_path: str = "configs/training_config.yaml",
    model_config_path: str = "configs/model_config.yaml",
    epochs: Optional[int] = None,
    dry_run: bool = False,
) -> Tuple[WakeWordCNNGRU, dict]:
    """
    Train the CNN-GRU wake-word model on synthetic data.

    Parameters
    ----------
    config_path : str
        Path to training_config.yaml.
    model_config_path : str
        Path to model_config.yaml.
    epochs : int | None
        Override epoch count from config.
    dry_run : bool
        If True, use minimal data (100 samples) and 1 epoch.

    Returns
    -------
    (model, test_metrics)
    """
    train_cfg = load_config(config_path).wake_word_training
    model_cfg = load_config(model_config_path).wake_word_model

    seed = int(train_cfg.seed)
    set_seed(seed)
    device = get_device()
    logger.info("Training device: %s", device)

    # ── Hyper-params ──────────────────────────────────────────────────────────
    n_pos = 50 if dry_run else int(train_cfg.num_samples_positive)
    n_neg = 50 if dry_run else int(train_cfg.num_samples_negative)
    n_epochs = 1 if dry_run else int(epochs or train_cfg.epochs)
    batch_size = int(train_cfg.batch_size)
    lr = float(train_cfg.learning_rate)
    wd = float(train_cfg.weight_decay)
    patience = int(train_cfg.early_stopping_patience)
    val_frac = float(train_cfg.val_split)
    test_frac = float(train_cfg.test_split)
    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / train_cfg.checkpoint_name

    logger.info("Generating synthetic wake-word dataset (pos=%d, neg=%d)...", n_pos, n_neg)
    full_dataset = WakeWordDataset(
        n_positive=n_pos,
        n_negative=n_neg,
        n_mfcc=int(model_cfg.n_mfcc),
        max_frames=int(model_cfg.max_frames),
        seed=seed,
        augment=not dry_run,
    )

    n_total = len(full_dataset)
    n_test = max(1, int(n_total * test_frac))
    n_val = max(1, int(n_total * val_frac))
    n_train = n_total - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Dataset splits: train=%d, val=%d, test=%d", n_train, n_val, n_test)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = WakeWordCNNGRU.from_config(dict(model_cfg)).to(device)
    logger.info("Model parameters: %d", model.count_parameters())

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler_name = str(getattr(train_cfg, "scheduler", "cosine")).lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(getattr(train_cfg, "scheduler_step_size", 10)),
            gamma=float(getattr(train_cfg, "scheduler_gamma", 0.5)),
        )
    else:
        scheduler = None

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            preds = model(batch_inputs).squeeze(1)
            loss = criterion(preds, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        if scheduler:
            scheduler.step()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device)

        logger.info(
            "Epoch %03d/%03d | train_loss=%.4f | val_loss=%.4f | "
            "val_tpr=%.3f | val_far=%.3f | val_auc=%.3f",
            epoch, n_epochs,
            avg_loss,
            val_metrics["loss"],
            val_metrics["tpr"],
            val_metrics["far"],
            val_metrics.get("roc_auc", 0.0),
        )

        # ── Early stopping ────────────────────────────────────────────────────
        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), str(checkpoint_path))
            logger.info("  ✓ Checkpoint saved (val_loss=%.4f)", best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience and not dry_run:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    # ── Load best and evaluate on test ───────────────────────────────────────
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    logger.info(
        "Test Metrics → TPR=%.3f | FAR=%.3f | ROC-AUC=%.3f | F1=%.3f",
        test_metrics["tpr"],
        test_metrics["far"],
        test_metrics.get("roc_auc", 0.0),
        test_metrics.get("f1", 0.0),
    )

    return model, test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from typing import Optional  # local import for type hint in CLI
    parser = argparse.ArgumentParser(description="Train CNN-GRU Wake Word Model")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _, test_metrics = train_wake_word(
        config_path=args.config,
        model_config_path=args.model_config,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )
    print("\n=== Final Test Results ===")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<15}: {v:.4f}")


if __name__ == "__main__":
    from typing import Optional
    main()
