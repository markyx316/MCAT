"""
training/trainer.py — Training loop for MCAT model.
=====================================================
Implements:
  - Huber loss (robust to fat-tailed return outliers)
  - AdamW optimizer with cosine annealing + warmup
  - Early stopping on validation loss
  - Gradient clipping
  - Per-fold fresh initialization
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TRAINING_CONFIG, RANDOM_SEED
from utils import setup_logger, set_seed

logger = setup_logger(__name__)


class HuberLoss(nn.Module):
    """
    Huber loss: MSE for small errors, MAE for large errors.

    For financial returns (target in pp), delta=1.0 means:
      - Errors < 1 pp: quadratic (smooth gradients near optimum)
      - Errors > 1 pp: linear (robust to ±8% COVID crash outliers)
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = pred - target
        abs_error = torch.abs(error)
        quadratic = 0.5 * error ** 2
        linear = self.delta * abs_error - 0.5 * self.delta ** 2
        loss = torch.where(abs_error <= self.delta, quadratic, linear)
        return loss.mean()


def get_cosine_schedule_with_warmup(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float = 1e-6,
):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(min_lr / optimizer.defaults["lr"], 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Handles training, validation, and early stopping for a single fold.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict = None,
        device: torch.device = None,
    ):
        if config is None:
            config = TRAINING_CONFIG
        self.config = config

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = model.to(device)
        self.criterion = HuberLoss(delta=config["huber_delta"])

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            betas=config["betas"],
            weight_decay=config["weight_decay"],
        )

        # Scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_epochs=config["warmup_epochs"],
            total_epochs=config["max_epochs"],
            min_lr=config["min_learning_rate"],
        )

        # Early stopping state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_state_dict = None

    def _batch_to_device(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def train_epoch(self, dataloader) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = self._batch_to_device(batch)

            self.optimizer.zero_grad()

            output = self.model(
                price=batch["price"],
                sentiment=batch["sentiment"],
                fundamentals=batch["fundamentals"],
                macro=batch["macro"],
                stock_id=batch["stock_id"],
            )

            loss = self.criterion(output["prediction"], batch["label"])
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["gradient_clip_norm"],
            )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Run validation. Returns (avg_loss, predictions, actuals).
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []

        for batch in dataloader:
            batch = self._batch_to_device(batch)

            output = self.model(
                price=batch["price"],
                sentiment=batch["sentiment"],
                fundamentals=batch["fundamentals"],
                macro=batch["macro"],
                stock_id=batch["stock_id"],
            )

            loss = self.criterion(output["prediction"], batch["label"])
            total_loss += loss.item()
            n_batches += 1

            all_preds.append(output["prediction"].cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        return avg_loss, preds, labels

    def check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Returns True if patience exhausted.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            # Save best model state
            self.best_state_dict = {
                k: v.clone() for k, v in self.model.state_dict().items()
            }
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config["early_stopping_patience"]

    def restore_best_model(self):
        """Load the best model state (from before overfitting)."""
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            logger.info("Restored best model weights")

    def train_fold(
        self,
        train_loader,
        val_loader,
        fold_num: int = 0,
    ) -> Dict:
        """
        Full training loop for one walk-forward fold.

        Returns:
            Dict with training history (losses per epoch, best epoch, etc.)
        """
        logger.info(f"═══ Fold {fold_num} Training ═══")
        logger.info(f"  Train: {len(train_loader.dataset)} samples | "
                     f"Val: {len(val_loader.dataset)} samples | "
                     f"Device: {self.device}")

        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "best_epoch": 0,
            "best_val_loss": float("inf"),
        }

        start_time = time.time()

        for epoch in range(self.config["max_epochs"]):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_preds, val_labels = self.validate(val_loader)

            # Learning rate step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["learning_rates"].append(current_lr)

            # Log progress
            if epoch % 5 == 0 or epoch < 3:
                logger.info(
                    f"  Epoch {epoch:3d} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Patience: {self.patience_counter}/{self.config['early_stopping_patience']}"
                )

            # Early stopping check
            if self.check_early_stopping(val_loss):
                logger.info(f"  Early stopping at epoch {epoch} "
                           f"(best: epoch {history['best_epoch']})")
                break

            if val_loss < history["best_val_loss"]:
                history["best_val_loss"] = val_loss
                history["best_epoch"] = epoch

        # Restore best model
        self.restore_best_model()

        elapsed = time.time() - start_time
        logger.info(
            f"  Fold {fold_num} complete: {elapsed:.0f}s | "
            f"Best epoch: {history['best_epoch']} | "
            f"Best val loss: {history['best_val_loss']:.4f}"
        )

        return history


if __name__ == "__main__":
    # Quick integration test with random data
    from model.mcat import MCAT
    from features.dataset import collate_fn

    set_seed(42)

    model = MCAT(n_price_features=39, n_sent_features=769,
                 n_fund_features=7, n_macro_features=20)

    trainer = Trainer(model)

    # Create minimal dataloader with random data
    class DummyDataset:
        def __init__(self, n=128):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            return {
                "price": np.random.randn(60, 39).astype(np.float32),
                "sentiment": np.random.randn(60, 769).astype(np.float32),
                "fundamentals": np.random.randn(7).astype(np.float32),
                "macro": np.random.randn(60, 20).astype(np.float32),
                "label": np.float32(np.random.randn() * 3),
                "stock_id": np.int64(np.random.randint(0, 15)),
            }

    from torch.utils.data import DataLoader

    train_ds = DummyDataset(128)
    val_ds = DummyDataset(32)
    train_loader = DataLoader(train_ds, batch_size=32, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, collate_fn=collate_fn)

    # Train for a few epochs
    history = trainer.train_fold(train_loader, val_loader, fold_num=0)
    print(f"\nTraining complete! Epochs: {len(history['train_loss'])}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Best val loss: {history['best_val_loss']:.4f}")
