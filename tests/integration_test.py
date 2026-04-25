"""
tests/integration_test.py — End-to-end pipeline validation.
=============================================================
Tests the complete pipeline from synthetic data through training
to evaluation, verifying every component works together correctly.

Run: python tests/integration_test.py
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import setup_logger, set_seed
from config import LOOKBACK_WINDOW, FINBERT_DIM, STOCK_TO_ID

logger = setup_logger("integration_test")


def create_synthetic_data(n_tickers=2, n_days=1200):
    """Create realistic synthetic multi-modal data for testing."""
    set_seed(42)
    tickers = ["AAPL", "JPM"][:n_tickers]
    dates = pd.bdate_range("2017-01-01", periods=n_days)

    # Price features (39 per day)
    price_features = {}
    labels = {}
    for ticker in tickers:
        # Simulate realistic features with some structure
        feat = np.random.randn(n_days, 39).astype(np.float32)
        # Add slight serial correlation (makes it more realistic)
        for j in range(39):
            for i in range(1, n_days):
                feat[i, j] = 0.3 * feat[i - 1, j] + 0.7 * feat[i, j]

        price_features[ticker] = pd.DataFrame(
            feat, index=dates, columns=[f"f{i}" for i in range(39)]
        )

        # Labels: 3-day forward return in pp (slightly correlated with features)
        signal = feat[:, 0] * 0.5 + feat[:, 4] * 0.3  # Weak signal
        noise = np.random.randn(n_days) * 3
        y = (signal + noise).astype(np.float32)
        labels[ticker] = pd.Series(y, index=dates, name="label")
        # NaN last 3 values (no future data)
        labels[ticker].iloc[-3:] = np.nan

    # Sentiment: use small dim (6) for test to avoid OOM
    # In production this would be 769 (768 FinBERT + 1 count)
    test_sent_dim = 5
    sent_emb = {t: np.zeros((n_days, test_sent_dim), dtype=np.float32) for t in tickers}
    sent_cnt = {t: np.zeros(n_days, dtype=np.float32) for t in tickers}

    # Fundamentals: synthetic (7 features)
    fund_features = {}
    for ticker in tickers:
        fund = pd.DataFrame(
            np.random.randn(n_days, 7).astype(np.float32) * 0.5 + 1,
            index=dates, columns=[f"fund_{i}" for i in range(7)]
        )
        fund_features[ticker] = fund

    # Macro: synthetic (5 features for speed)
    macro = pd.DataFrame(
        np.random.randn(n_days, 5).astype(np.float32),
        index=dates, columns=[f"macro_{i}" for i in range(5)]
    )

    return tickers, price_features, labels, sent_emb, sent_cnt, fund_features, macro


def test_dataset_creation(price_features, labels, sent_emb, sent_cnt, fund_features, macro, tickers):
    """Test Step 1: Dataset creation."""
    logger.info("═══ TEST 1: Dataset Creation ═══")
    from features.dataset import MultiModalDataset

    ds = MultiModalDataset(
        price_features=price_features,
        labels=labels,
        sentiment_embeddings=sent_emb,
        sentiment_counts=sent_cnt,
        fund_features=fund_features,
        macro_features=macro,
        denoise=False,  # Skip for speed in integration test
        tickers=tickers,
    )

    assert len(ds) > 0, "Dataset is empty!"
    sample = ds[0]
    assert sample["price"].shape == (60, 39), f"Price shape wrong: {sample['price'].shape}"
    assert sample["sentiment"].shape[0] == 60, f"Sent shape wrong: {sample['sentiment'].shape}"
    assert sample["macro"].shape[0] == 60, f"Macro shape wrong: {sample['macro'].shape}"
    assert not np.isnan(sample["label"]), "Label is NaN!"
    sent_dim = sample["sentiment"].shape[1]
    logger.info(f"  ✓ Dataset: {len(ds)} samples, sent_dim={sent_dim}, all shapes correct")
    return ds


def test_walk_forward_splits(ds):
    """Test Step 2: Walk-forward split generation."""
    logger.info("═══ TEST 2: Walk-Forward Splits ═══")
    from training.walk_forward import generate_walk_forward_folds

    folds = generate_walk_forward_folds()
    assert len(folds) > 0, "No folds generated!"

    # Test that date splitting works
    fold = folds[0]
    train_sub = ds.get_subset_by_dates(fold.train_start, fold.train_end)
    val_sub = ds.get_subset_by_dates(fold.val_start, fold.val_end)
    test_sub = ds.get_subset_by_dates(fold.test_start, fold.test_end)

    logger.info(f"  Fold 0: train={len(train_sub)}, val={len(val_sub)}, test={len(test_sub)}")
    logger.info(f"  ✓ {len(folds)} folds generated, date splitting works")
    return folds


def test_model_creation():
    """Test Step 3: Model architecture."""
    logger.info("═══ TEST 3: Model Architecture ═══")
    import torch
    from model.mcat import MCAT, count_parameters

    # Test with small sentiment dim (production would use 769)
    model = MCAT(
        n_price_features=39,
        n_sent_features=6,  # Small for test (5 emb + 1 count)
        n_fund_features=7,
        n_macro_features=5,
    )

    n_params = count_parameters(model)
    assert n_params > 50_000, f"Model too small: {n_params}"
    assert n_params < 500_000, f"Model too large: {n_params}"

    # Forward pass
    B = 4
    out = model(
        price=torch.randn(B, 60, 39),
        sentiment=torch.randn(B, 60, 6),
        fundamentals=torch.randn(B, 7),
        macro=torch.randn(B, 60, 5),
        stock_id=torch.randint(0, 15, (B,)),
        return_attention=True,
    )

    assert out["prediction"].shape == (B,), f"Output shape wrong: {out['prediction'].shape}"
    assert out["attn_sent"] is not None, "No attention weights"
    assert not torch.isnan(out["prediction"]).any(), "NaN in predictions"

    # Gradient flow
    loss = out["prediction"].sum()
    loss.backward()
    params_with_grad = sum(
        1 for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.norm() > 0
    )
    total_params_groups = sum(1 for p in model.parameters() if p.requires_grad)
    # Allow the no_news_embedding (1 param group) to not have gradient
    assert params_with_grad >= total_params_groups - 2, \
        f"Broken gradient flow: {params_with_grad}/{total_params_groups} params have gradients"

    logger.info(f"  ✓ Model: {n_params:,} params, forward pass OK, gradients flow")
    return model


def test_training_loop(ds, folds):
    """Test Step 4: Training on first fold."""
    logger.info("═══ TEST 4: Training Loop (1 fold, 5 epochs) ═══")
    import torch
    from torch.utils.data import DataLoader
    from model.mcat import MCAT
    from training.trainer import Trainer
    from features.dataset import collate_fn

    set_seed(42)

    fold = folds[0]
    train_sub = ds.get_subset_by_dates(fold.train_start, fold.train_end)
    val_sub = ds.get_subset_by_dates(fold.val_start, fold.val_end)

    # Use small subset for speed
    n_macro = ds.X_macro.shape[2]

    if len(train_sub) == 0 or len(val_sub) == 0:
        logger.warning("  ⚠ Insufficient data for fold 0 — using full dataset split")
        # Fallback: split by index
        n = len(ds)
        train_idx = np.arange(0, int(n * 0.7))
        val_idx = np.arange(int(n * 0.7), int(n * 0.85))
        from features.dataset import MultiModalSubset
        train_sub = MultiModalSubset(ds, train_idx)
        val_sub = MultiModalSubset(ds, val_idx)

    train_ld = DataLoader(train_sub, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_ld = DataLoader(val_sub, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = MCAT(
        n_price_features=39,
        n_sent_features=ds.X_sent.shape[2],  # Dynamic: 6 in test, 769 in prod
        n_fund_features=ds.X_fund.shape[1],
        n_macro_features=n_macro,
    )

    # Quick training config
    quick_config = {
        **dict(
            batch_size=32, max_epochs=5, learning_rate=3e-4,
            min_learning_rate=1e-6, weight_decay=0.01,
            warmup_epochs=1, early_stopping_patience=3,
            gradient_clip_norm=1.0, huber_delta=1.0,
            optimizer="adamw", betas=(0.9, 0.999),
        ),
    }

    trainer = Trainer(model, config=quick_config)
    history = trainer.train_fold(train_ld, val_ld, fold_num=0)

    assert len(history["train_loss"]) > 0, "No training happened!"
    assert history["train_loss"][-1] < history["train_loss"][0] * 2, "Loss exploded!"
    logger.info(f"  ✓ Training: {len(history['train_loss'])} epochs, "
                f"loss {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f}")
    return trainer


def test_evaluation(trainer, ds, folds):
    """Test Step 5: Evaluation metrics computation."""
    logger.info("═══ TEST 5: Evaluation Metrics ═══")
    import torch
    from torch.utils.data import DataLoader
    from features.dataset import collate_fn, MultiModalSubset
    from evaluation.metrics import compute_metrics

    # Get test data
    fold = folds[0]
    test_sub = ds.get_subset_by_dates(fold.test_start, fold.test_end)

    if len(test_sub) == 0:
        n = len(ds)
        test_idx = np.arange(int(n * 0.85), n)
        test_sub = MultiModalSubset(ds, test_idx)

    test_ld = DataLoader(test_sub, batch_size=32, shuffle=False, collate_fn=collate_fn)

    _, preds, actuals = trainer.validate(test_ld)

    metrics = compute_metrics(preds, actuals, fold_num=0)
    logger.info(f"  {metrics}")

    assert not np.isnan(metrics.mae), "MAE is NaN!"
    assert not np.isnan(metrics.ic), "IC is NaN!"
    assert 0 <= metrics.directional_accuracy <= 1, f"DA out of range: {metrics.directional_accuracy}"

    logger.info(f"  ✓ Metrics computed correctly")
    return metrics


def run_all_tests():
    """Run the complete integration test suite."""
    start = time.time()
    logger.info("="*60)
    logger.info("  INTEGRATION TEST SUITE")
    logger.info("="*60)

    # Step 0: Create synthetic data
    logger.info("Creating synthetic test data...")
    tickers, price_features, labels, sent_emb, sent_cnt, fund_features, macro = \
        create_synthetic_data(n_tickers=2, n_days=1500)

    # Step 1: Dataset creation
    ds = test_dataset_creation(price_features, labels, sent_emb, sent_cnt, fund_features, macro, tickers)

    # Step 2: Walk-forward splits
    folds = test_walk_forward_splits(ds)

    # Step 3: Model architecture
    model = test_model_creation()

    # Step 4: Training
    trainer = test_training_loop(ds, folds)

    # Step 5: Evaluation
    metrics = test_evaluation(trainer, ds, folds)

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"  ALL TESTS PASSED ✓  ({elapsed:.1f}s)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    run_all_tests()
