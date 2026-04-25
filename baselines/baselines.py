"""
baselines/baselines.py — Baseline models for comparison.
==========================================================
B1: Historical Mean — predict training set average return
B2: Ridge Regression — L2-regularized linear model on flattened price features
B3: LightGBM — gradient boosted trees on flattened price features
B4: Single-layer Transformer — price-only, no cross-attention
B5: LSTM + Concatenation Fusion — previous architecture (our v1)

All baselines use the same walk-forward splits, labels, and evaluation metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RANDOM_SEED, LOOKBACK_WINDOW
from evaluation.metrics import compute_metrics, FoldMetrics
from training.walk_forward import WalkForwardFold
from utils import setup_logger, set_seed

logger = setup_logger(__name__)


def _extract_flat_price_and_labels(dataset, subset):
    """Extract flattened price features and labels from a dataset subset."""
    n = len(subset)
    if n == 0:
        return np.array([]), np.array([])

    X = np.array([subset[i]["price"].flatten() for i in range(n)])
    y = np.array([subset[i]["label"] for i in range(n)])
    return X, y


# ─────────────────────────────────────────────────────────────
# B1: HISTORICAL MEAN
# ─────────────────────────────────────────────────────────────

def run_historical_mean(
    dataset,
    folds: List[WalkForwardFold],
) -> List[FoldMetrics]:
    """
    Baseline B1: Predict the training set mean return for all test samples.
    This is the simplest possible baseline — the floor that all models must beat.
    """
    logger.info("Running baseline: Historical Mean")
    all_metrics = []

    for fold in folds:
        train_sub = dataset.get_subset_by_dates(fold.train_start, fold.train_end)
        test_sub = dataset.get_subset_by_dates(fold.test_start, fold.test_end)

        if len(train_sub) == 0 or len(test_sub) == 0:
            continue

        train_y = np.array([train_sub[i]["label"] for i in range(len(train_sub))])
        test_y = np.array([test_sub[i]["label"] for i in range(len(test_sub))])

        mean_pred = np.full(len(test_y), train_y.mean())
        metrics = compute_metrics(mean_pred, test_y, fold.fold_num)
        all_metrics.append(metrics)
        logger.info(f"  Fold {fold.fold_num}: {metrics}")

    return all_metrics


# ─────────────────────────────────────────────────────────────
# B2: RIDGE REGRESSION
# ─────────────────────────────────────────────────────────────

def run_ridge_regression(
    dataset,
    folds: List[WalkForwardFold],
    alpha: float = 1.0,
) -> List[FoldMetrics]:
    """
    Baseline B2: L2-regularized linear regression on flattened price features.
    Input: 45 days × 39 features = 2,340-dim vector.
    """
    from sklearn.linear_model import Ridge

    logger.info(f"Running baseline: Ridge Regression (alpha={alpha})")
    all_metrics = []

    for fold in folds:
        train_sub = dataset.get_subset_by_dates(fold.train_start, fold.train_end)
        test_sub = dataset.get_subset_by_dates(fold.test_start, fold.test_end)

        if len(train_sub) == 0 or len(test_sub) == 0:
            continue

        train_X, train_y = _extract_flat_price_and_labels(dataset, train_sub)
        test_X, test_y = _extract_flat_price_and_labels(dataset, test_sub)

        model = Ridge(alpha=alpha)
        model.fit(train_X, train_y)
        preds = model.predict(test_X)

        metrics = compute_metrics(preds, test_y, fold.fold_num)
        all_metrics.append(metrics)
        logger.info(f"  Fold {fold.fold_num}: {metrics}")

    return all_metrics


# ─────────────────────────────────────────────────────────────
# B3: LIGHTGBM
# ─────────────────────────────────────────────────────────────

def run_lightgbm(
    dataset,
    folds: List[WalkForwardFold],
) -> List[FoldMetrics]:
    """
    Baseline B3: Gradient boosted trees on flattened price features.
    Often competitive with or superior to deep learning on tabular data
    (Grinsztajn et al., 2022).
    """
    logger.info("Running baseline: LightGBM")
    all_metrics = []

    try:
        import lightgbm as lgb
        use_lgb = True
    except ImportError:
        logger.warning("LightGBM not installed — falling back to GradientBoostingRegressor")
        from sklearn.ensemble import GradientBoostingRegressor
        use_lgb = False

    for fold in folds:
        train_sub = dataset.get_subset_by_dates(fold.train_start, fold.train_end)
        test_sub = dataset.get_subset_by_dates(fold.test_start, fold.test_end)

        if len(train_sub) == 0 or len(test_sub) == 0:
            continue

        train_X, train_y = _extract_flat_price_and_labels(dataset, train_sub)
        test_X, test_y = _extract_flat_price_and_labels(dataset, test_sub)

        if use_lgb:
            model = lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=RANDOM_SEED,
                verbose=-1,
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=RANDOM_SEED,
            )

        model.fit(train_X, train_y)
        preds = model.predict(test_X)

        metrics = compute_metrics(preds, test_y, fold.fold_num)
        all_metrics.append(metrics)
        logger.info(f"  Fold {fold.fold_num}: {metrics}")

    return all_metrics


# ─────────────────────────────────────────────────────────────
# B4: SINGLE-LAYER TRANSFORMER (price only)
# ─────────────────────────────────────────────────────────────

def _run_nn_baseline(
    model_cls,
    model_kwargs: dict,
    dataset,
    folds: List[WalkForwardFold],
    baseline_name: str,
    config_overrides: dict = None,
) -> List[FoldMetrics]:
    """
    Generic runner for neural network baselines (B4, B5).
    Uses the same Trainer/walk-forward/early-stopping infrastructure as MCAT.
    """
    import torch
    from torch.utils.data import DataLoader
    from training.trainer import Trainer
    from features.dataset import collate_fn
    from config import TRAINING_CONFIG

    logger.info(f"Running baseline: {baseline_name}")
    all_metrics = []

    for fold in folds:
        set_seed(RANDOM_SEED + fold.fold_num)

        train_sub = dataset.get_subset_by_dates(fold.train_start, fold.train_end)
        val_sub = dataset.get_subset_by_dates(fold.val_start, fold.val_end)
        test_sub = dataset.get_subset_by_dates(fold.test_start, fold.test_end)

        if len(train_sub) == 0 or len(val_sub) == 0 or len(test_sub) == 0:
            logger.warning(f"  Fold {fold.fold_num}: Insufficient data — skipped")
            continue

        train_ld = DataLoader(train_sub, batch_size=64, shuffle=True, collate_fn=collate_fn)
        val_ld = DataLoader(val_sub, batch_size=64, collate_fn=collate_fn)
        test_ld = DataLoader(test_sub, batch_size=64, collate_fn=collate_fn)

        model = model_cls(**model_kwargs)

        if fold.fold_num == 0:
            from baselines.nn_baselines import count_parameters
            logger.info(f"  {baseline_name} params: {count_parameters(model):,}")

        train_config = {**TRAINING_CONFIG}
        if config_overrides:
            train_config.update(config_overrides)

        trainer = Trainer(model, config=train_config)
        trainer.train_fold(train_ld, val_ld, fold_num=fold.fold_num)

        _, preds, actuals = trainer.validate(test_ld)
        metrics = compute_metrics(preds, actuals, fold.fold_num)
        all_metrics.append(metrics)
        logger.info(f"  {baseline_name} Fold {fold.fold_num}: {metrics}")

    return all_metrics


def run_single_transformer(
    dataset,
    folds: List[WalkForwardFold],
    config_overrides: dict = None,
) -> List[FoldMetrics]:
    """
    Baseline B4: Single-layer Transformer on price features only.
    No cross-attention, no multi-modal fusion.
    """
    from baselines.nn_baselines import PriceOnlyTransformer

    model_kwargs = {
        "n_price_features": dataset.X_price.shape[2],
        "n_fund_features": dataset.X_fund.shape[1],
    }
    return _run_nn_baseline(
        PriceOnlyTransformer, model_kwargs, dataset, folds,
        "PriceOnlyTransformer (B4)", config_overrides,
    )


# ─────────────────────────────────────────────────────────────
# B5: LSTM + CONCATENATION FUSION
# ─────────────────────────────────────────────────────────────

def run_lstm_concat(
    dataset,
    folds: List[WalkForwardFold],
    config_overrides: dict = None,
) -> List[FoldMetrics]:
    """
    Baseline B5: BiLSTM per modality + concatenation fusion.
    Our v1 architecture — the approach MCAT is designed to improve upon.
    Tests concatenation vs cross-attention, and LSTM vs Transformer.
    """
    from baselines.nn_baselines import LSTMConcatFusion

    model_kwargs = {
        "n_price_features": dataset.X_price.shape[2],
        "n_sent_features": dataset.X_sent.shape[2],
        "n_fund_features": dataset.X_fund.shape[1],
        "n_macro_features": dataset.X_macro.shape[2],
    }
    return _run_nn_baseline(
        LSTMConcatFusion, model_kwargs, dataset, folds,
        "LSTMConcatFusion (B5)", config_overrides,
    )


# ─────────────────────────────────────────────────────────────
# UNIFIED RUNNER
# ─────────────────────────────────────────────────────────────

BASELINE_REGISTRY = {
    "historical_mean": run_historical_mean,
    "ridge": run_ridge_regression,
    "lightgbm": run_lightgbm,
    "single_transformer": run_single_transformer,
    "lstm_concat": run_lstm_concat,
}


def run_baseline(
    name: str,
    dataset,
    folds: List[WalkForwardFold],
    **kwargs,
) -> List[FoldMetrics]:
    """Run a named baseline model."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(BASELINE_REGISTRY.keys())}")
    return BASELINE_REGISTRY[name](dataset, folds, **kwargs)


if __name__ == "__main__":
    # Quick test with synthetic data
    from features.dataset import MultiModalDataset
    from training.walk_forward import generate_walk_forward_folds

    set_seed(42)
    n = 1200
    dates = pd.bdate_range("2017-01-01", periods=n)

    price_features = {
        "AAPL": pd.DataFrame(np.random.randn(n, 39).astype(np.float32),
                             index=dates, columns=[f"f{i}" for i in range(39)])
    }
    labels = {
        "AAPL": pd.Series(np.random.randn(n).astype(np.float32) * 3, index=dates)
    }

    ds = MultiModalDataset(price_features=price_features, labels=labels,
                           denoise=False, tickers=["AAPL"])
    folds = generate_walk_forward_folds()

    # Test historical mean
    results = run_historical_mean(ds, folds[:2])
    print(f"\nHistorical Mean: {len(results)} folds evaluated")

    # Test ridge
    results = run_ridge_regression(ds, folds[:2])
    print(f"Ridge: {len(results)} folds evaluated")
