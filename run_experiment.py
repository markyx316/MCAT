"""
run_experiment.py — Main experiment orchestrator.
===================================================
Runs a complete walk-forward evaluation for a given model configuration:
  1. Load/build the full dataset
  2. For each of 3 walk-forward folds:
     a. Split data by date ranges
     b. Initialize fresh model
     c. Train with early stopping
     d. Evaluate on test set
     e. Record metrics
  3. Aggregate results across folds
  4. Save results

Usage:
    python run_experiment.py --experiment full_mcat
    python run_experiment.py --experiment price_only
    python run_experiment.py --experiment lightgbm
    ...
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    TRAINING_CONFIG, MODEL_CONFIG, RANDOM_SEED, N_PRICE_FEATURES, N_SENTIMENT_FEATURES,
    N_FUND_FEATURES_SYNTHETIC, RESULTS_DIR,
)
from data.provenance import provenance
from utils import setup_logger, set_seed
from training.walk_forward import generate_walk_forward_folds, generate_focused_folds, WalkForwardFold
from evaluation.metrics import (
    compute_metrics, aggregate_fold_metrics, print_results_table, FoldMetrics,
)
from experiment_log import log_experiment, print_leaderboard

logger = setup_logger(__name__)


def _json_default(obj):
    """Handle numpy types during JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def build_dataloaders(dataset, fold: WalkForwardFold, batch_size: int = 64):
    """Create train/val/test DataLoaders for a single fold."""
    import torch
    from torch.utils.data import DataLoader
    from features.dataset import collate_fn

    train_subset = dataset.get_subset_by_dates(fold.train_start, fold.train_end)
    val_subset = dataset.get_subset_by_dates(fold.val_start, fold.val_end)
    test_subset = dataset.get_subset_by_dates(fold.test_start, fold.test_end)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, drop_last=False,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, drop_last=False,
    )

    return train_loader, val_loader, test_loader


def run_mcat_experiment(
    dataset,
    folds: List[WalkForwardFold],
    experiment_name: str = "full_mcat",
    n_price_features: int = N_PRICE_FEATURES,
    n_sent_features: int = N_SENTIMENT_FEATURES,
    n_fund_features: int = N_FUND_FEATURES_SYNTHETIC,
    n_macro_features: int = 20,
    config_overrides: dict = None,
    disable_modalities: list = None,
) -> List[FoldMetrics]:
    """
    Run the MCAT model across all walk-forward folds.

    Args:
        disable_modalities: List of modality names to disable for ablation.
            Options: "sentiment", "fundamentals", "macro".
            Empty list or None = full MCAT (all modalities enabled).

    For each fold:
      1. Create dataloaders from date-split subsets
      2. Initialize a FRESH model (no weight carryover)
      3. Train with early stopping
      4. Evaluate on test set
      5. Collect metrics

    Returns:
        List of FoldMetrics, one per fold.
    """
    import torch
    from model.mcat import MCAT, count_parameters
    from training.trainer import Trainer

    all_fold_metrics = []
    all_predictions = []
    all_actuals = []
    all_dates = []

    for fold in folds:
        logger.info(f"\n{'═'*60}")
        logger.info(f"  FOLD {fold.fold_num}: {fold}")
        logger.info(f"{'═'*60}")

        # 1. Build dataloaders
        train_loader, val_loader, test_loader = build_dataloaders(
            dataset, fold, batch_size=TRAINING_CONFIG["batch_size"],
        )

        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        n_test = len(test_loader.dataset)

        if n_train == 0 or n_val == 0 or n_test == 0:
            logger.warning(f"  Fold {fold.fold_num}: Insufficient data "
                          f"(train={n_train}, val={n_val}, test={n_test}) — SKIPPED")
            continue

        logger.info(f"  Samples: train={n_train}, val={n_val}, test={n_test}")

        # 2. Fresh model initialization
        set_seed(RANDOM_SEED + fold.fold_num)

        model = MCAT(
            n_price_features=n_price_features,
            n_sent_features=n_sent_features,
            n_fund_features=n_fund_features,
            n_macro_features=n_macro_features,
            disable_modalities=disable_modalities,
        )

        if fold.fold_num == 0:
            disabled_str = f" (disabled: {disable_modalities})" if disable_modalities else ""
            logger.info(f"  Model parameters: {count_parameters(model):,}{disabled_str}")

        # 3. Train with early stopping
        train_config = {**TRAINING_CONFIG}
        if config_overrides:
            train_config.update(config_overrides)

        trainer = Trainer(model, config=train_config)
        history = trainer.train_fold(train_loader, val_loader, fold_num=fold.fold_num)

        # 4. Evaluate on test set
        test_loss, test_preds, test_labels = trainer.validate(test_loader)

        # 5. Compute and record metrics
        fold_metrics = compute_metrics(test_preds, test_labels, fold_num=fold.fold_num)
        all_fold_metrics.append(fold_metrics)

        logger.info(f"  Test: {fold_metrics}")

        all_predictions.append(test_preds)
        all_actuals.append(test_labels)

    return all_fold_metrics


def run_baseline_experiment(
    dataset,
    folds: List[WalkForwardFold],
    baseline_type: str = "ridge",
) -> List[FoldMetrics]:
    """
    Run a baseline model (Ridge, LightGBM, or Historical Mean) across folds.
    """
    all_fold_metrics = []

    for fold in folds:
        # Get data subsets
        train_sub = dataset.get_subset_by_dates(fold.train_start, fold.train_end)
        test_sub = dataset.get_subset_by_dates(fold.test_start, fold.test_end)

        if len(train_sub) == 0 or len(test_sub) == 0:
            continue

        # Extract flattened price features and labels
        train_X = np.array([train_sub[i]["price"].flatten() for i in range(len(train_sub))])
        train_y = np.array([train_sub[i]["label"] for i in range(len(train_sub))])
        test_X = np.array([test_sub[i]["price"].flatten() for i in range(len(test_sub))])
        test_y = np.array([test_sub[i]["label"] for i in range(len(test_sub))])

        if baseline_type == "historical_mean":
            preds = np.full(len(test_y), train_y.mean())

        elif baseline_type == "ridge":
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            model.fit(train_X, train_y)
            preds = model.predict(test_X)

        elif baseline_type == "lightgbm":
            try:
                import lightgbm as lgb
                model = lgb.LGBMRegressor(
                    n_estimators=650, max_depth=8, learning_rate=0.06,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_SEED, verbose=-1,
                )
                model.fit(train_X, train_y)
                preds = model.predict(test_X)
            except ImportError:
                logger.warning("LightGBM not installed — using Ridge fallback")
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0)
                model.fit(train_X, train_y)
                preds = model.predict(test_X)

        else:
            raise ValueError(f"Unknown baseline: {baseline_type}")

        fold_metrics = compute_metrics(preds, test_y, fold_num=fold.fold_num)
        all_fold_metrics.append(fold_metrics)
        logger.info(f"  {baseline_type} Fold {fold.fold_num}: {fold_metrics}")

    return all_fold_metrics


def save_results(
    experiment_name: str,
    fold_metrics: List[FoldMetrics],
    output_dir: Path = None,
):
    """Save fold-level and aggregated results to JSON, including provenance."""
    if output_dir is None:
        output_dir = RESULTS_DIR / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)

    # Fold-level results (convert numpy types for JSON compatibility)
    fold_data = [{
        "fold": int(m.fold_num),
        "mae": float(m.mae),
        "rmse": float(m.rmse),
        "r2": float(m.r2),
        "ic": float(m.ic),
        "directional_accuracy": float(m.directional_accuracy),
        "skill_score": float(m.skill_score),
        "n_samples": int(m.n_samples),
    } for m in fold_metrics]

    # Aggregated results
    agg = aggregate_fold_metrics(fold_metrics)
    agg_data = {
        name: {"mean": float(v["mean"]), "std": float(v["std"])}
        for name, v in agg.items()
    }

    # Include provenance report — CRITICAL for transparency
    provenance_data = provenance.to_dict()

    results = {
        "experiment": experiment_name,
        "n_folds": len(fold_metrics),
        "folds": fold_data,
        "aggregated": agg_data,
        "data_provenance": provenance_data,
    }

    out_path = output_dir / f"{experiment_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    logger.info(f"Results saved → {out_path}")

    # Also save provenance as standalone file
    provenance.save()

    # Print provenance report to log
    provenance.report()

    return results


def main():
    parser = argparse.ArgumentParser(description="Run MCAT experiment")
    parser.add_argument("--experiment", type=str, default="quick_test",
                        choices=["quick_test", "full_mcat", "all_baselines",
                                 "ridge", "lightgbm", "historical_mean",
                                 "single_transformer", "lstm_concat",
                                 # Ablation studies
                                 "ablate_no_sentiment",
                                 "ablate_no_fundamentals",
                                 "ablate_no_macro",
                                 "ablate_price_only",
                                 "all_ablations",
                                 ],
                        help="Experiment configuration to run")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (1 fold, reduced epochs)")
    parser.add_argument("--fold-mode", type=str, default="focused",
                        choices=["focused", "full"],
                        help="Fold strategy: 'focused' (3 folds, 6-mo test, default) "
                             "or 'full' (14 folds, 3-mo test)")
    parser.add_argument("--fnspid", type=str, default=None,
                        help="Path to FNSPID CSV for real sentiment data")
    parser.add_argument("--force-synthetic-fund", action="store_true",
                        help="Skip Alpha Vantage, use synthetic fundamentals")
    parser.add_argument("--no-denoise", action="store_true",
                        help="Skip wavelet denoising")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override max training epochs (default: 100)")
    parser.add_argument("--lookback", type=int, default=None,
                        help="Lookback window in trading days (default: 60). "
                             "Try 30, 45, 60, or 90 for window tuning.")
    parser.add_argument("--train-years", type=float, default=None,
                        help="Training window in years per fold. Default: expanding "
                             "from 2017. E.g., --train-years 3 uses a sliding 3-year "
                             "window ending at each fold's train_end.")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    logger.info(f"Experiment: {args.experiment} | Quick: {args.quick}")
    if args.lookback:
        logger.info(f"  Lookback window: {args.lookback} days (override)")
    if args.train_years:
        logger.info(f"  Training window: {args.train_years} years (sliding)")

    # ─── Step 1: Build dataset ───
    if args.experiment == "quick_test":
        from data.build_dataset import build_quick_test_dataset
        dataset, report = build_quick_test_dataset(n_tickers=1, n_days=1200)
    else:
        from data.build_dataset import build_full_dataset
        dataset, report = build_full_dataset(
            fnspid_path=args.fnspid,
            force_synthetic_fundamentals=args.force_synthetic_fund,
            denoise=not args.no_denoise,
            lookback=args.lookback,
        )

    # ─── Step 2: Generate walk-forward folds ───
    if args.experiment == "quick_test":
        # Quick test uses full fold generator (backward compatible)
        folds = generate_walk_forward_folds()
    elif args.fold_mode == "focused":
        folds = generate_focused_folds(train_years=args.train_years)
    else:
        folds = generate_walk_forward_folds()

    if args.quick:
        # Use only the first fold for fast iteration
        folds = [folds[0]]
        logger.info(f"Quick mode: using 1 fold")

    # ─── Step 3: Run experiments ───

    # Capture CLI flags for logging
    cli_flags = {
        "lookback": args.lookback,
        "train_years": args.train_years,
        "fold_mode": args.fold_mode,
        "quick": args.quick,
        "no_denoise": args.no_denoise,
        "max_epochs_override": args.max_epochs,
    }

    # Define ablation configurations: experiment_name → list of disabled modalities
    ABLATION_CONFIGS = {
        "ablate_no_sentiment":    ["sentiment"],
        "ablate_no_fundamentals": ["fundamentals"],
        "ablate_no_macro":        ["macro"],
        "ablate_price_only":      ["sentiment", "fundamentals", "macro"],
    }

    # Helper: compute param count for a given MCAT config
    def _count_mcat_params(disable_modalities=None):
        from model.mcat import MCAT, count_parameters
        m = MCAT(
            n_price_features=dataset.X_price.shape[2],
            n_sent_features=dataset.X_sent.shape[2],
            n_fund_features=dataset.X_fund.shape[1],
            n_macro_features=dataset.X_macro.shape[2],
            disable_modalities=disable_modalities,
        )
        return count_parameters(m)

    experiment_start = time.time()

    if args.experiment in ("quick_test", "full_mcat") or args.experiment in ABLATION_CONFIGS:
        config_overrides = {}
        if args.max_epochs is not None:
            config_overrides["max_epochs"] = args.max_epochs
        elif args.quick:
            config_overrides["max_epochs"] = 10
            config_overrides["early_stopping_patience"] = 5

        disable_modalities = ABLATION_CONFIGS.get(args.experiment, [])
        if disable_modalities:
            logger.info(f"  Ablation: disabling {disable_modalities}")

        fold_metrics = run_mcat_experiment(
            dataset, folds,
            experiment_name=args.experiment,
            n_sent_features=dataset.X_sent.shape[2],
            n_fund_features=dataset.X_fund.shape[1],
            n_macro_features=dataset.X_macro.shape[2],
            config_overrides=config_overrides,
            disable_modalities=disable_modalities,
        )

    elif args.experiment in ("ridge", "lightgbm", "historical_mean"):
        fold_metrics = run_baseline_experiment(
            dataset, folds, baseline_type=args.experiment,
        )

    elif args.experiment in ("single_transformer", "lstm_concat"):
        from baselines.baselines import run_baseline
        config_overrides = {}
        if args.max_epochs is not None:
            config_overrides["max_epochs"] = args.max_epochs
        elif args.quick:
            config_overrides["max_epochs"] = 10
            config_overrides["early_stopping_patience"] = 5
        fold_metrics = run_baseline(
            args.experiment, dataset, folds,
            config_overrides=config_overrides if config_overrides else None,
        )

    elif args.experiment == "all_baselines":
        from baselines.baselines import run_baseline
        all_results = {}

        # Simple baselines (fast)
        for baseline in ["historical_mean", "ridge"]:
            logger.info(f"\n{'═'*60}\n  BASELINE: {baseline}\n{'═'*60}")
            t0 = time.time()
            metrics = run_baseline_experiment(dataset, folds, baseline_type=baseline)
            if metrics:
                agg = aggregate_fold_metrics(metrics)
                all_results[baseline] = agg
                save_results(baseline, metrics)
                log_experiment(
                    baseline, metrics,
                    training_config={"type": "classical_baseline"},
                    cli_flags=cli_flags,
                    training_time_s=time.time() - t0,
                    seed=RANDOM_SEED,
                )

        # LightGBM
        try:
            logger.info(f"\n{'═'*60}\n  BASELINE: lightgbm\n{'═'*60}")
            t0 = time.time()
            metrics = run_baseline_experiment(dataset, folds, baseline_type="lightgbm")
            if metrics:
                agg = aggregate_fold_metrics(metrics)
                all_results["lightgbm"] = agg
                save_results("lightgbm", metrics)
                log_experiment(
                    "lightgbm", metrics,
                    training_config={"type": "lightgbm"},
                    cli_flags=cli_flags,
                    training_time_s=time.time() - t0,
                    seed=RANDOM_SEED,
                )
        except Exception as e:
            logger.warning(f"  LightGBM failed: {e}")

        # Neural baselines
        config_overrides = {}
        if args.quick:
            config_overrides = {"max_epochs": 10, "early_stopping_patience": 5}
        for nn_baseline in ["single_transformer", "lstm_concat"]:
            try:
                logger.info(f"\n{'═'*60}\n  BASELINE: {nn_baseline}\n{'═'*60}")
                t0 = time.time()
                metrics = run_baseline(
                    nn_baseline, dataset, folds,
                    config_overrides=config_overrides if config_overrides else None,
                )
                if metrics:
                    agg = aggregate_fold_metrics(metrics)
                    all_results[nn_baseline] = agg
                    save_results(nn_baseline, metrics)
                    log_experiment(
                        nn_baseline, metrics,
                        model_config={**MODEL_CONFIG, "type": nn_baseline},
                        training_config={**TRAINING_CONFIG, **config_overrides},
                        cli_flags=cli_flags,
                        training_time_s=time.time() - t0,
                        seed=RANDOM_SEED,
                    )
            except Exception as e:
                logger.warning(f"  {nn_baseline} failed: {e}")

        print_results_table(all_results, "All Baseline Results")
        print_leaderboard()
        return

    elif args.experiment == "all_ablations":
        all_results = {}
        config_overrides = {}
        if args.quick:
            config_overrides = {"max_epochs": 10, "early_stopping_patience": 5}

        for ablation_name, disabled in ABLATION_CONFIGS.items():
            logger.info(f"\n{'═'*60}\n  ABLATION: {ablation_name} (disabling {disabled})\n{'═'*60}")
            try:
                t0 = time.time()
                metrics = run_mcat_experiment(
                    dataset, folds,
                    experiment_name=ablation_name,
                    n_sent_features=dataset.X_sent.shape[2],
                    n_fund_features=dataset.X_fund.shape[1],
                    n_macro_features=dataset.X_macro.shape[2],
                    config_overrides=config_overrides,
                    disable_modalities=disabled,
                )
                if metrics:
                    agg = aggregate_fold_metrics(metrics)
                    all_results[ablation_name] = agg
                    save_results(ablation_name, metrics)
                    log_experiment(
                        ablation_name, metrics,
                        model_config={**MODEL_CONFIG, "disabled_modalities": disabled},
                        training_config={**TRAINING_CONFIG, **config_overrides},
                        cli_flags=cli_flags,
                        n_params=_count_mcat_params(disabled),
                        training_time_s=time.time() - t0,
                        seed=RANDOM_SEED,
                    )
            except Exception as e:
                logger.warning(f"  {ablation_name} failed: {e}")

        print_results_table(all_results, "Ablation Results")
        print_leaderboard()
        return

    else:
        logger.error(f"Unknown experiment: {args.experiment}")
        return

    training_time = time.time() - experiment_start

    # ─── Step 4: Save, log, and report results ───
    if not fold_metrics:
        logger.error("No folds produced results! Check data coverage.")
        return

    save_results(args.experiment, fold_metrics)

    # Determine effective configs for logging
    is_mcat = args.experiment in ("quick_test", "full_mcat") or args.experiment in ABLATION_CONFIGS
    if is_mcat:
        disable_modalities = ABLATION_CONFIGS.get(args.experiment, [])
        effective_model_config = {**MODEL_CONFIG}
        if disable_modalities:
            effective_model_config["disabled_modalities"] = disable_modalities
        effective_training_config = {**TRAINING_CONFIG, **config_overrides}
        n_params = _count_mcat_params(disable_modalities)
    else:
        effective_model_config = {"type": args.experiment}
        effective_training_config = {**TRAINING_CONFIG}
        n_params = None

    log_experiment(
        experiment=args.experiment,
        fold_metrics=fold_metrics,
        model_config=effective_model_config,
        training_config=effective_training_config,
        cli_flags=cli_flags,
        n_params=n_params,
        training_time_s=training_time,
        seed=RANDOM_SEED,
    )

    agg = aggregate_fold_metrics(fold_metrics)
    print_results_table({args.experiment: agg}, f"Results: {args.experiment}")
    print_leaderboard()


if __name__ == "__main__":
    main()
