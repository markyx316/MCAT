"""
run_comparison.py — Run MCAT vs all baselines on simulated data.
=================================================================
Comprehensive comparison experiment:
  1. Build simulated market dataset (3 tickers, full date range)
  2. Run all models on the same walk-forward folds
  3. Generate comparison table and figures
  4. Save everything with provenance

Usage:
    python run_comparison.py                    # Full comparison (5 models, 3+ folds)
    python run_comparison.py --quick            # Quick test (2 models, 1 fold)
    python run_comparison.py --tickers 2        # Fewer tickers
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TICKERS, TRAINING_CONFIG, RANDOM_SEED, RESULTS_DIR
from data.provenance import provenance
from data.simulate_market import build_simulated_dataset
from training.walk_forward import generate_walk_forward_folds, WalkForwardFold
from evaluation.metrics import (
    compute_metrics, aggregate_fold_metrics, print_results_table,
    FoldMetrics, paired_t_test,
)
from utils import setup_logger, set_seed

logger = setup_logger(__name__)


def run_model_on_folds(
    model_name: str,
    dataset,
    folds: List[WalkForwardFold],
    max_epochs: int = 30,
) -> List[FoldMetrics]:
    """Run a single model across walk-forward folds."""

    if model_name == "historical_mean":
        from baselines.baselines import run_historical_mean
        return run_historical_mean(dataset, folds)

    elif model_name == "ridge":
        from baselines.baselines import run_ridge_regression
        return run_ridge_regression(dataset, folds)

    elif model_name == "mcat":
        from run_experiment import run_mcat_experiment
        return run_mcat_experiment(
            dataset, folds,
            experiment_name="mcat",
            n_sent_features=dataset.X_sent.shape[2],
            n_fund_features=dataset.X_fund.shape[1],
            n_macro_features=dataset.X_macro.shape[2],
            config_overrides={
                "max_epochs": max_epochs,
                "early_stopping_patience": min(10, max_epochs // 2),
            },
        )

    elif model_name == "price_only_transformer":
        from baselines.baselines import run_single_transformer
        return run_single_transformer(
            dataset, folds,
            config_overrides={
                "max_epochs": max_epochs,
                "early_stopping_patience": min(10, max_epochs // 2),
            },
        )

    elif model_name == "lstm_concat":
        from baselines.baselines import run_lstm_concat
        return run_lstm_concat(
            dataset, folds,
            config_overrides={
                "max_epochs": max_epochs,
                "early_stopping_patience": min(10, max_epochs // 2),
            },
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Run MCAT vs baselines comparison")
    parser.add_argument("--quick", action="store_true", help="Quick mode (1 fold, 10 epochs)")
    parser.add_argument("--tickers", type=int, default=3, help="Number of tickers (1-15)")
    parser.add_argument("--max-epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of walk-forward folds")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    start_time = time.time()

    # ─── Configuration ───
    ticker_list = TICKERS[:args.tickers]
    max_epochs = 10 if args.quick else args.max_epochs
    n_folds = 1 if args.quick else args.n_folds

    models_to_run = ["historical_mean", "ridge"]
    if not args.quick:
        models_to_run += ["price_only_transformer", "lstm_concat", "mcat"]
    else:
        models_to_run += ["mcat"]

    logger.info("=" * 60)
    logger.info("  MCAT vs BASELINES COMPARISON")
    logger.info("=" * 60)
    logger.info(f"  Tickers: {ticker_list}")
    logger.info(f"  Models: {models_to_run}")
    logger.info(f"  Max epochs: {max_epochs}")
    logger.info(f"  Folds: {n_folds}")
    logger.info("")

    # ─── Step 1: Build dataset ───
    logger.info("─── Building simulated market dataset ───")
    dataset, prov_report = build_simulated_dataset(
        tickers=ticker_list,
        denoise=False,  # Skip denoising for speed in comparison
        seed=RANDOM_SEED,
    )

    # ─── Step 2: Select folds ───
    all_folds = generate_walk_forward_folds()
    dates = pd.to_datetime(dataset.dates)
    date_max = dates.max()
    available = [
        f for f in all_folds
        if f.test_end <= date_max + pd.Timedelta(days=30)
    ]
    use_folds = available[:n_folds]
    logger.info(f"\nUsing {len(use_folds)} folds out of {len(available)} available")

    # ─── Step 3: Run all models ───
    all_results = {}
    all_fold_metrics = {}

    for model_name in models_to_run:
        logger.info(f"\n{'━' * 50}")
        logger.info(f"  Running: {model_name}")
        logger.info(f"{'━' * 50}")

        try:
            fold_metrics = run_model_on_folds(
                model_name, dataset, use_folds, max_epochs,
            )
            if fold_metrics:
                agg = aggregate_fold_metrics(fold_metrics)
                all_results[model_name] = agg
                all_fold_metrics[model_name] = fold_metrics
                logger.info(f"  {model_name}: IC={agg['ic']['mean']:.4f}±{agg['ic']['std']:.4f}")
            else:
                logger.warning(f"  {model_name}: No results")
        except Exception as e:
            logger.error(f"  {model_name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ─── Step 4: Print comparison table ───
    if all_results:
        print_results_table(all_results, "MCAT vs Baselines (Simulated Market)")

    # ─── Step 5: Statistical tests ───
    if "mcat" in all_fold_metrics and len(all_fold_metrics["mcat"]) > 1:
        logger.info("\n─── Statistical Tests ───")
        mcat_metrics = all_fold_metrics["mcat"]

        for baseline_name, baseline_metrics in all_fold_metrics.items():
            if baseline_name == "mcat":
                continue
            if len(baseline_metrics) != len(mcat_metrics):
                continue

            for metric in ["ic", "mae"]:
                t_stat, p_val = paired_t_test(mcat_metrics, baseline_metrics, metric)
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else "n.s."
                logger.info(f"  MCAT vs {baseline_name} ({metric}): t={t_stat:.3f}, p={p_val:.4f} {sig}")

    # ─── Step 6: Save results ───
    output_dir = RESULTS_DIR / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "experiment": "mcat_vs_baselines",
        "tickers": ticker_list,
        "n_folds": len(use_folds),
        "max_epochs": max_epochs,
        "models": {},
        "data_provenance": provenance.to_dict(),
    }

    for model_name, agg in all_results.items():
        results_data["models"][model_name] = {
            metric: {"mean": float(v["mean"]), "std": float(v["std"])}
            for metric, v in agg.items()
        }

    out_path = output_dir / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"\nResults saved → {out_path}")

    provenance.save()

    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  COMPARISON COMPLETE ({elapsed:.0f}s / {elapsed/60:.1f}min)")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
