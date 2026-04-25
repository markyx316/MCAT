"""
run_hp_search.py — Systematic Hyperparameter Search for MCAT
=============================================================
Two-phase search:
  Phase 1 (screening): Run N random configs on 1 fold (~40s each)
  Phase 2 (validation): Run top-K configs on all 3 folds (~2 min each)

Usage:
  # Phase 1: screen 40 random configs on Fold 1 (fastest fold)
  python run_hp_search.py --n-configs 40

  # Phase 2: validate top 5 from phase 1 on all 3 folds
  python run_hp_search.py --phase2 --top-k 5

  # Full pipeline: screen + validate in one go
  python run_hp_search.py --n-configs 40 --phase2 --top-k 5

  # Quick test: 3 configs, 1 fold
  python run_hp_search.py --n-configs 3
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MODEL_CONFIG, TRAINING_CONFIG, RANDOM_SEED,
    N_PRICE_FEATURES, N_SENTIMENT_FEATURES, RESULTS_DIR,
)
from utils import setup_logger, set_seed
from evaluation.metrics import compute_metrics, aggregate_fold_metrics
from training.walk_forward import generate_focused_folds
from experiment_log import log_experiment, print_leaderboard

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────
# SEARCH SPACE
# ─────────────────────────────────────────────────────────────
# Informed by experiments:
#   - dropout=0.05 >> 0.20 (cross-attention needs stable projections)
#   - lr=1e-4 with warmup=5 was best so far
#   - conv_kernel=5 > 3 (captures weekly patterns)
#   - model stops at epoch 1-3, so warmup length matters a lot

SEARCH_SPACE = {
    # Training hyperparameters
    "learning_rate":      {"type": "log_uniform",  "low": 3e-5,  "high": 2e-4},
    "dropout":            {"type": "uniform",      "low": 0.01,  "high": 0.15},
    "weight_decay":       {"type": "log_uniform",  "low": 0.003, "high": 0.02},
    "warmup_epochs":      {"type": "choice",       "values": [4, 5, 6, 7]},
    "huber_delta":        {"type": "choice",       "values": [1.0, 1.1, 1.2, 1.3]},
    "gradient_clip_norm": {"type": "choice",       "values": [0.5, 1.0, 1.5]},
    "batch_size":         {"type": "choice",       "values": [32, 64, 128]},

    # Architecture hyperparameters
    "causal_conv_kernel": {"type": "choice",       "values": [5, 6, 7]},

    # Model dimensions — pre-validated (d_model, n_heads, d_ff) tuples
    # to ensure n_heads always divides d_model (d_k = d_model / n_heads).
    # Format: (d_model, n_heads, d_ff)
    "model_dims":         {"type": "choice",       "values": [
        (32, 2, 64),     # ~80K params, d_k=16
        (128, 4, 512),     # ~170K params, d_k=12
        (64, 4, 128),    # ~301K params, d_k=16 (current default)
        (64, 4, 256),    # ~340K params, d_k=16, wider FFN
    ]},
}

# Which keys belong to MODEL_CONFIG vs TRAINING_CONFIG
# model_dims is handled specially in split_config (exploded into d_model/n_heads/d_ff)
MODEL_KEYS = {"dropout", "causal_conv_kernel", "d_model", "n_heads", "d_ff"}
TRAINING_KEYS = {
    "learning_rate", "weight_decay", "warmup_epochs",
    "huber_delta", "gradient_clip_norm", "batch_size",
}


def sample_config(rng: np.random.RandomState) -> dict:
    """Sample a random hyperparameter configuration."""
    config = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "uniform":
            config[name] = float(rng.uniform(spec["low"], spec["high"]))
        elif spec["type"] == "log_uniform":
            log_low = np.log(spec["low"])
            log_high = np.log(spec["high"])
            config[name] = float(np.exp(rng.uniform(log_low, log_high)))
        elif spec["type"] == "choice":
            config[name] = spec["values"][rng.randint(len(spec["values"]))]

    # Explode model_dims tuple into individual keys
    if "model_dims" in config:
        d_model, n_heads, d_ff = config.pop("model_dims")
        config["d_model"] = d_model
        config["n_heads"] = n_heads
        config["d_ff"] = d_ff

    return config


def split_config(hp: dict) -> tuple:
    """Split HP config into model overrides and training overrides."""
    model_overrides = {k: v for k, v in hp.items() if k in MODEL_KEYS}
    training_overrides = {k: v for k, v in hp.items() if k in TRAINING_KEYS}
    return model_overrides, training_overrides


def config_to_str(hp: dict) -> str:
    """Short human-readable string for a config."""
    return (
        f"lr={hp['learning_rate']:.1e} "
        f"do={hp['dropout']:.3f} "
        f"wd={hp['weight_decay']:.4f} "
        f"wu={hp['warmup_epochs']} "
        f"hd={hp['huber_delta']} "
        f"gc={hp['gradient_clip_norm']} "
        f"bs={hp['batch_size']} "
        f"ck={hp['causal_conv_kernel']} "
        f"dm={hp.get('d_model', 64)}/{hp.get('n_heads', 4)}/{hp.get('d_ff', 128)}"
    )


# ─────────────────────────────────────────────────────────────
# SINGLE CONFIG EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_config(
    hp: dict,
    dataset,
    folds: list,
    config_id: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Train and evaluate MCAT with a specific hyperparameter config.

    Returns dict with per-fold metrics and aggregate scores.
    """
    import torch
    from model.mcat import MCAT, count_parameters
    from training.trainer import Trainer
    from run_experiment import build_dataloaders

    model_overrides, training_overrides = split_config(hp)

    # Build model config
    model_config = {**MODEL_CONFIG, **model_overrides}
    train_config = {**TRAINING_CONFIG, **training_overrides}
    batch_size = train_config.pop("batch_size", TRAINING_CONFIG["batch_size"])

    fold_metrics_list = []
    fold_details = []

    for fold in folds:
        set_seed(RANDOM_SEED + fold.fold_num)

        train_loader, val_loader, test_loader = build_dataloaders(
            dataset, fold, batch_size=batch_size,
        )

        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        n_test = len(test_loader.dataset)

        if n_train == 0 or n_val == 0 or n_test == 0:
            continue

        # Build model with custom config
        model = MCAT(
            n_price_features=dataset.X_price.shape[2],
            n_sent_features=dataset.X_sent.shape[2],
            n_fund_features=dataset.X_fund.shape[1],
            n_macro_features=dataset.X_macro.shape[2],
            config=model_config,
        )

        # Train
        trainer = Trainer(model, config=train_config)
        history = trainer.train_fold(
            train_loader, val_loader, fold_num=fold.fold_num,
        )

        # Evaluate
        test_loss, test_preds, test_labels = trainer.validate(test_loader)
        metrics = compute_metrics(test_preds, test_labels, fold_num=fold.fold_num)
        fold_metrics_list.append(metrics)

        fold_details.append({
            "fold": fold.fold_num,
            "r2": metrics.r2,
            "ic": metrics.ic,
            "da": metrics.directional_accuracy,
            "skill": metrics.skill_score,
            "best_epoch": history.get("best_epoch", -1) if isinstance(history, dict) else -1,
            "n_train": n_train,
            "n_test": n_test,
        })

    if not fold_metrics_list:
        return {"config_id": config_id, "hp": hp, "failed": True}

    # Aggregate
    agg_raw = aggregate_fold_metrics(fold_metrics_list)

    # Flatten: agg_raw["ic"]["mean"] → agg["ic_mean"], etc.
    agg = {}
    for key in ["r2", "ic", "directional_accuracy", "skill_score"]:
        agg[f"{key}_mean"] = float(agg_raw[key]["mean"])
        agg[f"{key}_std"] = float(agg_raw[key]["std"])

    # Convenience aliases
    agg["da_mean"] = agg["directional_accuracy_mean"]
    agg["skill_mean"] = agg["skill_score_mean"]

    # Composite score: IC + R² + (DA - 0.5)
    # - IC: ranking ability (can the model sort stocks correctly?)
    # - R²: calibration quality (are magnitude predictions accurate?)
    # - DA - 0.5: directional accuracy above chance
    # All three are in similar numeric ranges (~±0.1 in our data).
    score = agg["ic_mean"] + agg["r2_mean"] + (agg["da_mean"] - 0.5)

    result = {
        "config_id": config_id,
        "hp": hp,
        "hp_str": config_to_str(hp),
        "score": score,
        "r2_mean": agg["r2_mean"],
        "ic_mean": agg["ic_mean"],
        "da_mean": agg["da_mean"],
        "skill_mean": agg["skill_mean"],
        "n_folds": len(fold_metrics_list),
        "folds": fold_details,
        "failed": False,
        # For central experiment log (not serialized to hp_phase JSON)
        "_fold_metrics_list": fold_metrics_list,
        "_model_config": model_config,
        "_train_config": {**train_config, "batch_size": batch_size},
        "_n_params": count_parameters(model),
    }

    if verbose:
        logger.info(
            f"  Config {config_id:>3d}: score={score:+.4f} "
            f"IC={agg['ic_mean']:+.4f} DA={agg['da_mean']:.1%} "
            f"R²={agg['r2_mean']:+.4f} Skill={agg['skill_mean']:+.4f} "
            f"| {config_to_str(hp)}"
        )

    return result


# ─────────────────────────────────────────────────────────────
# REAL-TIME RESULT SAVING
# ─────────────────────────────────────────────────────────────

def _make_serializable(results: list) -> list:
    """Convert numpy types in results list for JSON serialization."""
    serializable = []
    for r in results:
        sr = {}
        for k, v in r.items():
            # Skip internal keys (FoldMetrics objects, model configs)
            if k.startswith("_"):
                continue
            if isinstance(v, (np.floating, np.integer)):
                sr[k] = float(v)
            elif isinstance(v, dict):
                sr[k] = {
                    kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                    for kk, vv in v.items()
                }
            elif isinstance(v, list):
                sr[k] = [
                    {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                     for kk, vv in item.items()} if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                sr[k] = v
        serializable.append(sr)
    return serializable


def save_results_incremental(results: list, out_path: Path):
    """
    Save results list to JSON after each config completes.

    Results are sorted by score (best first) before writing,
    so the file always shows the current leaderboard.
    """
    sorted_results = sorted(
        results,
        key=lambda r: r.get("score", -999),
        reverse=True,
    )
    serializable = _make_serializable(sorted_results)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────
# MAIN SEARCH LOGIC
# ─────────────────────────────────────────────────────────────

def run_phase1(dataset, folds, n_configs: int, seed: int = 42,
               out_path: Path = None) -> list:
    """
    Phase 1: Screen N random configs on a single fold.

    Results are saved to out_path after EACH config completes,
    so progress is never lost if the job is interrupted.
    """
    if out_path is None:
        out_path = RESULTS_DIR / "tables" / "hp_phase1.json"

    logger.info(f"\n{'═'*70}")
    logger.info(f"  PHASE 1: Screening {n_configs} random configs on Fold {folds[0].fold_num}")
    logger.info(f"  Results saving to: {out_path}")
    logger.info(f"{'═'*70}")

    rng = np.random.RandomState(seed)
    results = []

    for i in range(n_configs):
        hp = sample_config(rng)
        t0 = time.time()
        result = evaluate_config(hp, dataset, folds, config_id=i)
        result["time_seconds"] = time.time() - t0
        results.append(result)

        # Save after each config (sorted by score, best first)
        save_results_incremental(results, out_path)

    # Sort by score descending
    results.sort(key=lambda r: r.get("score", -999), reverse=True)

    # Print leaderboard
    logger.info(f"\n{'═'*70}")
    logger.info(f"  PHASE 1 LEADERBOARD (top 10)")
    logger.info(f"{'═'*70}")
    logger.info(f"  {'Rank':<5} {'Score':>7} {'IC':>7} {'DA':>6} {'R²':>7} {'Skill':>7}  Config")
    logger.info(f"  {'─'*85}")
    for rank, r in enumerate(results[:10]):
        if r.get("failed"):
            continue
        logger.info(
            f"  {rank+1:<5d} {r['score']:>+.4f} {r['ic_mean']:>+.4f} "
            f"{r['da_mean']:>5.1%} {r['r2_mean']:>+.4f} {r['skill_mean']:>+.4f}  "
            f"{r['hp_str']}"
        )

    logger.info(f"\n  Phase 1 results saved → {out_path}")
    return results


def run_phase2(dataset, folds, phase1_results: list, top_k: int = 5,
               out_path: Path = None, cli_flags: dict = None) -> list:
    """
    Phase 2: Validate top-K configs from Phase 1 on all 3 folds.

    Results are saved to out_path after EACH config completes.
    Each result is also logged to the central experiment ledger.
    """
    if out_path is None:
        out_path = RESULTS_DIR / "tables" / "hp_phase2.json"

    # Get top-K non-failed configs
    candidates = [r for r in phase1_results if not r.get("failed")][:top_k]

    logger.info(f"\n{'═'*70}")
    logger.info(f"  PHASE 2: Validating top {len(candidates)} configs on all {len(folds)} folds")
    logger.info(f"  Results saving to: {out_path}")
    logger.info(f"{'═'*70}")

    results = []
    for rank, candidate in enumerate(candidates):
        hp = candidate["hp"]
        logger.info(f"\n  ── Candidate {rank+1}/{len(candidates)}: {config_to_str(hp)} ──")

        t0 = time.time()
        result = evaluate_config(hp, dataset, folds, config_id=candidate["config_id"])
        elapsed = time.time() - t0
        result["time_seconds"] = elapsed
        result["phase1_score"] = candidate["score"]
        results.append(result)

        # Save to HP search results file
        save_results_incremental(results, out_path)

        # Log to central experiment ledger (only non-failed 3-fold runs)
        if not result.get("failed") and "_fold_metrics_list" in result:
            log_flags = {
                "source": "hp_search_phase2",
                "config_id": result["config_id"],
                **(cli_flags or {}),
            }
            log_experiment(
                experiment=f"hp_search_{config_to_str(hp)[:40]}",
                fold_metrics=result["_fold_metrics_list"],
                model_config=result["_model_config"],
                training_config=result["_train_config"],
                cli_flags=log_flags,
                n_params=result.get("_n_params"),
                training_time_s=elapsed,
                seed=RANDOM_SEED,
            )

    # Sort by score
    results.sort(key=lambda r: r.get("score", -999), reverse=True)

    # Print final leaderboard
    logger.info(f"\n{'═'*70}")
    logger.info(f"  PHASE 2 FINAL LEADERBOARD")
    logger.info(f"{'═'*70}")
    logger.info(f"  {'Rank':<5} {'Score':>7} {'IC':>7} {'DA':>6} {'R²':>7} {'Skill':>7}  Config")
    logger.info(f"  {'─'*85}")
    for rank, r in enumerate(results):
        if r.get("failed"):
            continue
        logger.info(
            f"  {rank+1:<5d} {r['score']:>+.4f} {r['ic_mean']:>+.4f} "
            f"{r['da_mean']:>5.1%} {r['r2_mean']:>+.4f} {r['skill_mean']:>+.4f}  "
            f"{r['hp_str']}"
        )

    # Print best config in copy-paste format
    if results and not results[0].get("failed"):
        best = results[0]
        logger.info(f"\n{'═'*70}")
        logger.info(f"  BEST CONFIG (copy to config.py)")
        logger.info(f"{'═'*70}")
        hp = best["hp"]
        logger.info(f"  MODEL_CONFIG overrides:")
        logger.info(f"    \"d_model\": {hp.get('d_model', 64)},")
        logger.info(f"    \"n_heads\": {hp.get('n_heads', 4)},")
        logger.info(f"    \"d_ff\": {hp.get('d_ff', 128)},")
        logger.info(f"    \"dropout\": {hp['dropout']:.4f},")
        logger.info(f"    \"causal_conv_kernel\": {hp['causal_conv_kernel']},")
        logger.info(f"  TRAINING_CONFIG overrides:")
        logger.info(f"    \"learning_rate\": {hp['learning_rate']:.2e},")
        logger.info(f"    \"weight_decay\": {hp['weight_decay']:.4f},")
        logger.info(f"    \"warmup_epochs\": {hp['warmup_epochs']},")
        logger.info(f"    \"huber_delta\": {hp['huber_delta']},")
        logger.info(f"    \"gradient_clip_norm\": {hp['gradient_clip_norm']},")
        logger.info(f"    \"batch_size\": {hp['batch_size']},")
        logger.info(f"  Performance: IC={best['ic_mean']:+.4f} DA={best['da_mean']:.1%} "
                     f"R²={best['r2_mean']:+.4f} Skill={best['skill_mean']:+.4f}")

    logger.info(f"\n  Phase 2 results saved → {out_path}")
    print_leaderboard()
    return results


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Systematic hyperparameter search for MCAT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Screen 40 configs on Fold 1 (~20 min)
  python run_hp_search.py --n-configs 40

  # Screen + validate top 5 on all 3 folds (~30 min total)
  python run_hp_search.py --n-configs 40 --phase2 --top-k 5

  # Quick test: 5 configs
  python run_hp_search.py --n-configs 5

  # Phase 2 only from saved results
  python run_hp_search.py --phase2 --from-file results/tables/hp_phase1.json --top-k 5
        """,
    )
    parser.add_argument("--n-configs", type=int, default=40,
                        help="Number of random configs to screen (Phase 1)")
    parser.add_argument("--phase2", action="store_true",
                        help="Run Phase 2 validation on all 3 folds")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top configs to validate in Phase 2")
    parser.add_argument("--screening-fold", type=int, default=1,
                        help="Which fold to use for Phase 1 screening (0, 1, or 2)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed for HP config sampling (default: {RANDOM_SEED}, "
                             f"same as RANDOM_SEED in config.py). Model training always "
                             f"uses RANDOM_SEED={RANDOM_SEED} from config.py.")
    parser.add_argument("--from-file", type=str, default=None,
                        help="Load Phase 1 results from file (skip Phase 1)")
    parser.add_argument("--fnspid", type=str, default=None,
                        help="Path to FNSPID CSV (optional, uses cache)")
    parser.add_argument("--lookback", type=int, default=None,
                        help="Lookback window in trading days (default: 60)")
    parser.add_argument("--train-years", type=float, default=None,
                        help="Training window in years per fold. Default: expanding "
                             "from 2017. E.g., --train-years 3 uses sliding 3yr window.")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    # ─── Build dataset ───
    logger.info("Building dataset...")
    from data.build_dataset import build_full_dataset
    dataset, report = build_full_dataset(
        fnspid_path=args.fnspid,
        lookback=args.lookback,
    )

    # ─── Generate folds ───
    all_folds = generate_focused_folds(train_years=args.train_years)

    # ─── Phase 1: Screening ───
    phase1_path = RESULTS_DIR / "tables" / "hp_phase1.json"

    if args.from_file:
        logger.info(f"Loading Phase 1 results from {args.from_file}")
        with open(args.from_file) as f:
            phase1_results = json.load(f)
    else:
        screening_folds = [all_folds[args.screening_fold]]
        logger.info(f"Screening fold: {screening_folds[0]}")

        phase1_results = run_phase1(
            dataset, screening_folds, args.n_configs,
            seed=args.seed, out_path=phase1_path,
        )

    # ─── Phase 2: Validation ───
    if args.phase2:
        phase2_path = RESULTS_DIR / "tables" / "hp_phase2.json"

        hp_cli_flags = {
            "lookback": args.lookback,
            "train_years": args.train_years,
            "hp_sampling_seed": args.seed,
        }

        phase2_results = run_phase2(
            dataset, all_folds, phase1_results,
            top_k=args.top_k, out_path=phase2_path,
            cli_flags=hp_cli_flags,
        )


if __name__ == "__main__":
    main()
