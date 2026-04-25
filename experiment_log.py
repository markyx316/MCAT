"""
experiment_log.py — Central Experiment Ledger
==============================================
Records every experiment run with full configuration and results
in a single JSON file, sorted by composite score (best first).

Automatic deduplication: after each log_experiment() call, exact
duplicates (same config + same results) are silently removed,
keeping the latest timestamp. Manual dedup with verbose reporting
is available via: python experiment_log.py --dedup

Usage:
    from experiment_log import log_experiment, print_leaderboard

    # After an experiment completes:
    log_experiment(
        experiment="full_mcat",
        fold_metrics=fold_metrics_list,
        model_config=effective_model_config,
        training_config=effective_training_config,
        cli_flags={"lookback": 60, "train_years": None},
        n_params=300929,
    )

    # View top results:
    print_leaderboard(top_n=20)

    # Manual dedup with report:
    python experiment_log.py --dedup
"""

import json
import time
import hashlib
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import RESULTS_DIR
from evaluation.metrics import FoldMetrics, aggregate_fold_metrics
from utils import setup_logger

logger = setup_logger(__name__)

LOG_PATH = RESULTS_DIR / "experiment_log.json"


def _to_serializable(obj):
    """Recursively convert numpy types for JSON."""
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def compute_score(agg: dict) -> float:
    """
    Composite score: IC + R² + (DA - 0.5)

    Balances ranking ability (IC), calibration quality (R²),
    and directional accuracy above chance (DA - 0.5).
    """
    ic = agg.get("ic_mean", 0)
    r2 = agg.get("r2_mean", 0)
    da = agg.get("da_mean", 0.5)
    return ic + r2 + (da - 0.5)


def log_experiment(
    experiment: str,
    fold_metrics: List[FoldMetrics],
    model_config: dict = None,
    training_config: dict = None,
    cli_flags: dict = None,
    n_params: int = None,
    training_time_s: float = None,
    seed: int = None,
    notes: str = None,
) -> dict:
    """
    Append an experiment result to the central log.

    The log file is sorted by composite score (best first) after each write.

    Args:
        experiment: Name (e.g. "full_mcat", "ablate_no_sentiment").
        fold_metrics: List of FoldMetrics from evaluation.
        model_config: Effective MODEL_CONFIG dict used for this run.
        training_config: Effective TRAINING_CONFIG dict used for this run.
        cli_flags: Dict of CLI flags (lookback, train_years, fold_mode, etc.).
        n_params: Total trainable parameters.
        training_time_s: Total wall-clock training time in seconds.
        seed: Random seed used for reproducibility (RANDOM_SEED from config).
        notes: Optional free-text notes.

    Returns:
        The entry dict that was appended.
    """
    # ─── Aggregate metrics ───
    agg_raw = aggregate_fold_metrics(fold_metrics)
    agg = {}
    for key in ["r2", "ic", "directional_accuracy", "skill_score", "mae", "rmse"]:
        if key in agg_raw:
            agg[f"{key}_mean"] = float(agg_raw[key]["mean"])
            agg[f"{key}_std"] = float(agg_raw[key]["std"])

    # Convenience aliases
    agg["da_mean"] = agg.get("directional_accuracy_mean", 0.5)
    agg["da_std"] = agg.get("directional_accuracy_std", 0)
    agg["skill_mean"] = agg.get("skill_score_mean", 0)
    agg["skill_std"] = agg.get("skill_score_std", 0)

    score = compute_score(agg)

    # ─── Per-fold details ───
    fold_details = []
    for m in fold_metrics:
        fold_details.append({
            "fold": int(m.fold_num),
            "r2": float(m.r2),
            "ic": float(m.ic),
            "da": float(m.directional_accuracy),
            "skill": float(m.skill_score),
            "mae": float(m.mae),
            "rmse": float(m.rmse),
            "n_samples": int(m.n_samples),
        })

    # ─── Build entry ───
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": experiment,
        "score": round(score, 6),
        "metrics": {
            "ic_mean": round(agg.get("ic_mean", 0), 6),
            "r2_mean": round(agg.get("r2_mean", 0), 6),
            "da_mean": round(agg.get("da_mean", 0.5), 4),
            "skill_mean": round(agg.get("skill_mean", 0), 6),
            "mae_mean": round(agg.get("mae_mean", 0), 4),
            "rmse_mean": round(agg.get("rmse_mean", 0), 4),
        },
        "folds": fold_details,
        "model_config": _to_serializable(model_config or {}),
        "training_config": _to_serializable(training_config or {}),
        "cli_flags": _to_serializable(cli_flags or {}),
        "meta": {
            "n_params": n_params,
            "n_folds": len(fold_metrics),
            "total_test_samples": sum(m.n_samples for m in fold_metrics),
            "training_time_s": round(training_time_s, 1) if training_time_s else None,
            "seed": seed,
        },
        "notes": notes,
    }

    # ─── Load existing log, append, sort, save ───
    log = _load_log()
    log.append(entry)
    log.sort(key=lambda e: e.get("score", -999), reverse=True)
    _save_log(log)

    # ─── Deduplicate (silently removes exact duplicates) ───
    n_removed = deduplicate_log(verbose=False)

    # ─── Print position in leaderboard ───
    log = _load_log()  # reload after dedup
    rank = next((i for i, e in enumerate(log)
                 if _compute_fingerprint(e) == _compute_fingerprint(entry)), 0) + 1
    status = f" ({n_removed} dup removed)" if n_removed else ""
    logger.info(
        f"  📊 Logged: {experiment} → rank #{rank}/{len(log)}{status} "
        f"(score={score:+.4f}, IC={agg.get('ic_mean',0):+.4f}, "
        f"DA={agg.get('da_mean',0.5):.1%}, R²={agg.get('r2_mean',0):+.4f})"
    )

    return entry


def _load_log() -> list:
    """Load existing log or create empty."""
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning(f"  Corrupt log at {LOG_PATH}, starting fresh")
            return []
    return []


def _save_log(log: list):
    """Save log to disk."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────

def _compute_fingerprint(entry: dict) -> str:
    """
    Compute a deterministic fingerprint for an experiment entry.

    The fingerprint captures the effective configuration AND results,
    so two runs are duplicates only if they used the same config and
    produced the same metrics. We deliberately EXCLUDE:
      - timestamp (varies between runs)
      - training_time_s (varies slightly)
      - notes (free text)

    Includes:
      - score (6 decimal places)
      - per-fold IC, R², DA (6 decimal places)
      - model_config (sorted)
      - training_config (sorted)
      - cli_flags (sorted, excluding 'source' which is metadata)
    """
    parts = []

    # Score
    parts.append(f"score={entry.get('score', 0):.6f}")

    # Per-fold metrics (order by fold number for determinism)
    folds = sorted(entry.get("folds", []), key=lambda f: f.get("fold", 0))
    for f in folds:
        parts.append(
            f"f{f.get('fold',0)}_ic={f.get('ic',0):.6f}_"
            f"r2={f.get('r2',0):.6f}_da={f.get('da',0):.6f}"
        )

    # Model config (sorted keys for determinism)
    mc = entry.get("model_config", {})
    parts.append("mc=" + json.dumps(mc, sort_keys=True, default=str))

    # Training config (sorted keys for determinism)
    tc = entry.get("training_config", {})
    parts.append("tc=" + json.dumps(tc, sort_keys=True, default=str))

    # CLI flags (exclude 'source' which is just metadata about where it came from)
    cf = {k: v for k, v in entry.get("cli_flags", {}).items() if k != "source"}
    parts.append("cf=" + json.dumps(cf, sort_keys=True, default=str))

    fingerprint_str = "|".join(parts)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]


def deduplicate_log(verbose: bool = True) -> int:
    """
    Remove duplicate entries from the experiment log.

    Two entries are duplicates if they have the same fingerprint
    (same config + same results). When duplicates are found, the
    entry with the LATEST timestamp is kept.

    Args:
        verbose: If True, print details of each duplicate removed.

    Returns:
        Number of duplicates removed.
    """
    log = _load_log()
    if not log:
        if verbose:
            logger.info("  Log is empty, nothing to deduplicate.")
        return 0

    # Group entries by fingerprint
    fingerprints = {}
    for i, entry in enumerate(log):
        fp = _compute_fingerprint(entry)
        if fp not in fingerprints:
            fingerprints[fp] = []
        fingerprints[fp].append((i, entry))

    # Find duplicates (fingerprints with >1 entry)
    duplicates_found = 0
    indices_to_remove = set()

    for fp, entries in fingerprints.items():
        if len(entries) <= 1:
            continue

        # Sort by timestamp descending — keep the latest
        entries.sort(
            key=lambda ie: ie[1].get("timestamp", ""),
            reverse=True,
        )

        kept = entries[0]
        removed = entries[1:]
        duplicates_found += len(removed)

        for idx, entry in removed:
            indices_to_remove.add(idx)

        if verbose:
            exp_name = kept[1].get("experiment", "?")[:40]
            score = kept[1].get("score", 0)
            kept_ts = kept[1].get("timestamp", "?")
            logger.info(
                f"  🔍 Duplicate group (fingerprint {fp}): "
                f"{len(entries)} copies of \"{exp_name}\" (score={score:+.4f})"
            )
            logger.info(f"     Kept:    {kept_ts}")
            for idx, entry in removed:
                logger.info(f"     Removed: {entry.get('timestamp', '?')}")

    if duplicates_found == 0:
        if verbose:
            logger.info("  ✓ No duplicates found.")
        return 0

    # Remove duplicates (iterate in reverse to preserve indices)
    new_log = [e for i, e in enumerate(log) if i not in indices_to_remove]
    new_log.sort(key=lambda e: e.get("score", -999), reverse=True)
    _save_log(new_log)

    if verbose:
        logger.info(
            f"\n  ✓ Removed {duplicates_found} duplicate(s). "
            f"Log: {len(log)} → {len(new_log)} entries."
        )

    return duplicates_found


def print_leaderboard(top_n: int = 20):
    """Print the top-N experiments from the central log."""
    log = _load_log()
    if not log:
        print("  No experiments logged yet.")
        return

    print(f"\n{'═'*110}")
    print(f"  EXPERIMENT LEADERBOARD — Top {min(top_n, len(log))} of {len(log)} runs")
    print(f"{'═'*110}")
    print(f"  {'Rk':<3} {'Score':>7} {'IC':>7} {'DA':>6} {'R²':>7} {'Skill':>7} {'Params':>8} "
          f"{'Experiment':<25s} {'Key Config':>30s} {'Time':>12s}")
    print(f"  {'─'*108}")

    for i, e in enumerate(log[:top_n]):
        m = e.get("metrics", {})
        meta = e.get("meta", {})
        mc = e.get("model_config", {})
        tc = e.get("training_config", {})

        # Build compact config string highlighting key values
        try:
            lr = tc.get('learning_rate', 0)
            do = mc.get('dropout', 0)
            wd = tc.get('weight_decay', 0)
            dm = mc.get('d_model', '?')
            key_cfg = f"lr={lr:.0e} do={do:.2f} wd={wd:.3f} d={dm}"
        except (TypeError, ValueError):
            key_cfg = ""

        params_str = f"{meta.get('n_params', 0):,}" if meta.get('n_params') else "?"
        time_str = e.get("timestamp", "")[-8:]  # HH:MM:SS

        print(f"  {i+1:<3d} {e.get('score',0):>+.4f} "
              f"{m.get('ic_mean',0):>+.4f} {m.get('da_mean',0.5):>5.1%} "
              f"{m.get('r2_mean',0):>+.4f} {m.get('skill_mean',0):>+.4f} "
              f"{params_str:>8s} "
              f"{e.get('experiment','?'):<25s} "
              f"{key_cfg:>30s} "
              f"{time_str:>12s}")

    print(f"{'═'*110}")
    print(f"  Log file: {LOG_PATH}")


def get_log() -> list:
    """Return the full log as a list of dicts."""
    return _load_log()


def show_entry_detail(rank: int):
    """
    Pretty-print the full details of an experiment at a given leaderboard rank.

    Args:
        rank: 1-based rank in the leaderboard (1 = best score).
    """
    log = _load_log()
    if not log:
        print("  No experiments logged yet.")
        return

    if rank < 1 or rank > len(log):
        print(f"  Invalid rank {rank}. Log has {len(log)} entries (1–{len(log)}).")
        return

    e = log[rank - 1]
    m = e.get("metrics", {})
    mc = e.get("model_config", {})
    tc = e.get("training_config", {})
    cf = e.get("cli_flags", {})
    meta = e.get("meta", {})
    folds = e.get("folds", [])

    print(f"\n{'═'*70}")
    print(f"  EXPERIMENT DETAIL — Rank #{rank} of {len(log)}")
    print(f"{'═'*70}")

    # Header
    print(f"\n  Experiment:  {e.get('experiment', '?')}")
    print(f"  Timestamp:   {e.get('timestamp', '?')}")
    print(f"  Score:       {e.get('score', 0):+.6f}")
    if e.get("notes"):
        print(f"  Notes:       {e['notes']}")

    # Aggregate Metrics
    print(f"\n  {'─'*40}")
    print(f"  AGGREGATE METRICS")
    print(f"  {'─'*40}")
    print(f"  IC:          {m.get('ic_mean', 0):+.6f}")
    print(f"  R²:          {m.get('r2_mean', 0):+.6f}")
    print(f"  DA:          {m.get('da_mean', 0.5):.4f}  ({m.get('da_mean', 0.5):.1%})")
    print(f"  Skill:       {m.get('skill_mean', 0):+.6f}")
    print(f"  MAE:         {m.get('mae_mean', 0):.4f}")
    print(f"  RMSE:        {m.get('rmse_mean', 0):.4f}")

    # Per-Fold Breakdown
    if folds:
        print(f"\n  {'─'*40}")
        print(f"  PER-FOLD BREAKDOWN")
        print(f"  {'─'*40}")
        print(f"  {'Fold':>5s} {'IC':>8s} {'R²':>8s} {'DA':>8s} {'Skill':>8s} {'MAE':>8s} {'RMSE':>8s} {'N':>6s}")
        for f in sorted(folds, key=lambda x: x.get("fold", 0)):
            print(f"  {f.get('fold',0):>5d} {f.get('ic',0):>+8.4f} {f.get('r2',0):>+8.4f} "
                  f"{f.get('da',0):>8.4f} {f.get('skill',0):>+8.4f} "
                  f"{f.get('mae',0):>8.4f} {f.get('rmse',0):>8.4f} {f.get('n_samples',0):>6d}")

    # Model Config
    print(f"\n  {'─'*40}")
    print(f"  MODEL CONFIG")
    print(f"  {'─'*40}")
    for k in sorted(mc.keys()):
        v = mc[k]
        if isinstance(v, float):
            print(f"  {k:<30s} {v:.6g}")
        else:
            print(f"  {k:<30s} {v}")

    # Training Config
    print(f"\n  {'─'*40}")
    print(f"  TRAINING CONFIG")
    print(f"  {'─'*40}")
    for k in sorted(tc.keys()):
        v = tc[k]
        if isinstance(v, float):
            print(f"  {k:<30s} {v:.6g}")
        else:
            print(f"  {k:<30s} {v}")

    # CLI Flags
    if cf:
        print(f"\n  {'─'*40}")
        print(f"  CLI FLAGS")
        print(f"  {'─'*40}")
        for k in sorted(cf.keys()):
            print(f"  {k:<30s} {cf[k]}")

    # Meta
    print(f"\n  {'─'*40}")
    print(f"  META")
    print(f"  {'─'*40}")
    print(f"  {'n_params':<30s} {meta.get('n_params', '?'):,}" if meta.get('n_params') else f"  {'n_params':<30s} ?")
    print(f"  {'n_folds':<30s} {meta.get('n_folds', '?')}")
    print(f"  {'total_test_samples':<30s} {meta.get('total_test_samples', '?')}")
    t = meta.get('training_time_s')
    print(f"  {'training_time':<30s} {t:.1f}s ({t/60:.1f}min)" if t else f"  {'training_time':<30s} ?")
    print(f"  {'seed':<30s} {meta.get('seed', '?')}")

    # Reproducibility command
    print(f"\n  {'─'*40}")
    print(f"  REPRODUCE THIS RUN")
    print(f"  {'─'*40}")
    cmd_parts = ["python run_experiment.py --experiment full_mcat"]
    if cf.get("lookback"):
        cmd_parts.append(f"--lookback {cf['lookback']}")
    if cf.get("train_years"):
        cmd_parts.append(f"--train-years {cf['train_years']}")
    if cf.get("fold_mode") and cf["fold_mode"] != "focused":
        cmd_parts.append(f"--fold-mode {cf['fold_mode']}")
    if cf.get("no_denoise"):
        cmd_parts.append("--no-denoise")
    print(f"  {' '.join(cmd_parts)}")
    print(f"  # Then set in config.py:")
    if mc.get("dropout") is not None:
        print(f"  #   MODEL_CONFIG[\"dropout\"] = {mc['dropout']}")
    if mc.get("causal_conv_kernel") is not None:
        print(f"  #   MODEL_CONFIG[\"causal_conv_kernel\"] = {mc['causal_conv_kernel']}")
    if tc.get("learning_rate") is not None:
        print(f"  #   TRAINING_CONFIG[\"learning_rate\"] = {tc['learning_rate']}")
    if tc.get("weight_decay") is not None:
        print(f"  #   TRAINING_CONFIG[\"weight_decay\"] = {tc['weight_decay']}")
    if tc.get("batch_size") is not None:
        print(f"  #   TRAINING_CONFIG[\"batch_size\"] = {tc['batch_size']}")
    if tc.get("warmup_epochs") is not None:
        print(f"  #   TRAINING_CONFIG[\"warmup_epochs\"] = {tc['warmup_epochs']}")

    print(f"{'═'*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment log utilities")
    parser.add_argument("--dedup", action="store_true",
                        help="Deduplicate the experiment log (verbose report)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of entries to show in leaderboard")
    parser.add_argument("--show-rank", type=int, default=None,
                        help="Show full details for the entry at this rank (1-based)")
    args = parser.parse_args()

    if args.dedup:
        n = deduplicate_log(verbose=True)
        print(f"\n  Duplicates removed: {n}")

    if args.show_rank is not None:
        show_entry_detail(args.show_rank)
    else:
        print_leaderboard(top_n=args.top_n)
