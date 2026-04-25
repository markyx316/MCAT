"""
training/walk_forward.py — Walk-forward cross-validation fold generator.
=========================================================================
Two modes:
  - Full (14 folds): 6-month val, 3-month test, rolling 3-month steps.
    Comprehensive but slow. Use for final paper validation.
  - Focused (3 folds): 3-month val, 6-month test, covering Jul 2022 → Dec 2023.
    2× statistical power per fold. Default for experimentation.

Anti-leakage mechanisms:
  1. Embargo gaps (5 days) prevent label-horizon overlap
  2. Fresh model initialization per fold (caller's responsibility)
  3. Per-window normalization (in dataset.py)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_START_DATE, DATA_END_DATE,
    INITIAL_TRAIN_YEARS, VALIDATION_MONTHS, TEST_MONTHS,
    WALK_FORWARD_STEP_MONTHS, EMBARGO_DAYS,
    FOCUSED_FOLDS,
)
from utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class WalkForwardFold:
    """A single walk-forward fold with date boundaries."""
    fold_num: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self):
        return (
            f"Fold {self.fold_num:2d}: "
            f"Train [{self.train_start.strftime('%Y-%m')} → {self.train_end.strftime('%Y-%m')}] | "
            f"Val [{self.val_start.strftime('%Y-%m')} → {self.val_end.strftime('%Y-%m')}] | "
            f"Test [{self.test_start.strftime('%Y-%m')} → {self.test_end.strftime('%Y-%m')}]"
        )


def generate_walk_forward_folds(
    data_start: str = DATA_START_DATE,
    data_end: str = DATA_END_DATE,
    train_years: int = INITIAL_TRAIN_YEARS,
    val_months: int = VALIDATION_MONTHS,
    test_months: int = TEST_MONTHS,
    step_months: int = WALK_FORWARD_STEP_MONTHS,
    embargo_days: int = EMBARGO_DAYS,
) -> List[WalkForwardFold]:
    """
    Generate walk-forward cross-validation folds.

    The training window expands from `train_years` in fold 0 to
    nearly the full dataset in the final fold. Validation and test
    windows are fixed-size and slide forward by `step_months` each fold.

    5-day embargo gaps are inserted between train/val and val/test
    to prevent label-horizon overlap (our 3-day labels mean the last
    training sample's label period overlaps the first validation
    sample's features without a gap).

    Returns:
        List of WalkForwardFold objects.
    """
    start = pd.Timestamp(data_start)
    end = pd.Timestamp(data_end)
    embargo = pd.DateOffset(days=embargo_days)

    folds = []
    fold_num = 0

    # First fold: train window ends after initial_train_years
    train_end = start + pd.DateOffset(years=train_years)

    while True:
        val_start = train_end + embargo
        val_end = val_start + pd.DateOffset(months=val_months)

        test_start = val_end + embargo
        test_end = test_start + pd.DateOffset(months=test_months)

        # Stop if test period extends beyond data
        if test_start >= end:
            break

        # Clip test end to data end
        if test_end > end:
            test_end = end

        fold = WalkForwardFold(
            fold_num=fold_num,
            train_start=start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        )
        folds.append(fold)
        fold_num += 1

        # Advance training window (expanding)
        train_end = train_end + pd.DateOffset(months=step_months)

        if test_end >= end:
            break

    logger.info(f"Generated {len(folds)} walk-forward folds:")
    for f in folds:
        logger.info(f"  {f}")

    return folds


def generate_focused_folds(
    data_start: str = DATA_START_DATE,
    embargo_days: int = EMBARGO_DAYS,
    fold_specs: list = None,
    train_years: float = None,
) -> List[WalkForwardFold]:
    """
    Generate the focused 3-fold validation for efficient experimentation.

    Uses wider test windows (6 months) and narrower val windows (3 months)
    compared to the full 14-fold setup. This gives:
      - 2× more statistical power per fold (1,920 vs 960 test samples)
      - 50% more total test samples than a 4-fold narrow setup
      - Full coverage from Jul 2022 → Dec 2023 with non-overlapping tests

    Args:
        data_start: Start of training data (default: 2017-01-01).
        embargo_days: Gap between splits (default 5 days).
        fold_specs: List of (train_end, val_end, test_end) date strings.
                    Defaults to FOCUSED_FOLDS from config.py.
        train_years: If specified, use a SLIDING window of this many years
                     instead of expanding from data_start. Each fold trains
                     on [train_end - train_years, train_end].
                     None = expanding window from data_start (default).

    Returns:
        List of 3 WalkForwardFold objects.
    """
    if fold_specs is None:
        fold_specs = FOCUSED_FOLDS

    default_start = pd.Timestamp(data_start)
    embargo = pd.DateOffset(days=embargo_days)

    folds = []
    for fold_num, (train_end_str, val_end_str, test_end_str) in enumerate(fold_specs):
        train_end = pd.Timestamp(train_end_str)

        # Sliding window: train_start = train_end - train_years
        # Expanding window: train_start = data_start (fixed)
        if train_years is not None:
            # Convert fractional years to DateOffset
            years_int = int(train_years)
            months_frac = int((train_years - years_int) * 12)
            train_start = train_end - pd.DateOffset(years=years_int, months=months_frac)
            # Clamp to data_start (can't train before data exists)
            if train_start < default_start:
                train_start = default_start
        else:
            train_start = default_start

        val_start = train_end + embargo
        val_end = pd.Timestamp(val_end_str)
        test_start = val_end + embargo
        test_end = pd.Timestamp(test_end_str)

        fold = WalkForwardFold(
            fold_num=fold_num,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        )
        folds.append(fold)

    mode = f"sliding {train_years}yr" if train_years else "expanding"
    logger.info(f"Generated {len(folds)} focused folds ({mode}, 3-mo val, 6-mo test):")
    for f in folds:
        logger.info(f"  {f}")

    # Verify non-overlapping test windows
    for i in range(len(folds) - 1):
        if folds[i].test_end > folds[i + 1].test_start:
            logger.warning(f"  ⚠ Test overlap between fold {i} and {i+1}!")

    return folds


if __name__ == "__main__":
    print("=" * 60)
    print("  Full 14-fold walk-forward validation:")
    print("=" * 60)
    folds = generate_walk_forward_folds()
    print(f"\n{len(folds)} folds generated")
    print(f"Test periods span: {folds[0].test_start.date()} → {folds[-1].test_end.date()}")

    print(f"\n{'=' * 60}")
    print("  Focused 3-fold validation:")
    print("=" * 60)
    folds = generate_focused_folds()
    print(f"\n{len(folds)} folds generated")
    print(f"Test periods span: {folds[0].test_start.date()} → {folds[-1].test_end.date()}")
