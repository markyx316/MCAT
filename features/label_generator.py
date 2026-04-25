"""
features/label_generator.py — Construct regression targets.
=============================================================
Target: 3-day forward return in percentage points.

y_{i,t} = (P_{i,t+3} - P_{i,t}) / P_{i,t} * 100

This strips out the persistence component (today's price ≈ yesterday's price)
and forces the model to predict genuine innovations only.
"""

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FORWARD_HORIZON, RETURN_SCALE
from utils import setup_logger

logger = setup_logger(__name__)


def compute_forward_returns(
    close: pd.Series,
    horizon: int = FORWARD_HORIZON,
    scale: float = RETURN_SCALE,
) -> pd.Series:
    """
    Compute forward N-day returns scaled to percentage points.

    Args:
        close: Series of closing prices, indexed by date.
        horizon: Number of trading days to look forward.
        scale: Multiplier (100 → percentage points).

    Returns:
        Series of forward returns. Last `horizon` rows are NaN.

    Example:
        If close[t] = 100 and close[t+3] = 102.5,
        forward_return[t] = (102.5 - 100) / 100 * 100 = 2.5 (percentage points)
    """
    fwd_ret = (close.shift(-horizon) / close - 1) * scale
    return fwd_ret


def compute_labels(
    price_df: pd.DataFrame,
    horizon: int = FORWARD_HORIZON,
    scale: float = RETURN_SCALE,
) -> pd.Series:
    """
    Compute labels from a price DataFrame.

    Args:
        price_df: DataFrame with 'close' column.
        horizon: Forward return horizon in trading days.
        scale: Multiplier (100 → pp).

    Returns:
        Series of forward returns in percentage points.
    """
    labels = compute_forward_returns(price_df["close"], horizon, scale)
    labels.name = f"fwd_ret_{horizon}d_pp"

    # Report statistics
    valid = labels.dropna()
    logger.info(
        f"Labels: {len(valid)} valid / {len(labels)} total | "
        f"mean={valid.mean():.3f}pp, std={valid.std():.3f}pp, "
        f"min={valid.min():.1f}pp, max={valid.max():.1f}pp | "
        f"pct_positive={100*(valid>0).mean():.1f}%"
    )
    return labels


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    dates = pd.bdate_range("2017-01-01", "2023-12-31")
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, len(dates))))
    test_df = pd.DataFrame({"close": price}, index=dates)
    labels = compute_labels(test_df)
    print(f"\nLabel distribution:\n{labels.describe()}")
