"""
features/macro_preprocessor.py — Clean macro data issues.
============================================================
Fixes discovered in data inspection:

1. XLC_rel_ret: XLC ETF launched June 2018; before that, forward-fill
   created 621 days of stale constant values → zero-fill pre-launch
2. Other sector ETFs may have similar launch-date issues → check all
3. FEDFUNDS constant for 123 days → NOT a bug (Fed held rates at 0%)
"""

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import setup_logger

logger = setup_logger(__name__)

# Known ETF launch dates (approximate)
ETF_LAUNCH_DATES = {
    "XLC_rel_ret": "2018-06-18",  # Communication Services: launched June 2018
    "XLRE_rel_ret": "2015-10-08",  # Real Estate: launched Oct 2015
}


def preprocess_macro(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known issues in macro data.

    1. Zero-fill sector ETF relative returns before their launch dates
    2. Report any remaining stale-data issues

    Args:
        macro_df: Raw macro DataFrame from macro_fetcher.

    Returns:
        Cleaned macro DataFrame.
    """
    df = macro_df.copy()

    for col, launch_date_str in ETF_LAUNCH_DATES.items():
        if col not in df.columns:
            continue
        launch_date = pd.Timestamp(launch_date_str)
        pre_launch_mask = df.index < launch_date
        n_pre = pre_launch_mask.sum()
        if n_pre > 0:
            old_val = df.loc[pre_launch_mask, col].iloc[0] if n_pre > 0 else 0
            df.loc[pre_launch_mask, col] = 0.0
            logger.info(f"  {col}: Zeroed {n_pre} pre-launch days (before {launch_date_str}, was {old_val:.6f})")

    return df


if __name__ == "__main__":
    path = Path("/mnt/user-data/uploads/macro_features.parquet")
    if path.exists():
        df = pd.read_parquet(path)
        cleaned = preprocess_macro(df)
        # Verify XLC fix
        xlc = cleaned["XLC_rel_ret"]
        pre_2018 = xlc[xlc.index < "2018-06-18"]
        print(f"XLC pre-launch: all zero = {(pre_2018 == 0).all()}, count = {len(pre_2018)}")
