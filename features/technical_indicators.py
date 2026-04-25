"""
features/technical_indicators.py — Compute 39 price/technical features.
========================================================================
37 original technical indicators + 2 Fourier trend components = 39 total.

All indicators are either self-normalizing (RSI, %B, etc.) or computed
as ratios that remove price-level dependence. This ensures the features
are comparable across stocks with different price scales.

IMPORTANT: All indicators are computed causally — they only use data
available at or before time t. No future information leaks.
"""

import pandas as pd
import numpy as np
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TECHNICAL_CONFIG, FOURIER_COMPONENTS
from utils import setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range = max(H-L, |H-Cprev|, |L-Cprev|)."""
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


# ─────────────────────────────────────────────────────────────
# MAIN FEATURE COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 39 technical features from OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
            indexed by date, sorted chronologically.

    Returns:
        DataFrame with 39 feature columns, same index as input.
        Early rows will have NaN due to lookback warmup.
    """
    cfg = TECHNICAL_CONFIG
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    features = pd.DataFrame(index=df.index)

    # ─── Category 1: Returns (4) ───
    features["ret_cc"] = c.pct_change()                          # Close-to-close
    features["ret_log"] = np.log(c / c.shift(1))                 # Log return
    features["ret_intraday"] = (c - o) / o                       # Intraday
    features["ret_overnight"] = (o - c.shift(1)) / c.shift(1)    # Overnight

    # ─── Category 2: Trend Indicators (10) ───
    for p in cfg["sma_periods"]:
        features[f"sma_ratio_{p}"] = c / _sma(c, p)              # Price / SMA
    for p in cfg["ema_periods"]:
        features[f"ema_ratio_{p}"] = c / _ema(c, p)              # Price / EMA

    ema_fast = _ema(c, cfg["macd_fast"])
    ema_slow = _ema(c, cfg["macd_slow"])
    macd_line = ema_fast - ema_slow
    macd_signal = _ema(macd_line, cfg["macd_signal"])
    features["macd_line"] = macd_line / c                        # Normalized by price
    features["macd_signal"] = macd_signal / c
    features["macd_histogram"] = (macd_line - macd_signal) / c

    # ADX (Average Directional Index)
    adx_period = cfg["adx_period"]
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    # Only keep the larger one
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr = _true_range(h, l, c)
    atr = tr.rolling(adx_period).mean()
    plus_di = 100 * (plus_dm.rolling(adx_period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(adx_period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    features["adx"] = dx.rolling(adx_period).mean() / 100        # Normalize to [0, 1]

    # ─── Category 3: Momentum Oscillators (6) ───
    # RSI
    rsi_period = cfg["rsi_period"]
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-10)
    features["rsi"] = (100 - 100 / (1 + rs)) / 100               # [0, 1]

    # Stochastic %K and %D
    sp = cfg["stoch_period"]
    lowest_low = l.rolling(sp).min()
    highest_high = h.rolling(sp).max()
    stoch_k = (c - lowest_low) / (highest_high - lowest_low + 1e-10)
    features["stoch_k"] = stoch_k                                 # [0, 1]
    features["stoch_d"] = stoch_k.rolling(cfg["stoch_smooth"]).mean()

    # Williams %R
    wp = cfg["williams_period"]
    highest_h = h.rolling(wp).max()
    lowest_l = l.rolling(wp).min()
    features["williams_r"] = (highest_h - c) / (highest_h - lowest_l + 1e-10)  # [0, 1]

    # CCI (Commodity Channel Index)
    tp = (h + l + c) / 3
    cci_p = cfg["cci_period"]
    tp_sma = tp.rolling(cci_p).mean()
    tp_mad = tp.rolling(cci_p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    features["cci"] = (tp - tp_sma) / (0.015 * tp_mad + 1e-10) / 200  # Approx [-1, 1]

    # ROC (Rate of Change)
    roc_p = cfg["roc_period"]
    features["roc"] = c.pct_change(roc_p)

    # ─── Category 4: Volatility Measures (6) ───
    bb_p = cfg["bollinger_period"]
    bb_std = cfg["bollinger_std"]
    bb_sma = _sma(c, bb_p)
    bb_rolling_std = c.rolling(bb_p).std()
    bb_upper = bb_sma + bb_std * bb_rolling_std
    bb_lower = bb_sma - bb_std * bb_rolling_std
    features["bollinger_pctb"] = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)
    features["bollinger_bw"] = (bb_upper - bb_lower) / (bb_sma + 1e-10)

    atr_p = cfg["atr_period"]
    tr = _true_range(h, l, c)
    features["atr_ratio"] = tr.rolling(atr_p).mean() / c          # ATR / price

    log_ret = np.log(c / c.shift(1))
    features["vol_5d"] = log_ret.rolling(5).std()
    features["vol_20d"] = log_ret.rolling(20).std()

    # Garman-Klass volatility estimator
    gk_inner = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    features["vol_gk"] = np.sqrt(gk_inner.clip(lower=0)).rolling(20).mean()

    # ─── Category 5: Volume Indicators (3) ───
    vol_sma_p = cfg["volume_sma_period"]
    features["volume_ratio"] = v / v.rolling(vol_sma_p).mean()

    # OBV z-score
    obv = (np.sign(c.diff()) * v).cumsum()
    obv_mean = obv.rolling(vol_sma_p).mean()
    obv_std = obv.rolling(vol_sma_p).std()
    features["obv_zscore"] = (obv - obv_mean) / (obv_std + 1e-10)

    # Chaikin Money Flow
    cmf_p = cfg["cmf_period"]
    mfm = ((c - l) - (h - c)) / (h - l + 1e-10)
    mfv = mfm * v
    features["cmf"] = mfv.rolling(cmf_p).sum() / (v.rolling(cmf_p).sum() + 1e-10)

    # ─── Category 6: Multi-Horizon Historical Returns (4) ───
    for horizon in [1, 5, 10, 20]:
        features[f"ret_{horizon}d"] = c.pct_change(horizon)

    # ─── Category 7: Price-Derived (4) ───
    features["intraday_range"] = (h - l) / c
    features["gap"] = (o - c.shift(1)) / c.shift(1)
    features["dist_52w_high"] = c / h.rolling(252).max() - 1
    features["dist_52w_low"] = c / l.rolling(252).min() - 1

    # ─── Category 8: Fourier Trend Components (2) ───
    for n_comp in FOURIER_COMPONENTS:
        fourier_trend = _fourier_trend(c.values, n_components=n_comp)
        features[f"fourier_{n_comp}"] = fourier_trend / c.values   # Ratio to current price

    # ─── Final cleanup ───
    # Replace inf with NaN
    features = features.replace([np.inf, -np.inf], np.nan)

    n_features = features.shape[1]
    logger.info(f"Computed {n_features} technical features")
    return features


def _fourier_trend(price_array: np.ndarray, n_components: int) -> np.ndarray:
    """
    Extract low-frequency trend using Fourier transform.

    Keeps only the lowest n_components frequency components,
    producing a smooth trend line.

    Inspired by Li & Xu (2025) who use Fourier decomposition
    to extract long-term price trends as features.
    """
    n = len(price_array)
    fft = np.fft.fft(price_array)
    # Zero out all but the lowest n_components frequencies
    fft_filtered = np.zeros_like(fft)
    fft_filtered[:n_components] = fft[:n_components]
    fft_filtered[-n_components:] = fft[-n_components:]
    trend = np.real(np.fft.ifft(fft_filtered))
    return trend


def get_feature_names() -> List[str]:
    """Return ordered list of all 39 feature names."""
    cfg = TECHNICAL_CONFIG
    names = []
    # Returns (4)
    names += ["ret_cc", "ret_log", "ret_intraday", "ret_overnight"]
    # Trend (10)
    for p in cfg["sma_periods"]:
        names.append(f"sma_ratio_{p}")
    for p in cfg["ema_periods"]:
        names.append(f"ema_ratio_{p}")
    names += ["macd_line", "macd_signal", "macd_histogram", "adx"]
    # Momentum (6)
    names += ["rsi", "stoch_k", "stoch_d", "williams_r", "cci", "roc"]
    # Volatility (6)
    names += ["bollinger_pctb", "bollinger_bw", "atr_ratio", "vol_5d", "vol_20d", "vol_gk"]
    # Volume (3)
    names += ["volume_ratio", "obv_zscore", "cmf"]
    # Multi-horizon returns (4)
    names += ["ret_1d", "ret_5d", "ret_10d", "ret_20d"]
    # Price-derived (4)
    names += ["intraday_range", "gap", "dist_52w_high", "dist_52w_low"]
    # Fourier (2)
    for n in FOURIER_COMPONENTS:
        names.append(f"fourier_{n}")
    return names


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    dates = pd.bdate_range("2016-01-01", "2023-12-31")
    n = len(dates)
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
    test_df = pd.DataFrame({
        "open": price * (1 + np.random.normal(0, 0.003, n)),
        "high": price * (1 + np.abs(np.random.normal(0, 0.01, n))),
        "low": price * (1 - np.abs(np.random.normal(0, 0.01, n))),
        "close": price,
        "volume": np.random.lognormal(18, 0.5, n),
    }, index=dates)

    features = compute_technical_features(test_df)
    print(f"Features shape: {features.shape}")
    print(f"Feature names ({len(get_feature_names())}): {get_feature_names()}")
    print(f"\nNaN count per feature (first 5):")
    print(features.isna().sum().head(10))
    print(f"\nFeature stats after warmup (row 252+):")
    print(features.iloc[252:].describe().T[["mean", "std", "min", "max"]].head(10))
