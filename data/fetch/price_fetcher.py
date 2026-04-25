"""
data/fetch/price_fetcher.py — Fetch historical OHLCV data from yfinance.
============================================================================
Downloads daily price data for all tickers in our universe, caches locally.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    TICKERS, RAW_DIR, DATA_FETCH_START, DATA_END_DATE
)
from data.provenance import provenance
from utils import setup_logger, timer

logger = setup_logger(__name__)


def fetch_single_ticker(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a single ticker from yfinance.

    Returns DataFrame with columns: open, high, low, close, volume, adj_close
    indexed by date. Returns None on failure.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            logger.warning(f"{ticker}: No data returned from yfinance")
            return None

        # ── Handle yfinance column formats ──
        # Newer yfinance (≥0.2.31) returns MultiIndex columns:
        #   ('Open', 'AAPL'), ('High', 'AAPL'), ...
        # Older versions return flat strings: 'Open', 'High', ...
        # We must check for MultiIndex FIRST before any string operations.
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten: take only the first level (metric name), drop ticker level
            df.columns = [c[0] for c in df.columns]

        # Now columns are guaranteed to be strings — safe to lowercase
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

        # De-duplicate columns (yfinance sometimes returns duplicate 'close'/'adj_close')
        df = df.loc[:, ~df.columns.duplicated()]

        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                logger.error(f"{ticker}: Missing column '{col}'")
                return None

        # Add adj_close if missing (use close as fallback)
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]

        # Sort by date and remove timezone info if present
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()

        # Drop rows with any NaN in OHLCV
        n_before = len(df)
        df = df.dropna(subset=required)
        if n_before - len(df) > 0:
            logger.info(f"  {ticker}: Dropped {n_before - len(df)} rows with NaN")

        logger.info(f"  {ticker}: {len(df)} trading days [{df.index[0].date()} → {df.index[-1].date()}]")
        return df[["open", "high", "low", "close", "volume", "adj_close"]]

    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return None
    except Exception as e:
        logger.error(f"{ticker}: Failed to fetch — {e}")
        return None


@timer
def fetch_all_prices(
    tickers: list = None,
    start: str = DATA_FETCH_START,
    end: str = DATA_END_DATE,
    cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for all tickers. Uses parquet cache.

    Returns:
        Dict mapping ticker → DataFrame with OHLCV data.
    """
    if tickers is None:
        tickers = TICKERS

    price_data = {}
    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] Fetching {ticker}...")

        cache_path = RAW_DIR / f"price_{ticker}.parquet"

        # Check cache
        if cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            logger.info(f"  {ticker}: Loaded from cache ({len(df)} days)")
            price_data[ticker] = df
            provenance.register(ticker, "price", "real",
                                f"yfinance OHLCV ({len(df)} days, cached)", n_features=39)
            continue

        # Fetch from yfinance
        df = fetch_single_ticker(ticker, start, end)
        if df is not None:
            if cache:
                df.to_parquet(cache_path)
            price_data[ticker] = df
            provenance.register(ticker, "price", "real",
                                f"yfinance OHLCV ({len(df)} days)", n_features=39)
        else:
            logger.warning(f"  {ticker}: SKIPPED — no data")
            provenance.register(ticker, "price", "unavailable",
                                "yfinance fetch failed")

    logger.info(f"Price data ready for {len(price_data)}/{len(tickers)} tickers")
    return price_data


def fetch_index_data(
    symbols: dict = None,
    start: str = DATA_FETCH_START,
    end: str = DATA_END_DATE,
    cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch index and ETF data for macro features.

    Args:
        symbols: Dict of name → yfinance symbol (e.g., {"VIX": "^VIX"})
    """
    if symbols is None:
        from config import MACRO_YFINANCE, SECTOR_ETFS
        symbols = {**MACRO_YFINANCE}
        for etf in SECTOR_ETFS:
            symbols[etf] = etf
        symbols["SPY"] = "SPY"

    data = {}
    for name, symbol in symbols.items():
        cache_path = RAW_DIR / f"index_{name}.parquet"

        if cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            data[name] = df
            continue

        df = fetch_single_ticker(symbol, start, end)
        if df is not None:
            if cache:
                df.to_parquet(cache_path)
            data[name] = df
        else:
            logger.warning(f"  {name} ({symbol}): SKIPPED")

    logger.info(f"Index/macro data ready: {len(data)} series")
    return data


if __name__ == "__main__":
    prices = fetch_all_prices()
    for ticker, df in prices.items():
        print(f"{ticker}: {len(df)} days, cols={list(df.columns)}")
