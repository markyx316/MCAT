"""
data/fetch/macro_fetcher.py — Macroeconomic data pipeline.
============================================================
Fetches market-wide indicators from yfinance and FRED.

Daily from yfinance: VIX, 10Y yield, 13W bill, USD index, 11 sector ETFs
Monthly from FRED (forward-filled): Fed funds rate, yield spread, unemployment

Output: Single DataFrame indexed by trading date with ~20 macro features.
"""

import pandas as pd
import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    MACRO_YFINANCE, SECTOR_ETFS, MACRO_FRED,
    RAW_DIR, PROCESSED_DIR, DATA_FETCH_START, DATA_END_DATE, TICKERS,
)
from data.provenance import provenance
from utils import setup_logger, timer

logger = setup_logger(__name__)


def _fetch_yf_series(symbol: str, start: str, end: str) -> Optional[pd.Series]:
    """Fetch a single yfinance series (close price)."""
    try:
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None

        # Handle MultiIndex columns from newer yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        # Get close price — try multiple possible column names
        close = None
        for col_name in ["Close", "close", "Adj Close", "adj_close"]:
            if col_name in df.columns:
                close = df[col_name]
                break

        if close is None:
            # Fallback: take first numeric column
            close = df.iloc[:, 0]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close.index = pd.to_datetime(close.index)
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        return close
    except Exception as e:
        logger.warning(f"  Failed to fetch {symbol}: {e}")
        return None


FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _fetch_fred_csv(
    series_id: str,
    start: str,
    end: str,
) -> Optional[pd.Series]:
    """
    Download a FRED series directly via public CSV URL.
    NO API key required — FRED provides this as a free public service.

    URL format:
      https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS&cosd=2016-01-01&coed=2024-01-01
    """
    import io

    try:
        import requests
        url = (
            f"{FRED_CSV_BASE}"
            f"?id={series_id}"
            f"&cosd={start}"
            f"&coed={end}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        # Parse CSV without assuming column names — detect them dynamically
        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty:
            return None

        # Normalize column names to lowercase for robust matching
        df.columns = [c.strip().lower() for c in df.columns]

        # Find the date column (could be 'DATE', 'date', 'Date', etc.)
        date_col = None
        for col in df.columns:
            if 'date' in col:
                date_col = col
                break

        if date_col is None:
            # If no 'date' column, assume first column is date
            date_col = df.columns[0]
            logger.info(f"  FRED {series_id}: No 'date' column found, using '{date_col}'")

        # Parse dates and set as index
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col)

        if df.empty:
            return None

        # The value column is whatever remains (should be the series_id or 'value')
        series = df.iloc[:, 0]
        # FRED uses '.' for missing values — convert to numeric
        series = pd.to_numeric(series, errors="coerce")
        series.index = series.index.tz_localize(None) if series.index.tz else series.index
        series = series.dropna()

        return series

    except ImportError:
        logger.warning("  requests not installed — cannot download FRED CSV")
        return None
    except Exception as e:
        logger.warning(f"  FRED CSV download failed for {series_id}: {e}")
        return None


@timer
def fetch_macro_data(
    start: str = DATA_FETCH_START,
    end: str = DATA_END_DATE,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch all macro features and combine into a single DataFrame.

    Returns:
        DataFrame indexed by trading date with ~20 macro feature columns.
    """
    cache_path = PROCESSED_DIR / "macro_features.parquet"
    if cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        logger.info(f"Macro data loaded from cache: {df.shape}")

        # Register provenance even on cache hit
        fred_cols = [c for c in MACRO_FRED.keys() if c in df.columns]
        n_total = df.shape[1]
        if fred_cols:
            source_detail = (
                f"yfinance + FRED CSV ({len(fred_cols)} series: {', '.join(fred_cols)}) "
                f"= {n_total} total features (cached)"
            )
            source_type = "real"
        else:
            source_detail = f"yfinance only ({n_total} features, cached)"
            source_type = "partial"

        for ticker in TICKERS:
            provenance.register(ticker, "macro", source_type, source_detail, n_features=n_total)

        return df

    # Trading date index from SPY
    spy = _fetch_yf_series("SPY", start, end)
    if spy is None:
        logger.error("Cannot fetch SPY — creating business day range")
        trading_dates = pd.bdate_range(start, end)
    else:
        trading_dates = spy.index

    macro = pd.DataFrame(index=trading_dates)

    # ─── yfinance market indicators ───
    for name, symbol in MACRO_YFINANCE.items():
        series = _fetch_yf_series(symbol, start, end)
        if series is not None:
            macro[name] = series.reindex(trading_dates, method="ffill")
            logger.info(f"  {name} ({symbol}): {series.notna().sum()} values")
        else:
            logger.warning(f"  {name} ({symbol}): FAILED — filling with 0")
            macro[name] = 0.0

    # ─── Sector ETF relative returns ───
    if spy is not None:
        spy_ret = spy.pct_change()
        for etf in SECTOR_ETFS:
            series = _fetch_yf_series(etf, start, end)
            if series is not None:
                etf_ret = series.pct_change()
                # Relative return = ETF return - SPY return (sector-specific signal)
                macro[f"{etf}_rel_ret"] = (etf_ret - spy_ret).reindex(trading_dates, method="ffill")
                logger.info(f"  {etf} relative return: computed")
            else:
                macro[f"{etf}_rel_ret"] = 0.0
                logger.warning(f"  {etf}: FAILED — filling with 0")

    # ─── FRED data (direct CSV download — NO API key required) ───
    fred_source_details = []
    for series_id, name in MACRO_FRED.items():
        fred_series = _fetch_fred_csv(series_id, start, end)
        if fred_series is not None:
            # Forward-fill monthly/weekly data to daily trading dates
            macro[series_id] = fred_series.reindex(trading_dates, method="ffill")
            n_vals = fred_series.notna().sum()
            logger.info(f"  FRED {series_id} ({name}): {n_vals} values via CSV download")
            fred_source_details.append(series_id)
        else:
            logger.warning(f"  FRED {series_id} ({name}): FAILED — filling with NaN")
            macro[series_id] = np.nan

    # ─── Cleanup ───
    macro = macro.ffill().bfill()
    macro = macro.replace([np.inf, -np.inf], 0.0)
    macro = macro.fillna(0.0)

    if cache:
        macro.to_parquet(cache_path)

    logger.info(f"Macro data: {macro.shape[1]} features, {len(macro)} days")

    # Register provenance for all tickers (macro is shared)
    n_yf_cols = len(MACRO_YFINANCE) + len(SECTOR_ETFS)
    n_fred_cols = len(fred_source_details)
    n_total_cols = macro.shape[1]

    if n_fred_cols > 0:
        source_detail = (
            f"yfinance ({n_yf_cols} series) + "
            f"FRED CSV download ({n_fred_cols} series: {', '.join(fred_source_details)}) "
            f"= {n_total_cols} total features"
        )
        source_type = "real"
    else:
        source_detail = (
            f"yfinance only ({n_yf_cols} features), "
            f"FRED download failed — missing {list(MACRO_FRED.keys())}"
        )
        source_type = "partial"

    for ticker in TICKERS:
        provenance.register(
            ticker, "macro", source_type, source_detail, n_features=n_total_cols,
        )

    return macro


if __name__ == "__main__":
    macro = fetch_macro_data()
    print(f"\nShape: {macro.shape}")
    print(f"Columns: {list(macro.columns)}")
    print(f"\nFirst 5 rows:\n{macro.head()}")
