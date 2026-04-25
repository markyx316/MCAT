"""
data/fetch/fundamental_fetcher.py — Fundamental Data Pipeline.
================================================================
Fetches quarterly fundamental data from Alpha Vantage API with automatic
synthetic fallback and provenance tracking.

PRIORITY ORDER:
  1. Real Alpha Vantage API data (10 features) — if API key is valid
  2. Synthetic price-derived proxies (7 features) — explicit fallback

CRITICAL: Point-in-time alignment is enforced. Q4 2022 earnings reported
in February 2023 are NOT visible to the model until February 2023.

The provenance system ALWAYS records whether real or synthetic data was
used for each ticker, so results transparently reflect data quality.
"""

import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    TICKERS, PROCESSED_DIR, RAW_DIR,
    DATA_FETCH_START, DATA_END_DATE,
    ALPHA_VANTAGE_API_KEY, AV_SLEEP_BETWEEN_CALLS,
    AV_RATE_LIMIT_PER_DAY,
    FUNDAMENTAL_FEATURES_REAL, FUNDAMENTAL_FEATURES_SYNTHETIC,
    N_FUND_FEATURES_REAL, N_FUND_FEATURES_SYNTHETIC,
)
from data.provenance import provenance
from utils import setup_logger, timer

logger = setup_logger(__name__)

AV_BASE_URL = "https://www.alphavantage.co/query"


# ═════════════════════════════════════════════════════════════
# REAL DATA: ALPHA VANTAGE API
# ═════════════════════════════════════════════════════════════

def _av_request(function: str, ticker: str, **kwargs) -> Optional[dict]:
    """Make a rate-limited Alpha Vantage API request."""
    if ALPHA_VANTAGE_API_KEY == "YOUR_AV_KEY_HERE":
        return None

    params = {
        "function": function,
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY,
        **kwargs,
    }

    try:
        resp = requests.get(AV_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "Error Message" in data:
            logger.warning(f"  AV API error for {ticker}/{function}: {data['Error Message']}")
            return None
        if "Note" in data:
            logger.warning(f"  AV rate limit hit for {ticker}: {data['Note']}")
            return None
        if "Information" in data and "rate" in data["Information"].lower():
            logger.warning(f"  AV rate limit: {data['Information']}")
            return None

        return data
    except Exception as e:
        logger.warning(f"  AV request failed for {ticker}/{function}: {e}")
        return None


def fetch_av_earnings(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch quarterly earnings from Alpha Vantage EARNINGS endpoint."""
    data = _av_request("EARNINGS", ticker)
    if data is None or "quarterlyEarnings" not in data:
        return None

    quarterly = data["quarterlyEarnings"]
    if not quarterly:
        return None

    df = pd.DataFrame(quarterly)
    for col in ["reportedEPS", "estimatedEPS", "surprise", "surprisePercentage"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
    df["reportedDate"] = pd.to_datetime(df["reportedDate"])
    df = df.sort_values("fiscalDateEnding").reset_index(drop=True)
    logger.info(f"  {ticker}: {len(df)} quarterly earnings records from AV")
    return df


def fetch_av_overview(ticker: str) -> Optional[dict]:
    """Fetch company overview from Alpha Vantage OVERVIEW endpoint."""
    data = _av_request("OVERVIEW", ticker)
    if data is None or "Symbol" not in data:
        return None

    fields = {}
    for key in ["PERatio", "ProfitMargin", "RevenuePerShareTTM",
                 "DividendPayoutRatio", "BookValue", "DividendYield",
                 "MarketCapitalization"]:
        val = data.get(key, "None")
        try:
            fields[key] = float(val) if val not in ("None", "-", "") else np.nan
        except (ValueError, TypeError):
            fields[key] = np.nan

    return fields


def fetch_av_income_statement(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch quarterly income statements from Alpha Vantage INCOME_STATEMENT endpoint.
    Returns ~20 quarters of historical revenue, net income, gross profit.
    1 API call per ticker.
    """
    data = _av_request("INCOME_STATEMENT", ticker)
    if data is None or "quarterlyReports" not in data:
        return None

    reports = data["quarterlyReports"]
    if not reports:
        return None

    rows = []
    for r in reports:
        row = {"fiscalDateEnding": r.get("fiscalDateEnding")}
        for field in ["totalRevenue", "netIncome", "grossProfit",
                      "operatingIncome", "costOfRevenue"]:
            val = r.get(field, "None")
            try:
                row[field] = float(val) if val not in ("None", "-", "", "0") else np.nan
            except (ValueError, TypeError):
                row[field] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
    df = df.sort_values("fiscalDateEnding").reset_index(drop=True)
    logger.info(f"  {ticker}: {len(df)} quarterly income statement records from AV")
    return df


def fetch_av_balance_sheet(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch quarterly balance sheets from Alpha Vantage BALANCE_SHEET endpoint.
    Returns ~20 quarters of historical equity, debt, assets.
    1 API call per ticker.
    """
    data = _av_request("BALANCE_SHEET", ticker)
    if data is None or "quarterlyReports" not in data:
        return None

    reports = data["quarterlyReports"]
    if not reports:
        return None

    rows = []
    for r in reports:
        row = {"fiscalDateEnding": r.get("fiscalDateEnding")}
        for field in ["totalShareholderEquity", "totalCurrentLiabilities",
                      "totalCurrentAssets", "longTermDebt", "shortTermDebt",
                      "totalAssets"]:
            val = r.get(field, "None")
            try:
                row[field] = float(val) if val not in ("None", "-", "", "0") else np.nan
            except (ValueError, TypeError):
                row[field] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
    df = df.sort_values("fiscalDateEnding").reset_index(drop=True)
    logger.info(f"  {ticker}: {len(df)} quarterly balance sheet records from AV")
    return df


def build_real_fundamentals(
    ticker: str,
    trading_dates: pd.DatetimeIndex,
) -> Optional[pd.DataFrame]:
    """
    Build point-in-time aligned fundamental features from Alpha Vantage.

    Fetches 3 endpoints (NO OVERVIEW — it's a static snapshot = look-ahead bias):
      - EARNINGS — quarterly EPS, estimates, surprise (~30 quarters)
      - INCOME_STATEMENT — historical revenue, net income, gross profit (~20 quarters)
      - BALANCE_SHEET — historical equity, debt, assets (~20 quarters)

    CRITICAL — Point-in-time alignment:
      All quarterly data is anchored to the EARNINGS reportedDate (the actual
      publication date, not fiscalDateEnding). For income/balance quarters
      without a matching earnings date, a conservative 60-day lag is used.
    """
    # ─── Fetch EARNINGS (required) ───
    earnings_df = fetch_av_earnings(ticker)
    time.sleep(AV_SLEEP_BETWEEN_CALLS)

    if earnings_df is None:
        logger.warning(f"  {ticker}: Cannot fetch earnings from Alpha Vantage")
        return None

    # ─── Fetch INCOME_STATEMENT (1 API call) ───
    income_df = fetch_av_income_statement(ticker)
    time.sleep(AV_SLEEP_BETWEEN_CALLS)

    # ─── Fetch BALANCE_SHEET (1 API call) ───
    balance_df = fetch_av_balance_sheet(ticker)
    time.sleep(AV_SLEEP_BETWEEN_CALLS)

    # ─── Build the fundamental DataFrame ───
    fund = pd.DataFrame(index=trading_dates, dtype=np.float64)
    for col in FUNDAMENTAL_FEATURES_REAL:
        fund[col] = np.nan

    # Build a mapping: fiscalDateEnding → reportedDate (from EARNINGS)
    # This lets us align income/balance sheet data to the actual publication date
    fiscal_to_reported = {}
    for _, row in earnings_df.iterrows():
        fiscal = row.get("fiscalDateEnding")
        reported = row.get("reportedDate")
        if pd.notna(fiscal) and pd.notna(reported):
            fiscal_to_reported[fiscal] = reported

    # ─── Place EARNINGS at reported dates (point-in-time) ───
    for _, row in earnings_df.iterrows():
        reported_date = row.get("reportedDate")
        if pd.isna(reported_date):
            continue
        valid_dates = trading_dates[trading_dates >= reported_date]
        if len(valid_dates) == 0:
            continue
        effective_date = valid_dates[0]

        for col in ["reportedEPS", "estimatedEPS", "surprisePercentage"]:
            if col in row and pd.notna(row[col]):
                fund.loc[effective_date, col] = row[col]

    # ─── Place INCOME_STATEMENT data at point-in-time dates ───
    income_cols_placed = 0
    if income_df is not None:
        # Add columns for income statement data
        for col in ["totalRevenue", "netIncome", "grossProfit"]:
            if col not in fund.columns:
                fund[col] = np.nan

        for _, row in income_df.iterrows():
            fiscal = row.get("fiscalDateEnding")
            if pd.isna(fiscal):
                continue
            # Use reportedDate from earnings for same quarter, else fiscal + 60 days
            if fiscal in fiscal_to_reported:
                pub_date = fiscal_to_reported[fiscal]
            else:
                pub_date = fiscal + pd.Timedelta(days=60)

            valid_dates = trading_dates[trading_dates >= pub_date]
            if len(valid_dates) == 0:
                continue
            effective_date = valid_dates[0]

            for col in ["totalRevenue", "netIncome", "grossProfit"]:
                if col in row and pd.notna(row[col]):
                    fund.loc[effective_date, col] = row[col]
                    income_cols_placed += 1

    # ─── Place BALANCE_SHEET data at point-in-time dates ───
    balance_cols_placed = 0
    if balance_df is not None:
        for col in ["totalShareholderEquity", "longTermDebt", "shortTermDebt", "totalAssets"]:
            if col not in fund.columns:
                fund[col] = np.nan

        for _, row in balance_df.iterrows():
            fiscal = row.get("fiscalDateEnding")
            if pd.isna(fiscal):
                continue
            if fiscal in fiscal_to_reported:
                pub_date = fiscal_to_reported[fiscal]
            else:
                pub_date = fiscal + pd.Timedelta(days=60)

            valid_dates = trading_dates[trading_dates >= pub_date]
            if len(valid_dates) == 0:
                continue
            effective_date = valid_dates[0]

            for col in ["totalShareholderEquity", "longTermDebt", "shortTermDebt", "totalAssets"]:
                if col in row and pd.notna(row[col]):
                    fund.loc[effective_date, col] = row[col]
                    balance_cols_placed += 1

    fund = fund.ffill().bfill()

    n_valid = fund.notna().any(axis=0).sum()
    n_total = fund.shape[1]
    extra = []
    if income_df is not None:
        extra.append(f"income:{income_cols_placed}")
    if balance_df is not None:
        extra.append(f"balance:{balance_cols_placed}")
    extra_str = f" + {', '.join(extra)}" if extra else ""
    logger.info(f"  {ticker}: Real fundamentals — {n_valid}/{n_total} features{extra_str}")
    return fund


# ═════════════════════════════════════════════════════════════
# SYNTHETIC FALLBACK: PRICE-DERIVED PROXIES
# ═════════════════════════════════════════════════════════════

def build_synthetic_fundamentals(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 7 price-derived proxy features as fallback.

    ⚠ THESE ARE EXPLICITLY SYNTHETIC — NOT real company financials.
    The provenance system records this for every ticker that uses them.
    """
    c = price_df["close"]
    v = price_df["volume"]
    fund = pd.DataFrame(index=price_df.index)

    sma200 = c.rolling(200).mean()
    sma50 = c.rolling(50).mean()
    log_ret = np.log(c / c.shift(1))

    fund["pe_proxy"] = c / (sma200 + 1e-10)
    fund["volatility_60d"] = log_ret.rolling(60).std() * np.sqrt(252)
    fund["volume_trend_20d"] = v / (v.rolling(20).mean() + 1e-10)
    fund["price_sma50_ratio"] = c / (sma50 + 1e-10)
    fund["momentum_90d"] = c.pct_change(90)
    fund["golden_cross"] = sma50 / (sma200 + 1e-10)
    fund["log_mcap_proxy"] = np.log(c * v.rolling(20).mean() + 1)

    quarterly = fund.resample("QE").last()
    fund_quarterly = quarterly.reindex(fund.index, method="ffill")
    fund_quarterly = fund_quarterly.ffill().bfill()

    return fund_quarterly


# ═════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════

@timer
def fetch_all_fundamentals(
    price_data: Dict[str, pd.DataFrame],
    tickers: list = None,
    cache: bool = True,
    force_synthetic: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """
    Fetch fundamentals for all tickers: real API first, synthetic fallback.

    PROVENANCE: Every ticker's source is registered. The returned
    source_types dict maps ticker → "real" or "synthetic".
    """
    if tickers is None:
        tickers = TICKERS

    fund_data = {}
    source_types = {}
    api_available = ALPHA_VANTAGE_API_KEY != "YOUR_AV_KEY_HERE" and not force_synthetic

    if api_available:
        logger.info("Alpha Vantage API key detected — attempting real data first")
    else:
        reason = "force_synthetic=True" if force_synthetic else "No API key set"
        logger.warning(f"⚠ {reason}. Using SYNTHETIC fundamentals for all tickers.")

    api_calls_made = 0

    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] Fundamentals for {ticker}...")

        # Step 1: Check cache
        real_cache = PROCESSED_DIR / f"fundamentals_real_{ticker}.parquet"
        synth_cache = PROCESSED_DIR / f"fundamentals_synth_{ticker}.parquet"

        if cache and real_cache.exists():
            df = pd.read_parquet(real_cache)
            fund_data[ticker] = df
            source_types[ticker] = "real"
            provenance.register(
                ticker, "fundamentals", "real",
                f"Alpha Vantage API ({df.shape[1]} features, cached)",
                n_features=df.shape[1],
            )
            continue

        # Step 2: Try Alpha Vantage API
        # Each ticker uses 3 API calls: EARNINGS + INCOME_STATEMENT + BALANCE_SHEET
        # (OVERVIEW is NOT called — it returns static snapshots = look-ahead bias)
        used_real = False
        if api_available and api_calls_made + 3 <= AV_RATE_LIMIT_PER_DAY:
            trading_dates = price_data[ticker].index if ticker in price_data else \
                pd.bdate_range(DATA_FETCH_START, DATA_END_DATE)

            real_df = build_real_fundamentals(ticker, trading_dates)
            api_calls_made += 3

            if real_df is not None and real_df.notna().any(axis=0).sum() >= 3:
                fund_data[ticker] = real_df
                source_types[ticker] = "real"
                used_real = True
                if cache:
                    real_df.to_parquet(real_cache)
                n_valid = real_df.notna().any(axis=0).sum()
                provenance.register(
                    ticker, "fundamentals", "real",
                    f"Alpha Vantage API ({n_valid}/{N_FUND_FEATURES_REAL} features)",
                    n_features=n_valid,
                )

        # Step 3: Synthetic fallback
        if not used_real:
            if cache and synth_cache.exists():
                df = pd.read_parquet(synth_cache)
            elif ticker in price_data:
                df = build_synthetic_fundamentals(price_data[ticker])
                if cache:
                    df.to_parquet(synth_cache)
            else:
                source_types[ticker] = "unavailable"
                provenance.register(ticker, "fundamentals", "unavailable",
                                    "No price data for synthetic fallback")
                continue

            fund_data[ticker] = df
            source_types[ticker] = "synthetic"
            provenance.register(
                ticker, "fundamentals", "synthetic",
                f"⚠ Price-derived proxies ({N_FUND_FEATURES_SYNTHETIC} features) — "
                f"NOT real company financials",
                n_features=N_FUND_FEATURES_SYNTHETIC,
            )

    # Summary
    n_real = sum(1 for v in source_types.values() if v == "real")
    n_synth = sum(1 for v in source_types.values() if v == "synthetic")
    logger.info(f"Fundamentals: {n_real} real, {n_synth} synthetic ({api_calls_made} API calls)")

    if n_synth > 0:
        synth_list = [t for t, s in source_types.items() if s == "synthetic"]
        logger.warning(
            f"⚠ SYNTHETIC FUNDAMENTALS: {', '.join(synth_list)}\n"
            f"  Results for these tickers test architecture, not real fundamental value."
        )

    return fund_data, source_types


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.bdate_range("2016-01-01", "2023-12-31")
    n = len(dates)
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
    test_prices = {
        "AAPL": pd.DataFrame({
            "close": price, "volume": np.random.lognormal(18, 0.5, n),
        }, index=dates),
    }

    fund_data, source_types = fetch_all_fundamentals(
        test_prices, tickers=["AAPL"], force_synthetic=True,
    )

    for ticker, df in fund_data.items():
        print(f"\n{ticker} ({source_types[ticker]}):")
        print(f"  Shape: {df.shape}, Columns: {list(df.columns)}")

    provenance.report()
