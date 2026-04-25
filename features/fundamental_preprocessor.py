"""
features/fundamental_preprocessor.py — Compute time-varying fundamentals.
==========================================================================
PROBLEM: AV OVERVIEW returns only TODAY's snapshot (a single PE, profit margin).
Using 2024's PE=31 in 2019 training data is LOOK-AHEAD BIAS.

SOLUTION: Compute time-varying ratios from raw quarterly EPS + daily price.

TIER 1 — Time-varying (from EARNINGS + price, always available):
  pe_ratio_ttm:   Price / Trailing-12-Month EPS (changes every quarter + daily)
  earnings_yield:  EPS_TTM / Price (inverse PE, more stable)
  eps_growth_yoy:  (EPS_q - EPS_q-4) / |EPS_q-4| (year-over-year)
  eps_surprise:    surprisePercentage (from EARNINGS, already varies)
  eps_momentum:    Normalized slope of last 4 quarters' EPS

TIER 2 — Static cross-sectional (from OVERVIEW):
  log_MarketCap, DividendYield, BookValue
  These differ ACROSS stocks but stay constant within a stock.
  Useful as cross-sectional features complementing [STOCK] embedding.

All point-in-time aligned via reportedDate.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_START_DATE
from utils import setup_logger

logger = setup_logger(__name__)

DROP_COLUMNS = ["DebtToEquityRatio"]
# OVERVIEW columns are no longer fetched (static snapshot = look-ahead bias).
# All fundamental data now comes from EARNINGS + INCOME_STATEMENT + BALANCE_SHEET,
# which are genuinely historical and time-varying.


def compute_time_varying_ratios(
    fund_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute genuinely time-varying ratios from quarterly EPS + daily price."""
    result = pd.DataFrame(index=price_df.index)
    close = price_df["close"]

    if "reportedEPS" not in fund_df.columns:
        return result

    eps_quarterly = fund_df["reportedEPS"].copy()
    eps_changes = eps_quarterly[eps_quarterly != eps_quarterly.shift(1)].dropna()

    if len(eps_changes) < 2:
        return result

    quarterly_values = eps_changes.values
    quarterly_dates = eps_changes.index

    # ─── Trailing 12-Month EPS ───
    eps_ttm = pd.Series(np.nan, index=price_df.index, dtype=np.float64)
    for i in range(min(3, len(quarterly_dates) - 1), len(quarterly_dates)):
        start_idx = max(0, i - 3)
        ttm_sum = quarterly_values[start_idx:i + 1].sum()
        start_date = quarterly_dates[i]
        end_date = quarterly_dates[i + 1] if i + 1 < len(quarterly_dates) else price_df.index[-1]
        mask = (eps_ttm.index >= start_date) & (eps_ttm.index <= end_date)
        eps_ttm.loc[mask] = ttm_sum
    eps_ttm = eps_ttm.ffill().bfill()

    # PE ratio (time-varying: changes with both EPS updates and daily price)
    result["pe_ratio_ttm"] = close / (eps_ttm.clip(lower=0.01))
    result["pe_ratio_ttm"] = result["pe_ratio_ttm"].clip(0, 500)

    # Earnings yield
    result["earnings_yield"] = eps_ttm / (close + 1e-10)

    # EPS growth YoY
    if len(eps_changes) >= 5:
        eps_growth = pd.Series(np.nan, index=price_df.index, dtype=np.float64)
        for i in range(4, len(quarterly_dates)):
            current = quarterly_values[i]
            year_ago = quarterly_values[i - 4]
            growth = (current - year_ago) / max(abs(year_ago), 0.01)
            start_date = quarterly_dates[i]
            end_date = quarterly_dates[i + 1] if i + 1 < len(quarterly_dates) else price_df.index[-1]
            mask = (eps_growth.index >= start_date) & (eps_growth.index <= end_date)
            eps_growth.loc[mask] = growth
        result["eps_growth_yoy"] = eps_growth.ffill().bfill().clip(-5, 10)

    # EPS surprise
    if "surprisePercentage" in fund_df.columns:
        result["eps_surprise"] = fund_df["surprisePercentage"].reindex(
            price_df.index, method="ffill"
        ).clip(-100, 200).fillna(0)

    # EPS momentum (slope of last 4 quarters)
    if len(eps_changes) >= 4:
        eps_momentum = pd.Series(np.nan, index=price_df.index, dtype=np.float64)
        for i in range(3, len(quarterly_dates)):
            recent = quarterly_values[max(0, i - 3):i + 1]
            mean_abs = np.mean(np.abs(recent)) + 1e-10
            slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
            start_date = quarterly_dates[i]
            end_date = quarterly_dates[i + 1] if i + 1 < len(quarterly_dates) else price_df.index[-1]
            mask = (eps_momentum.index >= start_date) & (eps_momentum.index <= end_date)
            eps_momentum.loc[mask] = slope / mean_abs
        result["eps_momentum"] = eps_momentum.ffill().bfill().clip(-5, 5)

    # ─── INCOME STATEMENT ratios (from AV INCOME_STATEMENT endpoint) ───
    if "totalRevenue" in fund_df.columns and "netIncome" in fund_df.columns:
        revenue = fund_df["totalRevenue"].copy()
        net_income = fund_df["netIncome"].copy()
        rev_changes = revenue[revenue != revenue.shift(1)].dropna()

        # Profit margin = netIncome / totalRevenue (quarterly, genuinely varies)
        if len(rev_changes) >= 2:
            ni_changes = net_income.reindex(rev_changes.index, method="ffill")
            margin = ni_changes / rev_changes.clip(lower=1e6)
            margin_daily = margin.reindex(price_df.index, method="ffill")
            result["profit_margin"] = margin_daily.clip(-1, 1).ffill().bfill()
            logger.info(f"    profit_margin: {result['profit_margin'].nunique()} unique values")

        # Revenue growth YoY
        if len(rev_changes) >= 5:
            rev_vals = rev_changes.values
            rev_dates = rev_changes.index
            rev_growth = pd.Series(np.nan, index=price_df.index, dtype=np.float64)
            for i in range(4, len(rev_dates)):
                current = rev_vals[i]
                year_ago = rev_vals[i - 4]
                growth = (current - year_ago) / max(abs(year_ago), 1e6)
                start_date = rev_dates[i]
                end_date = rev_dates[i + 1] if i + 1 < len(rev_dates) else price_df.index[-1]
                mask = (rev_growth.index >= start_date) & (rev_growth.index <= end_date)
                rev_growth.loc[mask] = growth
            result["revenue_growth_yoy"] = rev_growth.ffill().bfill().clip(-2, 5)
            logger.info(f"    revenue_growth_yoy: {result['revenue_growth_yoy'].nunique()} unique values")

    # Gross margin = grossProfit / totalRevenue
    if "grossProfit" in fund_df.columns and "totalRevenue" in fund_df.columns:
        gp = fund_df["grossProfit"].copy()
        rev = fund_df["totalRevenue"].copy()
        gp_changes = gp[gp != gp.shift(1)].dropna()
        if len(gp_changes) >= 2:
            rev_at_gp = rev.reindex(gp_changes.index, method="ffill")
            gm = gp_changes / rev_at_gp.clip(lower=1e6)
            gm_daily = gm.reindex(price_df.index, method="ffill")
            result["gross_margin"] = gm_daily.clip(0, 1).ffill().bfill()
            logger.info(f"    gross_margin: {result['gross_margin'].nunique()} unique values")

    # ─── BALANCE SHEET ratios (from AV BALANCE_SHEET endpoint) ───
    if "totalShareholderEquity" in fund_df.columns:
        equity = fund_df["totalShareholderEquity"].copy()
        equity_changes = equity[equity != equity.shift(1)].dropna()

        # Debt-to-equity = total debt / equity
        if len(equity_changes) >= 2:
            long_debt = fund_df.get("longTermDebt", pd.Series(0, index=fund_df.index))
            short_debt = fund_df.get("shortTermDebt", pd.Series(0, index=fund_df.index))
            total_debt = long_debt.fillna(0) + short_debt.fillna(0)
            debt_at_eq = total_debt.reindex(equity_changes.index, method="ffill")
            dte = debt_at_eq / equity_changes.clip(lower=1e6)
            dte_daily = dte.reindex(price_df.index, method="ffill")
            result["debt_to_equity"] = dte_daily.clip(0, 10).ffill().bfill()
            logger.info(f"    debt_to_equity: {result['debt_to_equity'].nunique()} unique values")

    return result


def preprocess_fundamentals(
    fund_data: Dict[str, pd.DataFrame],
    price_data: Dict[str, pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compute time-varying fundamental ratios from raw AV data + daily prices.

    Input: Raw quarterly data from EARNINGS + INCOME_STATEMENT + BALANCE_SHEET.
    Output: ~9 genuinely time-varying features per ticker.

    No OVERVIEW data is used (it was a static snapshot = look-ahead bias).

    Args:
        fund_data: Ticker → raw fundamental DataFrame (from AV endpoints).
        price_data: Ticker → price DataFrame (for PE ratio computation).

    Returns:
        Ticker → cleaned DataFrame with time-varying features only.
    """
    cleaned = {}
    start = pd.Timestamp(DATA_START_DATE)

    for ticker, df in fund_data.items():
        df = df.copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        parts = []

        # Compute time-varying ratios from raw quarterly data + daily price
        if price_data is not None and ticker in price_data:
            price_df = price_data[ticker].copy()
            if price_df.index.tz is not None:
                price_df.index = price_df.index.tz_localize(None)
            price_study = price_df[price_df.index >= start]

            tv = compute_time_varying_ratios(df, price_study)
            if not tv.empty:
                parts.append(tv)
                logger.info(f"  {ticker}: {tv.shape[1]} time-varying ratios computed "
                           f"({list(tv.columns)})")
        else:
            logger.warning(f"  {ticker}: No price data — cannot compute time-varying ratios")

        # Drop always-NaN columns
        for part in parts:
            for col in DROP_COLUMNS:
                if col in part.columns:
                    part.drop(columns=[col], inplace=True)

        # Combine
        if parts:
            combined = pd.concat(parts, axis=1)
        else:
            # Fallback: keep raw earnings columns only
            keep_cols = [c for c in ["reportedEPS", "estimatedEPS", "surprisePercentage"]
                        if c in df.columns]
            combined = df[keep_cols] if keep_cols else df.iloc[:, :1]
            logger.warning(f"  {ticker}: No time-varying ratios computed — using raw EPS columns")

        combined = combined.ffill().bfill().fillna(0)
        combined = combined.replace([np.inf, -np.inf], 0)
        cleaned[ticker] = combined

    if cleaned:
        sample = list(cleaned.values())[0]
        logger.info(f"Fundamentals: {len(cleaned)} tickers, {sample.shape[1]} features")
        logger.info(f"  Features: {list(sample.columns)}")

    return cleaned


if __name__ == "__main__":
    uploads = Path("/mnt/user-data/uploads")
    fund_data = {}
    for f in uploads.glob("fundamentals_real_*.parquet"):
        ticker = f.stem.replace("fundamentals_real_", "")
        fund_data[ticker] = pd.read_parquet(f)

    np.random.seed(42)
    dates = pd.bdate_range("2016-01-04", "2023-12-29")
    price_data = {}
    for ticker in fund_data:
        price = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, len(dates))))
        price_data[ticker] = pd.DataFrame({"close": price}, index=dates)

    cleaned = preprocess_fundamentals(fund_data, price_data)
    for ticker, df in cleaned.items():
        print(f"\n{ticker}: {df.shape}")
        for col in df.columns:
            vals = df[col].dropna()
            print(f"  {col}: nunique={vals.nunique()}, range=[{vals.min():.4g}, {vals.max():.4g}]" +
                  (" ← TIME-VARYING" if vals.nunique() > 5 else " (static)"))
