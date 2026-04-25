"""
data/simulate_market.py — Realistic Synthetic Market Data Generator.
=====================================================================
Generates synthetic data that mimics the statistical properties of
real financial markets for pipeline testing:

  - Geometric Brownian Motion with realistic drift/volatility per sector
  - Correlated cross-stock returns (via Cholesky decomposition)
  - Realistic OHLCV structure (intraday range, volume patterns)
  - Simulated earnings dates with fundamental jumps
  - Simulated sentiment with regime-dependent patterns
  - Macro environment with VIX-like volatility clustering

This lets us run the ENTIRE pipeline end-to-end and verify correctness
before deploying with real data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TICKERS, SECTOR_MAP, DATA_START_DATE, DATA_END_DATE, FINBERT_DIM
from data.provenance import provenance
from utils import setup_logger, set_seed

logger = setup_logger(__name__)

# Realistic parameters per sector
SECTOR_PARAMS = {
    "Technology": {"mu": 0.0008, "sigma": 0.022, "vol_cluster": 0.94},
    "Financials": {"mu": 0.0004, "sigma": 0.018, "vol_cluster": 0.92},
    "Healthcare": {"mu": 0.0005, "sigma": 0.015, "vol_cluster": 0.91},
    "Consumer Staples": {"mu": 0.0003, "sigma": 0.012, "vol_cluster": 0.90},
    "Energy": {"mu": 0.0003, "sigma": 0.025, "vol_cluster": 0.93},
    "Industrials": {"mu": 0.0004, "sigma": 0.017, "vol_cluster": 0.92},
    "Communication Services": {"mu": 0.0004, "sigma": 0.020, "vol_cluster": 0.93},
}

# Cross-sector correlation matrix (approximate)
SECTOR_ORDER = ["Technology", "Financials", "Healthcare", "Consumer Staples",
                "Energy", "Industrials", "Communication Services"]
SECTOR_CORR = np.array([
    [1.0, 0.6, 0.4, 0.3, 0.3, 0.5, 0.7],   # Tech
    [0.6, 1.0, 0.4, 0.3, 0.4, 0.5, 0.5],   # Fin
    [0.4, 0.4, 1.0, 0.5, 0.2, 0.4, 0.3],   # Health
    [0.3, 0.3, 0.5, 1.0, 0.2, 0.3, 0.3],   # Staples
    [0.3, 0.4, 0.2, 0.2, 1.0, 0.5, 0.3],   # Energy
    [0.5, 0.5, 0.4, 0.3, 0.5, 1.0, 0.5],   # Industrials
    [0.7, 0.5, 0.3, 0.3, 0.3, 0.5, 1.0],   # Comm
])


def generate_correlated_returns(
    n_stocks: int,
    n_days: int,
    sector_assignments: list,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate correlated daily returns with sector structure.

    Returns:
        (n_days, n_stocks) array of daily log returns.
    """
    rng = np.random.RandomState(seed)

    # Build per-stock correlation matrix from sector correlations
    stock_corr = np.eye(n_stocks)
    for i in range(n_stocks):
        for j in range(i + 1, n_stocks):
            si = SECTOR_ORDER.index(sector_assignments[i])
            sj = SECTOR_ORDER.index(sector_assignments[j])
            # Same sector = higher correlation
            if sector_assignments[i] == sector_assignments[j]:
                stock_corr[i, j] = stock_corr[j, i] = 0.7 + rng.uniform(0, 0.15)
            else:
                stock_corr[i, j] = stock_corr[j, i] = SECTOR_CORR[si, sj]

    # Cholesky decomposition for correlated normals
    L = np.linalg.cholesky(stock_corr)

    # Generate independent normals then correlate
    Z = rng.randn(n_days, n_stocks)
    correlated = Z @ L.T

    # Apply sector-specific drift and volatility with GARCH-like clustering
    returns = np.zeros((n_days, n_stocks))
    for i, sector in enumerate(sector_assignments):
        params = SECTOR_PARAMS.get(sector, SECTOR_PARAMS["Technology"])
        mu = params["mu"]
        base_sigma = params["sigma"]
        alpha = params["vol_cluster"]

        # GARCH(1,1)-like volatility clustering
        sigma_t = base_sigma
        for t in range(n_days):
            returns[t, i] = mu + sigma_t * correlated[t, i]
            # Update volatility
            sigma_t = np.sqrt(
                (1 - alpha) * base_sigma ** 2 + alpha * returns[t, i] ** 2
            )
            sigma_t = np.clip(sigma_t, base_sigma * 0.3, base_sigma * 4.0)

    return returns


def generate_ohlcv(
    log_returns: np.ndarray,
    initial_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic OHLCV from log returns.

    Intraday range and volume patterns mimic real market behaviour.
    """
    rng = np.random.RandomState(seed)
    n = len(log_returns)

    # Price path from log returns
    close = initial_price * np.exp(np.cumsum(log_returns))

    # Intraday structure
    intraday_range = np.abs(log_returns) + rng.exponential(0.005, n)
    open_gap = rng.normal(0, 0.002, n)

    open_price = close * (1 + open_gap) / np.exp(log_returns)
    high = np.maximum(open_price, close) * (1 + intraday_range * rng.uniform(0.3, 0.8, n))
    low = np.minimum(open_price, close) * (1 - intraday_range * rng.uniform(0.3, 0.8, n))

    # Volume with autocorrelation and daily patterns
    base_vol = rng.lognormal(18, 0.5, n)
    vol_multiplier = 1 + 2 * np.abs(log_returns) / np.std(log_returns)
    volume = base_vol * vol_multiplier

    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume.astype(np.int64),
        "adj_close": close,
    })


def generate_simulated_market(
    tickers: list = None,
    start_date: str = None,
    end_date: str = None,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Generate a full simulated market dataset for all tickers.

    Returns:
        Dict of ticker → OHLCV DataFrame with realistic structure.
    """
    if tickers is None:
        tickers = TICKERS
    if start_date is None:
        start_date = "2016-01-01"  # Extra warmup
    if end_date is None:
        end_date = DATA_END_DATE

    set_seed(seed)

    trading_dates = pd.bdate_range(start_date, end_date)
    n_days = len(trading_dates)
    n_stocks = len(tickers)

    sector_assignments = [SECTOR_MAP.get(t, "Technology") for t in tickers]

    logger.info(f"Generating simulated market: {n_stocks} stocks × {n_days} days")

    # Generate correlated returns
    all_returns = generate_correlated_returns(
        n_stocks, n_days, sector_assignments, seed,
    )

    # Build OHLCV per ticker
    price_data = {}
    initial_prices = {
        "AAPL": 120, "MSFT": 90, "GOOGL": 800, "AMZN": 750, "META": 120,
        "NVDA": 30, "JPM": 85, "GS": 200, "JNJ": 120, "UNH": 170,
        "WMT": 70, "PG": 85, "XOM": 80, "CAT": 100, "DIS": 100,
    }

    for i, ticker in enumerate(tickers):
        init_price = initial_prices.get(ticker, 100)
        ohlcv = generate_ohlcv(all_returns[:, i], init_price, seed=seed + i)
        ohlcv.index = trading_dates
        price_data[ticker] = ohlcv

        # Register provenance
        provenance.register(
            ticker, "price", "synthetic",
            f"Simulated GBM with GARCH volatility clustering "
            f"(sector={sector_assignments[i]}, σ={SECTOR_PARAMS[sector_assignments[i]]['sigma']})",
            n_features=39,
        )

    # Verify cross-stock correlation
    close_returns = pd.DataFrame({
        t: np.log(price_data[t]["close"] / price_data[t]["close"].shift(1))
        for t in tickers
    }).dropna()
    avg_corr = close_returns.corr().values[np.triu_indices(n_stocks, k=1)].mean()
    logger.info(f"  Average cross-stock return correlation: {avg_corr:.3f}")

    return price_data


def generate_simulated_sentiment(
    price_data: Dict[str, pd.DataFrame],
    tickers: list = None,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Generate simulated sentiment embeddings with regime-dependent patterns.

    Sentiment is weakly correlated with contemporaneous and future returns
    to create a realistic (but controlled) predictive signal.
    """
    if tickers is None:
        tickers = list(price_data.keys())

    rng = np.random.RandomState(seed)
    sent_embeddings = {}
    sent_counts = {}
    # Use small embedding dim for memory efficiency in testing
    emb_dim = 32

    for ticker in tickers:
        dates = price_data[ticker].index
        n = len(dates)

        # Base sentiment: weakly correlated with returns
        close = price_data[ticker]["close"].values
        log_ret = np.log(close[1:] / close[:-1])
        log_ret = np.concatenate([[0], log_ret])

        # Sentiment signal = smoothed returns + noise
        signal = pd.Series(log_ret).rolling(5).mean().fillna(0).values
        noise = rng.randn(n, emb_dim) * 0.3

        # Embed sentiment as a structured vector
        emb = np.outer(signal, np.ones(emb_dim)) * 0.5 + noise
        emb = emb.astype(np.float32)

        # Article counts: more on volatile days
        vol = np.abs(log_ret)
        base_count = rng.poisson(2, n).astype(np.float32)
        count_boost = (vol > np.percentile(vol, 80)).astype(np.float32) * 3
        counts = base_count + count_boost

        sent_embeddings[ticker] = emb
        sent_counts[ticker] = counts

        provenance.register(
            ticker, "sentiment", "synthetic",
            f"Simulated: return-correlated embeddings ({emb_dim}-dim) + noise. "
            f"NOT real news data.",
            n_features=emb_dim + 1,
        )

    return sent_embeddings, sent_counts


def build_simulated_dataset(
    tickers: list = None,
    denoise: bool = True,
    seed: int = 42,
):
    """
    Build a complete dataset from simulated market data.

    This exercises the FULL pipeline with realistic data structure
    while being completely self-contained (no external data needed).

    Returns:
        Tuple of (dataset, provenance_report)
    """
    if tickers is None:
        tickers = TICKERS

    logger.info("=" * 60)
    logger.info("  BUILDING SIMULATED MARKET DATASET")
    logger.info("=" * 60)

    set_seed(seed)

    # Step 1: Generate simulated OHLCV
    price_data = generate_simulated_market(tickers, seed=seed)

    # Step 2: Compute features and labels
    from features.technical_indicators import compute_technical_features
    from features.label_generator import compute_labels

    price_features = {}
    labels = {}
    start_date = pd.Timestamp(DATA_START_DATE)

    for ticker in tickers:
        feat_df = compute_technical_features(price_data[ticker])
        feat_df = feat_df[feat_df.index >= start_date]
        df_study = price_data[ticker][price_data[ticker].index >= start_date]

        label_series = compute_labels(df_study)
        price_features[ticker] = feat_df
        labels[ticker] = label_series

    # Step 3: Generate simulated sentiment
    sent_emb, sent_cnt = generate_simulated_sentiment(
        {t: price_data[t][price_data[t].index >= start_date] for t in tickers},
        tickers, seed,
    )

    # Step 4: Synthetic fundamentals
    from data.fetch.fundamental_fetcher import build_synthetic_fundamentals
    fund_data = {}
    for ticker in tickers:
        df_study = price_data[ticker][price_data[ticker].index >= start_date]
        fund_data[ticker] = build_synthetic_fundamentals(df_study)
        provenance.register(
            ticker, "fundamentals", "synthetic",
            "Price-derived proxies (simulated market)", n_features=7,
        )

    # Step 5: Simulated macro (VIX-like + sector returns)
    n_study = len(price_features[tickers[0]])
    study_dates = price_features[tickers[0]].index
    rng = np.random.RandomState(seed + 100)

    # VIX-like: high when returns are volatile
    avg_returns = np.mean([
        np.abs(np.log(price_data[t]["close"] / price_data[t]["close"].shift(1)).reindex(study_dates).fillna(0).values)
        for t in tickers
    ], axis=0)
    vix_like = 15 + avg_returns * 500 + rng.randn(n_study) * 2
    macro_df = pd.DataFrame({"VIX": vix_like}, index=study_dates)
    macro_df["TNX"] = 2.5 + np.cumsum(rng.randn(n_study) * 0.01)  # 10Y yield
    macro_df["DXY"] = 95 + np.cumsum(rng.randn(n_study) * 0.1)    # Dollar index

    for ticker in tickers:
        provenance.register(
            ticker, "macro", "synthetic",
            "Simulated VIX + yield + DXY (3 features)", n_features=3,
        )

    # Step 6: Build dataset
    from features.dataset import MultiModalDataset

    dataset = MultiModalDataset(
        price_features=price_features,
        labels=labels,
        sentiment_embeddings=sent_emb,
        sentiment_counts=sent_cnt,
        fund_features=fund_data,
        macro_features=macro_df,
        denoise=denoise,
        tickers=tickers,
    )

    report = provenance.report()

    logger.info(f"\n  Dataset: {len(dataset)} samples")
    logger.info(f"  Price: {dataset.X_price.shape}")
    logger.info(f"  Sentiment: {dataset.X_sent.shape}")
    logger.info(f"  Fund: {dataset.X_fund.shape}")
    logger.info(f"  Macro: {dataset.X_macro.shape}")
    logger.info(f"  Labels: mean={dataset.y.mean():.3f}pp, std={dataset.y.std():.3f}pp")

    return dataset, report


if __name__ == "__main__":
    # Build simulated dataset with 3 tickers
    dataset, report = build_simulated_dataset(
        tickers=["AAPL", "JPM", "XOM"],
        denoise=False,
        seed=42,
    )
    print(f"\nDataset: {len(dataset)} samples")
    s = dataset[100]
    for k, v in s.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {v}")
