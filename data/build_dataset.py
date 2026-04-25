"""
data/build_dataset.py — Full Data Pipeline Orchestrator.
==========================================================
Single entry point that chains all data fetchers, feature engineering,
and dataset construction into one call:

    dataset, provenance_report = build_full_dataset()

Pipeline:
  1. Fetch price data (yfinance) → compute 39 technical features + labels
  2. Fetch sentiment data (FNSPID → FinBERT) or synthetic fallback
  3. Fetch fundamental data (Alpha Vantage API) or synthetic fallback
  4. Fetch macro data (yfinance + FRED)
  5. Build MultiModalDataset with denoising + normalization
  6. Generate and validate provenance report

Every step logs clearly whether REAL or SYNTHETIC data was used.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    TICKERS, DATA_START_DATE, DATA_END_DATE,
    DATA_FETCH_START, PROCESSED_DIR,
)
from data.provenance import provenance
from utils import setup_logger, timer, set_seed

logger = setup_logger(__name__)


@timer
def build_full_dataset(
    tickers: list = None,
    fnspid_path: str = None,
    use_finbert: bool = True,
    force_synthetic_fundamentals: bool = False,
    denoise: bool = True,
    cache: bool = True,
    lookback: int = None,
):
    """
    Build the complete multi-modal dataset from scratch.

    This is the SINGLE ENTRY POINT for the entire data pipeline.
    It calls all fetchers, computes features, constructs labels,
    and assembles the PyTorch dataset — with full provenance tracking.

    Args:
        tickers: Stock tickers to process (default: all 15).
        fnspid_path: Path to FNSPID CSV. None = synthetic sentiment.
        use_finbert: If True, encode articles with FinBERT. Else VADER.
        force_synthetic_fundamentals: Skip Alpha Vantage, use proxies.
        denoise: Apply wavelet denoising to price features.
        cache: Use parquet caching for intermediate results.
        lookback: Lookback window in trading days (default: LOOKBACK_WINDOW from config).

    Returns:
        Tuple of:
          - dataset: MultiModalDataset ready for walk-forward training
          - provenance_report: String of the full provenance report
    """
    if tickers is None:
        tickers = TICKERS

    logger.info("=" * 60)
    logger.info("  BUILDING FULL MULTI-MODAL DATASET")
    logger.info("=" * 60)
    logger.info(f"  Tickers: {len(tickers)} | Denoise: {denoise}")
    logger.info(f"  FNSPID: {'provided' if fnspid_path else 'NOT provided (synthetic)'}")
    logger.info(f"  Fundamentals: {'force synthetic' if force_synthetic_fundamentals else 'try Alpha Vantage'}")
    logger.info("")

    # ═══════════════════════════════════════════════════════════
    # STEP 1: PRICE DATA + TECHNICAL FEATURES + LABELS
    # ═══════════════════════════════════════════════════════════
    logger.info("─── Step 1/4: Price Data + Features ───")

    from data.fetch.price_fetcher import fetch_all_prices
    from features.technical_indicators import compute_technical_features
    from features.label_generator import compute_labels

    price_data = fetch_all_prices(tickers=tickers, cache=cache)

    # Compute features and labels for each ticker
    price_features = {}
    labels = {}
    for ticker in tickers:
        if ticker not in price_data:
            logger.warning(f"  {ticker}: No price data — SKIPPED")
            continue

        df = price_data[ticker]

        # Technical features (39)
        feat_df = compute_technical_features(df)

        # Filter to study period (after indicator warmup)
        start_mask = feat_df.index >= pd.Timestamp(DATA_START_DATE)
        feat_df = feat_df[start_mask]
        df_study = df[start_mask]

        # Labels (3-day forward return × 100)
        label_series = compute_labels(df_study)

        price_features[ticker] = feat_df
        labels[ticker] = label_series

    logger.info(f"  Price features: {len(price_features)} tickers processed")

    # ═══════════════════════════════════════════════════════════
    # STEP 2: SENTIMENT DATA
    # ═══════════════════════════════════════════════════════════
    logger.info("\n─── Step 2/4: Sentiment Data ───")

    from data.fetch.sentiment_fetcher import fetch_all_sentiment

    # Build trading date index per ticker for alignment
    trading_dates_per_ticker = {
        t: price_features[t].index for t in price_features
    }

    sent_embeddings, sent_counts, sent_source_types = fetch_all_sentiment(
        fnspid_path=fnspid_path,
        trading_dates_per_ticker=trading_dates_per_ticker,
        price_data=price_data,
        tickers=list(price_features.keys()),
        use_finbert=use_finbert,
        cache=cache,
    )

    logger.info(f"  Sentiment: {len(sent_embeddings)} tickers processed")

    # ═══════════════════════════════════════════════════════════
    # STEP 3: FUNDAMENTAL DATA
    # ═══════════════════════════════════════════════════════════
    logger.info("\n─── Step 3/4: Fundamental Data ───")

    from data.fetch.fundamental_fetcher import fetch_all_fundamentals

    fund_data, fund_source_types = fetch_all_fundamentals(
        price_data=price_data,
        tickers=list(price_features.keys()),
        cache=cache,
        force_synthetic=force_synthetic_fundamentals,
    )

    # Preprocess: compute time-varying ratios from EPS + price, drop static OVERVIEW fields
    from features.fundamental_preprocessor import preprocess_fundamentals
    fund_data = preprocess_fundamentals(fund_data, price_data=price_data)

    logger.info(f"  Fundamentals: {len(fund_data)} tickers processed")

    # ═══════════════════════════════════════════════════════════
    # STEP 4: MACRO DATA
    # ═══════════════════════════════════════════════════════════
    logger.info("\n─── Step 4/4: Macro Data ───")

    from data.fetch.macro_fetcher import fetch_macro_data

    macro_df = fetch_macro_data(cache=cache)

    # Preprocess: fix XLC pre-launch stale data
    from features.macro_preprocessor import preprocess_macro
    macro_df = preprocess_macro(macro_df)

    # Filter to study period
    macro_df = macro_df[macro_df.index >= pd.Timestamp(DATA_START_DATE)]

    logger.info(f"  Macro: {macro_df.shape[1]} features, {len(macro_df)} days")

    # ═══════════════════════════════════════════════════════════
    # STEP 5: ASSEMBLE DATASET
    # ═══════════════════════════════════════════════════════════
    logger.info("\n─── Assembling Multi-Modal Dataset ───")

    from features.dataset import MultiModalDataset

    dataset_kwargs = dict(
        price_features=price_features,
        labels=labels,
        sentiment_embeddings=sent_embeddings,
        sentiment_counts=sent_counts,
        fund_features=fund_data,
        macro_features=macro_df,
        denoise=denoise,
        tickers=list(price_features.keys()),
    )
    if lookback is not None:
        dataset_kwargs["lookback"] = lookback

    dataset = MultiModalDataset(**dataset_kwargs)

    # ═══════════════════════════════════════════════════════════
    # STEP 6: VALIDATE PROVENANCE
    # ═══════════════════════════════════════════════════════════
    logger.info("\n─── Provenance Validation ───")

    try:
        provenance.check_completeness(
            tickers=list(price_features.keys()),
            modalities=["price", "sentiment", "fundamentals", "macro"],
        )
        logger.info("  ✓ Provenance completeness check PASSED")
    except ValueError as e:
        logger.warning(f"  ⚠ Provenance incomplete: {e}")

    # Generate and save report
    report = provenance.report()
    provenance.save()

    logger.info("\n" + "=" * 60)
    logger.info(f"  DATASET READY: {len(dataset)} samples")
    logger.info(f"  Price: {dataset.X_price.shape}")
    logger.info(f"  Sentiment: {dataset.X_sent.shape}")
    logger.info(f"  Fundamentals: {dataset.X_fund.shape}")
    logger.info(f"  Macro: {dataset.X_macro.shape}")
    logger.info(f"  Labels: mean={dataset.y.mean():.3f}pp, std={dataset.y.std():.3f}pp")
    logger.info("=" * 60)

    return dataset, report


def build_quick_test_dataset(n_tickers: int = 2, n_days: int = 1200):
    """
    Build a synthetic dataset for quick pipeline testing.
    Does NOT use any external data sources.
    All modalities are explicitly marked SYNTHETIC in provenance.
    """
    logger.info("Building QUICK TEST dataset (all synthetic)...")

    set_seed(42)
    tickers = TICKERS[:n_tickers]
    dates = pd.bdate_range("2017-01-01", periods=n_days)
    n = len(dates)

    price_features = {}
    labels_dict = {}
    sent_emb = {}
    sent_cnt = {}
    fund_data = {}

    for ticker in tickers:
        # Synthetic price features
        feat = np.random.randn(n, 39).astype(np.float32)
        for j in range(39):
            for i in range(1, n):
                feat[i, j] = 0.3 * feat[i - 1, j] + 0.7 * feat[i, j]

        price_features[ticker] = pd.DataFrame(
            feat, index=dates, columns=[f"f{i}" for i in range(39)]
        )

        # Synthetic labels
        signal = feat[:, 0] * 0.5 + feat[:, 4] * 0.3
        noise = np.random.randn(n) * 3
        y = (signal + noise).astype(np.float32)
        y_series = pd.Series(y, index=dates, name="label")
        y_series.iloc[-3:] = np.nan
        labels_dict[ticker] = y_series

        # Synthetic sentiment (small dim)
        sent_emb[ticker] = np.zeros((n, 5), dtype=np.float32)
        sent_cnt[ticker] = np.zeros(n, dtype=np.float32)

        # Synthetic fundamentals
        fund_data[ticker] = pd.DataFrame(
            np.random.randn(n, 7).astype(np.float32),
            index=dates, columns=[f"fund_{i}" for i in range(7)]
        )

        # Register ALL as synthetic
        provenance.register(ticker, "price", "synthetic",
                            "Random synthetic features (for testing only)", n_features=39)
        provenance.register(ticker, "sentiment", "synthetic",
                            "Zero embeddings (for testing only)", n_features=6)
        provenance.register(ticker, "fundamentals", "synthetic",
                            "Random synthetic values (for testing only)", n_features=7)
        provenance.register(ticker, "macro", "synthetic",
                            "Random synthetic values (for testing only)", n_features=5)

    macro = pd.DataFrame(
        np.random.randn(n, 5).astype(np.float32),
        index=dates, columns=[f"macro_{i}" for i in range(5)]
    )

    from features.dataset import MultiModalDataset

    dataset = MultiModalDataset(
        price_features=price_features,
        labels=labels_dict,
        sentiment_embeddings=sent_emb,
        sentiment_counts=sent_cnt,
        fund_features=fund_data,
        macro_features=macro,
        denoise=False,
        tickers=tickers,
    )

    report = provenance.report()
    logger.info(f"Quick test dataset ready: {len(dataset)} samples")
    return dataset, report


if __name__ == "__main__":
    # Demo: build quick test dataset
    dataset, report = build_quick_test_dataset(n_tickers=2, n_days=1200)
    print(f"\nDataset: {len(dataset)} samples")
    sample = dataset[0]
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {v}")
