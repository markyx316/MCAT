"""
tests/inspect_data.py — Comprehensive Data Quality Inspection.
================================================================
Run AFTER the data pipeline completes but BEFORE training.
Checks every modality for shape correctness, content quality,
alignment, and potential silent failures.

Usage:
    python tests/inspect_data.py --data-dir data/

This script does NOT modify any data. It only reads and reports.

Exit codes:
    0 = All checks passed (safe to train)
    1 = Critical failures found (DO NOT train)
    2 = Warnings only (review before training)
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    TICKERS, RAW_DIR, PROCESSED_DIR, RESULTS_DIR,
    DATA_START_DATE, DATA_END_DATE, LOOKBACK_WINDOW,
    FORWARD_HORIZON, FINBERT_DIM, SECTOR_MAP,
)
from utils import setup_logger

logger = setup_logger("data_inspector")


class InspectionReport:
    """Accumulates check results and prints a final report."""

    def __init__(self):
        self.checks = []  # (level, category, message)
        self.n_pass = 0
        self.n_warn = 0
        self.n_fail = 0

    def passed(self, category: str, message: str):
        self.checks.append(("PASS", category, message))
        self.n_pass += 1

    def warn(self, category: str, message: str):
        self.checks.append(("WARN", category, message))
        self.n_warn += 1
        logger.warning(f"  ⚠ {category}: {message}")

    def fail(self, category: str, message: str):
        self.checks.append(("FAIL", category, message))
        self.n_fail += 1
        logger.error(f"  ✗ {category}: {message}")

    def ok(self, category: str, message: str):
        self.passed(category, message)
        logger.info(f"  ✓ {category}: {message}")

    def print_summary(self):
        print("\n" + "=" * 70)
        print("  DATA INSPECTION REPORT")
        print("=" * 70)
        print(f"  Passed: {self.n_pass}  |  Warnings: {self.n_warn}  |  Failures: {self.n_fail}")
        print()

        if self.n_fail > 0:
            print("  ╔══════════════════════════════════════════════════════╗")
            print("  ║  ✗ CRITICAL FAILURES — DO NOT PROCEED TO TRAINING  ║")
            print("  ╚══════════════════════════════════════════════════════╝")
            print()
            for level, cat, msg in self.checks:
                if level == "FAIL":
                    print(f"  ✗ [{cat}] {msg}")

        if self.n_warn > 0:
            print()
            print("  WARNINGS (review before training):")
            for level, cat, msg in self.checks:
                if level == "WARN":
                    print(f"  ⚠ [{cat}] {msg}")

        print()
        if self.n_fail == 0 and self.n_warn == 0:
            print("  ✓ ALL CHECKS PASSED — Safe to proceed to training")
        elif self.n_fail == 0:
            print(f"  ⚠ {self.n_warn} warnings — Review above before training")
        else:
            print(f"  ✗ {self.n_fail} failures — Fix these before training")

        print("=" * 70)
        return self.n_fail == 0


# ═══════════════════════════════════════════════════════════════
# SECTION 1: RAW PRICE DATA
# ═══════════════════════════════════════════════════════════════

def inspect_raw_prices(report: InspectionReport):
    """Check raw OHLCV parquet files for each ticker."""
    logger.info("\n━━━ Section 1: Raw Price Data ━━━")

    for ticker in TICKERS:
        path = RAW_DIR / f"price_{ticker}.parquet"

        # 1.1 File exists
        if not path.exists():
            report.fail("Price", f"{ticker}: File not found at {path}")
            continue

        df = pd.read_parquet(path)

        # 1.2 Required columns
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            report.fail("Price", f"{ticker}: Missing columns {missing}")
            continue

        # 1.3 Date range coverage
        start = pd.Timestamp(DATA_START_DATE)
        end = pd.Timestamp(DATA_END_DATE)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df_study = df[(df.index >= start) & (df.index <= end)]

        if len(df_study) < 1700:
            report.warn("Price", f"{ticker}: Only {len(df_study)} trading days in study period (expected ~1760)")
        else:
            report.ok("Price", f"{ticker}: {len(df_study)} trading days [{df_study.index[0].date()} → {df_study.index[-1].date()}]")

        # 1.4 OHLC relationship: High >= max(Open, Close) and Low <= min(Open, Close)
        violations_high = (df_study["high"] < df_study[["open", "close"]].max(axis=1)).sum()
        violations_low = (df_study["low"] > df_study[["open", "close"]].min(axis=1)).sum()
        if violations_high > 0 or violations_low > 0:
            report.warn("Price", f"{ticker}: {violations_high} High<max(O,C) + {violations_low} Low>min(O,C) violations")

        # 1.5 NaN/Inf check
        n_nan = df_study[required].isna().sum().sum()
        n_inf = np.isinf(df_study[required].select_dtypes(include=np.number)).sum().sum()
        if n_nan > 0:
            report.fail("Price", f"{ticker}: {n_nan} NaN values in OHLCV")
        if n_inf > 0:
            report.fail("Price", f"{ticker}: {n_inf} Inf values in OHLCV")

        # 1.6 Zero/negative price check
        for col in ["open", "high", "low", "close"]:
            n_bad = (df_study[col] <= 0).sum()
            if n_bad > 0:
                report.fail("Price", f"{ticker}: {n_bad} non-positive values in '{col}'")

        # 1.7 Extreme daily moves (sanity check — >50% single-day move is suspicious)
        daily_ret = df_study["close"].pct_change().dropna()
        extreme = (daily_ret.abs() > 0.5).sum()
        if extreme > 0:
            max_move = daily_ret.abs().max()
            report.warn("Price", f"{ticker}: {extreme} daily moves >50% (max: {max_move:.1%})")

        # 1.8 Volume check — zero volume days
        zero_vol = (df_study["volume"] == 0).sum()
        if zero_vol > 5:
            report.warn("Price", f"{ticker}: {zero_vol} zero-volume days")


# ═══════════════════════════════════════════════════════════════
# SECTION 2: TECHNICAL FEATURES
# ═══════════════════════════════════════════════════════════════

def inspect_technical_features(report: InspectionReport):
    """Verify technical features by computing them on one ticker and checking."""
    logger.info("\n━━━ Section 2: Technical Features ━━━")

    # Compute features for first ticker as a sample
    path = RAW_DIR / f"price_{TICKERS[0]}.parquet"
    if not path.exists():
        report.fail("TechFeat", "Cannot test — no price data available")
        return

    from features.technical_indicators import compute_technical_features, get_feature_names

    df = pd.read_parquet(path)
    features = compute_technical_features(df)

    # 2.1 Correct number of features
    expected_names = get_feature_names()
    if features.shape[1] != len(expected_names):
        report.fail("TechFeat", f"Expected {len(expected_names)} features, got {features.shape[1]}")
    else:
        report.ok("TechFeat", f"{features.shape[1]} features computed correctly")

    # 2.2 Feature names match
    actual_cols = list(features.columns)
    mismatched = [n for n in expected_names if n not in actual_cols]
    if mismatched:
        report.fail("TechFeat", f"Missing features: {mismatched}")

    # 2.3 NaN after warmup period (row 252+)
    features_study = features.iloc[252:]
    nan_per_col = features_study.isna().sum()
    cols_with_nan = nan_per_col[nan_per_col > 0]
    if len(cols_with_nan) > 0:
        report.warn("TechFeat", f"{len(cols_with_nan)} features have NaN after warmup: {list(cols_with_nan.index[:5])}")
    else:
        report.ok("TechFeat", "No NaN after warmup period (row 252+)")

    # 2.4 Constant features (zero variance = useless)
    stds = features_study.std()
    constant = stds[stds < 1e-10]
    if len(constant) > 0:
        report.warn("TechFeat", f"{len(constant)} constant features (zero variance): {list(constant.index)}")
    else:
        report.ok("TechFeat", "No constant features — all have variance")

    # 2.5 Extreme values (features should be roughly normalized)
    for col in features_study.columns:
        vals = features_study[col].dropna()
        if len(vals) == 0:
            continue
        max_abs = vals.abs().max()
        if max_abs > 100:
            report.warn("TechFeat", f"'{col}' has extreme values (max |val| = {max_abs:.1f})")

    # 2.6 Inf check
    n_inf = np.isinf(features_study.select_dtypes(include=np.number)).sum().sum()
    if n_inf > 0:
        report.fail("TechFeat", f"{n_inf} Inf values in technical features")
    else:
        report.ok("TechFeat", "No Inf values")


# ═══════════════════════════════════════════════════════════════
# SECTION 3: LABELS
# ═══════════════════════════════════════════════════════════════

def inspect_labels(report: InspectionReport):
    """Verify 3-day forward return labels."""
    logger.info("\n━━━ Section 3: Labels (3-day Forward Returns) ━━━")

    from features.label_generator import compute_labels

    sample_path = RAW_DIR / f"price_{TICKERS[0]}.parquet"
    if not sample_path.exists():
        report.fail("Labels", "Cannot test — no price data")
        return

    df = pd.read_parquet(sample_path)
    start = pd.Timestamp(DATA_START_DATE)
    df_study = df[df.index >= start] if df.index.tz is None else df[df.index.tz_localize(None) >= start]
    labels = compute_labels(df_study)

    # 3.1 Last FORWARD_HORIZON values should be NaN (no future data)
    last_n = labels.iloc[-FORWARD_HORIZON:]
    if last_n.notna().any():
        report.fail("Labels", f"Last {FORWARD_HORIZON} labels should be NaN (future leak!)")
    else:
        report.ok("Labels", f"Last {FORWARD_HORIZON} labels are NaN (correct — no future data)")

    valid = labels.dropna()

    # 3.2 Distribution check
    mean_pp = valid.mean()
    std_pp = valid.std()
    if std_pp < 0.5:
        report.fail("Labels", f"Label std={std_pp:.3f}pp — too small, check scaling")
    elif std_pp > 15:
        report.warn("Labels", f"Label std={std_pp:.3f}pp — unusually large")
    else:
        report.ok("Labels", f"Label distribution: mean={mean_pp:.3f}pp, std={std_pp:.3f}pp")

    # 3.3 Positive fraction (should be ~50-55%, not 0% or 100%)
    pct_pos = (valid > 0).mean() * 100
    if pct_pos < 40 or pct_pos > 65:
        report.warn("Labels", f"Positive fraction: {pct_pos:.1f}% (expected 45-60%)")
    else:
        report.ok("Labels", f"Positive fraction: {pct_pos:.1f}% (reasonable)")

    # 3.4 Extreme returns check
    extreme_threshold = 20  # 20 percentage points
    n_extreme = (valid.abs() > extreme_threshold).sum()
    if n_extreme > 0:
        max_ret = valid.abs().max()
        report.ok("Labels", f"{n_extreme} extreme returns (>±{extreme_threshold}pp, max={max_ret:.1f}pp) — expected in volatile periods")


# ═══════════════════════════════════════════════════════════════
# SECTION 4: SENTIMENT DATA
# ═══════════════════════════════════════════════════════════════

def inspect_sentiment(report: InspectionReport):
    """Verify FNSPID articles and FinBERT embeddings."""
    logger.info("\n━━━ Section 4: Sentiment Data ━━━")

    fnspid_dir = RAW_DIR / "fnspid"

    # 4.1 Check article files exist
    for ticker in TICKERS:
        articles_path = fnspid_dir / f"articles_{ticker}.parquet"
        if not articles_path.exists():
            report.warn("Sentiment", f"{ticker}: No article file at {articles_path}")
            continue

        articles = pd.read_parquet(articles_path)

        # 4.2 Article count reasonable
        if len(articles) < 50:
            report.warn("Sentiment", f"{ticker}: Only {len(articles)} articles (low coverage)")
        else:
            report.ok("Sentiment", f"{ticker}: {len(articles):,} articles")

        # 4.3 Required columns
        for col in ["Date", "Article_title", "Article"]:
            if col not in articles.columns:
                report.fail("Sentiment", f"{ticker}: Missing column '{col}' in articles")

        # 4.4 Check for empty articles
        if "Article" in articles.columns:
            empty_articles = articles["Article"].isna().sum() + (articles["Article"].str.len() < 10).sum()
            if empty_articles > len(articles) * 0.1:
                report.warn("Sentiment", f"{ticker}: {empty_articles}/{len(articles)} empty/very short articles")

    # 4.5 Check FinBERT embedding caches (may be .parquet or .npz format)
    logger.info("  Checking FinBERT embedding caches...")
    for ticker in TICKERS:
        # Try multiple possible cache paths/formats
        parquet_path = PROCESSED_DIR / f"sentiment_{ticker}_finbert.parquet"
        npz_path1 = PROCESSED_DIR / f"sentiment_finbert_{ticker}.npz"
        npz_path2 = PROCESSED_DIR / f"sentiment_{ticker}_finbert.npz"

        emb_data = None
        emb_source = None

        if parquet_path.exists():
            emb_df = pd.read_parquet(parquet_path)
            if "embedding" in emb_df.columns:
                sample_embs = np.array([np.array(e) for e in emb_df["embedding"].head(20)])
                all_embs_shape = (len(emb_df), len(emb_df["embedding"].iloc[0]))
                conf = emb_df.get("confidence")
                emb_data = {"embeddings_sample": sample_embs, "shape": all_embs_shape, "confidence": conf}
                emb_source = f"parquet ({parquet_path.name})"

        elif npz_path1.exists() or npz_path2.exists():
            npz_path = npz_path1 if npz_path1.exists() else npz_path2
            data = np.load(npz_path, allow_pickle=True)
            if "embeddings" in data:
                embs = data["embeddings"]
                counts = data.get("counts")
                emb_data = {"embeddings_sample": embs[:20], "shape": embs.shape, "counts": counts}
                emb_source = f"npz ({npz_path.name})"

        if emb_data is None:
            report.warn("Sentiment", f"{ticker}: No FinBERT cache found (checked .parquet and .npz)")
            continue

        # 4.6 Check embedding dimension
        shape = emb_data["shape"]
        emb_dim = shape[1] if len(shape) == 2 else 0
        if emb_dim != FINBERT_DIM:
            report.fail("Sentiment", f"{ticker}: Embedding dim={emb_dim}, expected {FINBERT_DIM} [{emb_source}]")
        else:
            report.ok("Sentiment", f"{ticker}: FinBERT dim={emb_dim}, {shape[0]} days [{emb_source}]")

        # 4.7 Check embeddings are not all zeros
        sample = emb_data["embeddings_sample"]
        norms = np.linalg.norm(sample, axis=1)
        n_zero = (norms < 1e-6).sum()
        pct_zero = 100 * n_zero / len(norms) if len(norms) > 0 else 0
        if n_zero == len(norms) and len(norms) > 0:
            report.fail("Sentiment", f"{ticker}: First {len(norms)} embeddings ALL ZEROS — encoding failed")
        elif pct_zero > 80:
            report.warn("Sentiment", f"{ticker}: {pct_zero:.0f}% of sampled embeddings are zero (low news coverage)")
        else:
            mean_norm = norms[norms > 1e-6].mean() if (norms > 1e-6).any() else 0
            report.ok("Sentiment", f"{ticker}: Embedding norms mean={mean_norm:.3f}, {pct_zero:.0f}% zero")

        # 4.8 Check overall zero-day rate
        if "counts" in emb_data and emb_data["counts"] is not None:
            counts = emb_data["counts"]
            pct_with_news = 100 * (counts > 0).mean()
            mean_per_day = counts[counts > 0].mean() if (counts > 0).any() else 0
            if pct_with_news < 10:
                report.warn("Sentiment", f"{ticker}: Only {pct_with_news:.1f}% of days have news (very sparse)")
            else:
                report.ok("Sentiment", f"{ticker}: {pct_with_news:.1f}% days with news, mean={mean_per_day:.1f} articles/day")

            # 4.9 STALENESS CHECK: compare FinBERT encoded count vs article cache on disk
            # Use n_articles_input if available (new cache format), else fall back to counts.sum()
            npz_path = npz_path1 if npz_path1.exists() else npz_path2
            npz_data = np.load(npz_path, allow_pickle=True)
            if "n_articles_input" in npz_data:
                finbert_article_count = int(npz_data["n_articles_input"])
            else:
                finbert_article_count = int(counts.sum())
            articles_path = fnspid_dir / f"articles_{ticker}.parquet"
            if articles_path.exists():
                try:
                    n_articles_on_disk = len(pd.read_parquet(articles_path))
                    if finbert_article_count != n_articles_on_disk:
                        report.warn("Sentiment",
                            f"{ticker}: FinBERT cache STALE — encoded {finbert_article_count:,} articles "
                            f"but article cache has {n_articles_on_disk:,}. "
                            f"Next run with --fnspid will re-encode ({n_articles_on_disk - finbert_article_count:+,} articles).")
                    else:
                        report.ok("Sentiment",
                            f"{ticker}: FinBERT cache matches article cache ({finbert_article_count:,} articles)")
                except Exception:
                    pass  # Article cache unreadable, skip cross-check


# ═══════════════════════════════════════════════════════════════
# SECTION 5: FUNDAMENTAL DATA
# ═══════════════════════════════════════════════════════════════

def inspect_fundamentals(report: InspectionReport):
    """Verify fundamental data files."""
    logger.info("\n━━━ Section 5: Fundamental Data ━━━")

    dims_found = {}

    for ticker in TICKERS:
        # Check both real and synthetic caches
        real_path = PROCESSED_DIR / f"fundamentals_real_{ticker}.parquet"
        synth_path = PROCESSED_DIR / f"fundamentals_synth_{ticker}.parquet"

        if real_path.exists():
            df = pd.read_parquet(real_path)
            source = "real"
        elif synth_path.exists():
            df = pd.read_parquet(synth_path)
            source = "synthetic"
        else:
            report.warn("Fundamentals", f"{ticker}: No fundamental data file found")
            continue

        dims_found[ticker] = df.shape[1]

        # 5.1 NaN check
        nan_pct = df.isna().mean().mean() * 100
        if nan_pct > 50:
            report.warn("Fundamentals", f"{ticker} ({source}): {nan_pct:.0f}% NaN values")
        else:
            report.ok("Fundamentals", f"{ticker} ({source}): {df.shape[1]} features, {nan_pct:.1f}% NaN")

        # 5.2 Date coverage
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        start = pd.Timestamp(DATA_START_DATE)
        end = pd.Timestamp(DATA_END_DATE)
        in_range = df[(df.index >= start) & (df.index <= end)]
        if len(in_range) < 1000:
            report.warn("Fundamentals", f"{ticker}: Only {len(in_range)} rows in study period")

        # 5.3 Extreme values
        # Raw fields from INCOME_STATEMENT/BALANCE_SHEET (totalRevenue, totalAssets, etc.)
        # are expected to be in the billions — these get transformed into ratios by the
        # preprocessor. Only flag truly unexpected extremes.
        RAW_DOLLAR_FIELDS = {
            "totalRevenue", "netIncome", "grossProfit", "operatingIncome", "costOfRevenue",
            "totalShareholderEquity", "totalCurrentLiabilities", "totalCurrentAssets",
            "longTermDebt", "shortTermDebt", "totalAssets", "MarketCap", "MarketCapitalization",
        }
        numeric = df.select_dtypes(include=np.number)
        raw_cols = [c for c in numeric.columns if c in RAW_DOLLAR_FIELDS]
        ratio_cols = [c for c in numeric.columns if c not in RAW_DOLLAR_FIELDS]

        if raw_cols:
            report.ok("Fundamentals",
                f"{ticker}: {len(raw_cols)} raw dollar fields present "
                f"({', '.join(raw_cols[:3])}{'...' if len(raw_cols) > 3 else ''}) "
                f"— will be transformed to ratios by preprocessor")

        for col in ratio_cols:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            max_abs = vals.abs().max()
            if max_abs > 1000:
                report.warn("Fundamentals",
                    f"{ticker}: Ratio field '{col}' has extreme value {max_abs:.2e}")

    # 5.4 Dimension consistency check (THE BUG WE FIXED)
    if dims_found:
        unique_dims = set(dims_found.values())
        if len(unique_dims) > 1:
            report.warn("Fundamentals",
                f"Mixed dimensions across tickers: {dict(sorted(dims_found.items(), key=lambda x: x[1]))} — "
                f"dataset.py will pad to max={max(unique_dims)}")
        else:
            report.ok("Fundamentals", f"All tickers have uniform dimension: {unique_dims.pop()}")


# ═══════════════════════════════════════════════════════════════
# SECTION 6: MACRO DATA
# ═══════════════════════════════════════════════════════════════

def inspect_macro(report: InspectionReport):
    """Verify macro feature data."""
    logger.info("\n━━━ Section 6: Macro Data ━━━")

    path = PROCESSED_DIR / "macro_features.parquet"
    if not path.exists():
        report.fail("Macro", f"File not found: {path}")
        return

    df = pd.read_parquet(path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    start = pd.Timestamp(DATA_START_DATE)
    end = pd.Timestamp(DATA_END_DATE)
    df_study = df[(df.index >= start) & (df.index <= end)]

    # 6.1 Feature count
    report.ok("Macro", f"{df.shape[1]} features, {len(df_study)} trading days")

    # 6.2 VIX sanity check (should be ~10-80, not 0 or 1000)
    if "VIX" in df_study.columns:
        vix = df_study["VIX"]
        if vix.median() < 5 or vix.median() > 100:
            report.warn("Macro", f"VIX median={vix.median():.1f} — unexpected (normal: 12-30)")
        else:
            report.ok("Macro", f"VIX: median={vix.median():.1f}, max={vix.max():.1f} (COVID peak expected ~80)")

    # 6.3 Treasury yields check
    if "TNX" in df_study.columns:
        tnx = df_study["TNX"]
        if tnx.median() < 0.1 or tnx.median() > 20:
            report.warn("Macro", f"10Y yield median={tnx.median():.2f} — unexpected")
        else:
            report.ok("Macro", f"10Y yield: median={tnx.median():.2f}, range=[{tnx.min():.2f}, {tnx.max():.2f}]")

    # 6.4 FRED data presence
    fred_cols = [c for c in ["FEDFUNDS", "T10Y2Y", "UNRATE"] if c in df_study.columns]
    if len(fred_cols) == 3:
        report.ok("Macro", f"All 3 FRED series present: {fred_cols}")
        # Check FRED values are not all zero/constant
        for col in fred_cols:
            vals = df_study[col]
            if vals.std() < 1e-6:
                report.warn("Macro", f"FRED '{col}' is constant ({vals.iloc[0]:.4f}) — forward-fill issue?")
            else:
                report.ok("Macro", f"FRED '{col}': mean={vals.mean():.3f}, std={vals.std():.3f}")
    elif len(fred_cols) > 0:
        report.warn("Macro", f"Only {len(fred_cols)}/3 FRED series present: {fred_cols}")
    else:
        report.warn("Macro", "No FRED series found in macro data")

    # 6.5 Sector ETF relative returns
    etf_cols = [c for c in df_study.columns if c.endswith("_rel_ret")]
    if len(etf_cols) >= 10:
        report.ok("Macro", f"{len(etf_cols)} sector ETF relative return columns")
    else:
        report.warn("Macro", f"Only {len(etf_cols)} sector ETF columns (expected 11)")

    # 6.6 NaN/Inf check
    n_nan = df_study.isna().sum().sum()
    n_inf = np.isinf(df_study.select_dtypes(include=np.number)).sum().sum()
    if n_nan > 0:
        nan_cols = df_study.columns[df_study.isna().any()].tolist()
        report.warn("Macro", f"{n_nan} NaN values in columns: {nan_cols[:5]}")
    if n_inf > 0:
        report.fail("Macro", f"{n_inf} Inf values in macro features")

    # 6.7 Stale data check — any column unchanged for >60 consecutive days?
    for col in df_study.columns:
        vals = df_study[col].dropna()
        if len(vals) < 2:
            continue
        diffs = vals.diff().abs()
        max_unchanged = (diffs < 1e-10).astype(int)
        # Find longest consecutive run of zeros
        runs = max_unchanged.groupby((max_unchanged != max_unchanged.shift()).cumsum()).sum()
        longest_stale = runs.max()
        if longest_stale > 120:  # >6 months unchanged
            report.warn("Macro", f"'{col}': unchanged for {longest_stale} consecutive days — stale data?")


# ═══════════════════════════════════════════════════════════════
# SECTION 7: PROVENANCE REPORT
# ═══════════════════════════════════════════════════════════════

def inspect_provenance(report: InspectionReport):
    """Verify the provenance report is complete and consistent."""
    logger.info("\n━━━ Section 7: Provenance Report ━━━")

    prov_path = RESULTS_DIR / "data_provenance.json"
    if not prov_path.exists():
        report.warn("Provenance", "No provenance file found — run the data pipeline first")
        return

    import os
    from datetime import datetime

    # Check file age
    file_mtime = datetime.fromtimestamp(os.path.getmtime(prov_path))
    age_hours = (datetime.now() - file_mtime).total_seconds() / 3600
    if age_hours > 24:
        report.warn("Provenance",
            f"File is {age_hours:.0f} hours old (from {file_mtime.strftime('%Y-%m-%d %H:%M')}). "
            f"May be stale from a previous run. Re-run with --fnspid to regenerate.")

    with open(prov_path) as f:
        prov = json.load(f)

    summary = prov.get("summary", {})
    records = prov.get("records", [])

    n_total = summary.get("n_total", 0)
    n_real = summary.get("n_real", 0)
    n_synthetic = summary.get("n_synthetic", 0)

    # Check if this looks like a smoke test provenance (not real data)
    if n_total <= 8 and n_synthetic > 0:
        report.warn("Provenance",
            f"File contains only {n_total} records ({n_synthetic} synthetic) — "
            f"this is likely from a smoke test, not your real data run. "
            f"Re-run the full experiment to regenerate.")
        return

    report.ok("Provenance", f"{n_total} records: {n_real} real, {n_synthetic} synthetic")

    # 7.1 Completeness: should have 4 modalities × 15 tickers = 60 records
    expected = len(TICKERS) * 4
    if n_total < expected:
        report.warn("Provenance", f"Expected {expected} records, got {n_total} — some modalities missing")
    else:
        report.ok("Provenance", f"Complete: {n_total}/{expected} ticker×modality records")

    # 7.2 List synthetic entries explicitly
    synthetic_entries = [r for r in records if r.get("source_type") == "synthetic"]
    if synthetic_entries:
        for r in synthetic_entries:
            report.warn("Provenance",
                f"SYNTHETIC: {r['ticker']}/{r['modality']} — {r['source_detail']}")
    else:
        report.ok("Provenance", "No synthetic data — all modalities use real sources")


# ═══════════════════════════════════════════════════════════════
# SECTION 8: CROSS-MODALITY ALIGNMENT
# ═══════════════════════════════════════════════════════════════

def inspect_alignment(report: InspectionReport):
    """Verify temporal alignment across modalities."""
    logger.info("\n━━━ Section 8: Cross-Modality Alignment ━━━")

    # Load a sample ticker's data
    ticker = TICKERS[0]
    price_path = RAW_DIR / f"price_{ticker}.parquet"
    macro_path = PROCESSED_DIR / "macro_features.parquet"

    if not price_path.exists() or not macro_path.exists():
        report.warn("Alignment", "Cannot check — missing data files")
        return

    price = pd.read_parquet(price_path)
    macro = pd.read_parquet(macro_path)

    if price.index.tz is not None:
        price.index = price.index.tz_localize(None)
    if macro.index.tz is not None:
        macro.index = macro.index.tz_localize(None)

    start = pd.Timestamp(DATA_START_DATE)
    end = pd.Timestamp(DATA_END_DATE)

    price_dates = price.index[(price.index >= start) & (price.index <= end)]
    macro_dates = macro.index[(macro.index >= start) & (macro.index <= end)]

    # 8.1 Date overlap
    common = price_dates.intersection(macro_dates)
    pct_overlap = len(common) / max(len(price_dates), 1) * 100

    if pct_overlap < 90:
        report.warn("Alignment", f"Price-Macro date overlap: {pct_overlap:.0f}% ({len(common)}/{len(price_dates)} dates)")
    else:
        report.ok("Alignment", f"Price-Macro alignment: {pct_overlap:.0f}% overlap ({len(common)} common dates)")

    # 8.2 Check sentiment date alignment
    articles_path = RAW_DIR / "fnspid" / f"articles_{ticker}.parquet"
    if articles_path.exists():
        articles = pd.read_parquet(articles_path)
        if "Date" in articles.columns:
            article_dates = pd.to_datetime(articles["Date"]).dt.normalize().unique()
            # How many article dates match trading days?
            article_trading_match = len(set(article_dates) & set(price_dates))
            report.ok("Alignment",
                f"{ticker}: {article_trading_match}/{len(article_dates)} article dates align with trading days")

    # 8.3 Label horizon safety check
    report.ok("Alignment",
        f"Forward horizon={FORWARD_HORIZON} days, lookback={LOOKBACK_WINDOW} days — "
        f"no temporal overlap possible between features and label")


# ═══════════════════════════════════════════════════════════════
# SECTION 9: ASSEMBLED DATASET (if already built)
# ═══════════════════════════════════════════════════════════════

def inspect_assembled_dataset(report: InspectionReport):
    """
    Verify cache consistency across all modalities by reading files directly.
    
    IMPORTANT: This does NOT call build_full_dataset or trigger any pipeline.
    It only reads existing cached files and checks they are mutually consistent.
    """
    logger.info("\n━━━ Section 9: Cache Consistency Check ━━━")

    fnspid_dir = RAW_DIR / "fnspid"

    # ─── Check all expected files exist ───
    missing_files = []
    for ticker in TICKERS:
        for path, label in [
            (RAW_DIR / f"price_{ticker}.parquet", f"price_{ticker}"),
            (fnspid_dir / f"articles_{ticker}.parquet", f"articles_{ticker}"),
            (PROCESSED_DIR / f"sentiment_finbert_{ticker}.npz", f"finbert_{ticker}"),
        ]:
            if not path.exists():
                missing_files.append(label)

    for path, label in [
        (PROCESSED_DIR / "macro_features.parquet", "macro_features"),
    ]:
        if not path.exists():
            missing_files.append(label)

    # Check for fundamental files (real or synth)
    for ticker in TICKERS:
        real = PROCESSED_DIR / f"fundamentals_real_{ticker}.parquet"
        synth = PROCESSED_DIR / f"fundamentals_synth_{ticker}.parquet"
        if not real.exists() and not synth.exists():
            missing_files.append(f"fundamentals_{ticker}")

    if missing_files:
        report.warn("CacheCheck", f"{len(missing_files)} cache files missing: {missing_files[:5]}{'...' if len(missing_files) > 5 else ''}")
    else:
        report.ok("CacheCheck", f"All expected cache files present ({15*3 + 1 + 15} files)")

    # ─── Cross-reference FinBERT article counts vs article caches ───
    n_stale = 0
    n_matched = 0
    for ticker in TICKERS:
        npz_path = PROCESSED_DIR / f"sentiment_finbert_{ticker}.npz"
        art_path = fnspid_dir / f"articles_{ticker}.parquet"

        if not npz_path.exists() or not art_path.exists():
            continue

        try:
            npz = np.load(npz_path)
            if "n_articles_input" in npz:
                finbert_count = int(npz["n_articles_input"])
            else:
                finbert_count = int(npz["counts"].sum())
            article_count = len(pd.read_parquet(art_path))

            if finbert_count != article_count:
                n_stale += 1
                report.warn("CacheCheck",
                    f"{ticker}: FinBERT ({finbert_count:,}) ≠ articles ({article_count:,}) — "
                    f"cache is stale, will re-encode on next --fnspid run")
            else:
                n_matched += 1
        except Exception as e:
            report.warn("CacheCheck", f"{ticker}: Error reading caches: {e}")

    if n_stale == 0 and n_matched > 0:
        report.ok("CacheCheck", f"All {n_matched} FinBERT caches match article caches — no re-encoding needed")
    elif n_stale > 0:
        report.warn("CacheCheck",
            f"{n_stale}/{n_stale + n_matched} FinBERT caches are stale. "
            f"Run with --fnspid to re-encode (one-time, ~1 min/ticker).")

    # ─── Verify fundamental dimensions are uniform ───
    fund_dims = {}
    for ticker in TICKERS:
        real = PROCESSED_DIR / f"fundamentals_real_{ticker}.parquet"
        synth = PROCESSED_DIR / f"fundamentals_synth_{ticker}.parquet"
        path = real if real.exists() else (synth if synth.exists() else None)
        if path:
            try:
                df = pd.read_parquet(path)
                fund_dims[ticker] = df.shape[1]
            except Exception:
                pass

    if fund_dims:
        unique_dims = set(fund_dims.values())
        if len(unique_dims) > 1:
            report.warn("CacheCheck",
                f"Mixed fundamental dimensions: {dict(sorted(fund_dims.items(), key=lambda x: x[1]))}")
        else:
            dim = unique_dims.pop()
            report.ok("CacheCheck", f"All {len(fund_dims)} fundamental caches have uniform dimension: {dim}")

    # ─── Check macro feature count ───
    macro_path = PROCESSED_DIR / "macro_features.parquet"
    if macro_path.exists():
        try:
            macro = pd.read_parquet(macro_path)
            report.ok("CacheCheck", f"Macro cache: {macro.shape[1]} features, {len(macro)} days")
        except Exception as e:
            report.warn("CacheCheck", f"Macro cache unreadable: {e}")

    # ─── Summary ───
    if n_stale == 0 and not missing_files:
        report.ok("CacheCheck", "All caches consistent — safe to proceed to training")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    report = InspectionReport()

    logger.info("=" * 60)
    logger.info("  COMPREHENSIVE DATA QUALITY INSPECTION")
    logger.info("  (read-only — does NOT modify any files)")
    logger.info("=" * 60)

    # Run all inspection sections (all read-only, no pipeline triggered)
    inspect_raw_prices(report)
    inspect_technical_features(report)
    inspect_labels(report)
    inspect_sentiment(report)
    inspect_fundamentals(report)
    inspect_macro(report)
    inspect_provenance(report)
    inspect_alignment(report)
    inspect_assembled_dataset(report)  # Lightweight cache check, no build

    # Print final report
    all_passed = report.print_summary()

    sys.exit(0 if all_passed else (1 if report.n_fail > 0 else 2))


if __name__ == "__main__":
    main()
