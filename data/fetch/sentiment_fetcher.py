"""
data/fetch/sentiment_fetcher.py — FNSPID → FinBERT Sentiment Pipeline.
========================================================================
Processes the FNSPID dataset (>20GB) in chunks to extract news articles
for our 15 tickers, encodes them with FinBERT, and aggregates to daily
768-dim embedding vectors with confidence-weighted averaging.

Pipeline:
  1. Read FNSPID CSV in chunks (50K rows at a time)
  2. Filter to our 15 tickers + date range
  3. Save filtered articles per-ticker as parquet (Stage 1)
  4. Encode title + first 500 chars of article body with FinBERT (Stage 2)
  5. Aggregate to daily embeddings with confidence weighting (Stage 3)

The user's insight: combining Article_title + Article body is more
informative than title alone, since the article body contains details
about earnings, guidance, analyst reasoning, etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    TICKERS, TICKER_ALIASES, PROCESSED_DIR, RAW_DIR,
    DATA_START_DATE, DATA_END_DATE,
    FNSPID_CONFIG, FINBERT_DIM, FINBERT_ARTICLE_MAX_CHARS,
    FINBERT_MODEL_NAME, FINBERT_BATCH_SIZE,
)
from data.provenance import provenance
from utils import setup_logger, timer

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────
# STAGE 1: Extract and filter FNSPID (chunked reading)
# ─────────────────────────────────────────────────────────────

def _build_ticker_set() -> set:
    """Build the full set of ticker symbols to filter for, including aliases."""
    tickers = set(TICKERS)
    for canonical, aliases in TICKER_ALIASES.items():
        tickers.update(aliases)
    return tickers


def _parse_fnspid_date(date_str: str) -> Optional[pd.Timestamp]:
    """Parse FNSPID date format: '2023-12-16 23:00:00 UTC'."""
    try:
        dt = pd.to_datetime(date_str, utc=True)
        return dt.tz_localize(None)
    except Exception:
        return pd.NaT


def _normalize_ticker(ticker: str) -> str:
    """Map ticker aliases to canonical form (e.g., 'FB' → 'META')."""
    for canonical, aliases in TICKER_ALIASES.items():
        if ticker in aliases:
            return canonical
    return ticker


@timer
def extract_fnspid_articles(
    fnspid_path: str,
    output_dir: Path = None,
    chunk_size: int = None,
) -> Dict[str, pd.DataFrame]:
    """
    Stage 1: Read the FNSPID CSV in chunks, filter to our tickers and dates.

    This handles the >20GB file by reading in manageable chunks, never
    loading the entire dataset into memory.

    Args:
        fnspid_path: Path to the FNSPID CSV file.
        output_dir: Where to save per-ticker parquet files.
        chunk_size: Number of rows per chunk.

    Returns:
        Dict mapping ticker → DataFrame of filtered articles.
    """
    if output_dir is None:
        output_dir = RAW_DIR / "fnspid"
        output_dir.mkdir(parents=True, exist_ok=True)
    if chunk_size is None:
        chunk_size = FNSPID_CONFIG["chunk_size"]

    target_tickers = _build_ticker_set()
    start_date = pd.Timestamp(DATA_START_DATE)
    end_date = pd.Timestamp(DATA_END_DATE)

    # Accumulators per ticker
    ticker_articles: Dict[str, List[pd.DataFrame]] = {t: [] for t in TICKERS}
    total_read = 0
    total_kept = 0

    logger.info(f"Reading FNSPID from {fnspid_path} in chunks of {chunk_size:,}...")
    logger.info(f"Filtering for {len(target_tickers)} ticker symbols, "
                f"date range [{start_date.date()} → {end_date.date()}]")

    try:
        reader = pd.read_csv(
            fnspid_path,
            chunksize=chunk_size,
            usecols=FNSPID_CONFIG["usecols"],
            dtype={
                FNSPID_CONFIG["ticker_col"]: str,
                FNSPID_CONFIG["title_col"]: str,
                FNSPID_CONFIG["article_col"]: str,
            },
            on_bad_lines="skip",
        )
    except ValueError:
        # Older pandas versions don't support on_bad_lines
        reader = pd.read_csv(
            fnspid_path,
            chunksize=chunk_size,
            usecols=FNSPID_CONFIG["usecols"],
            dtype={
                FNSPID_CONFIG["ticker_col"]: str,
                FNSPID_CONFIG["title_col"]: str,
                FNSPID_CONFIG["article_col"]: str,
            },
            error_bad_lines=False,
        )

    for chunk_num, chunk in enumerate(reader):
        total_read += len(chunk)

        # ─── PATH A: Articles directly tagged to our tickers (original logic) ───
        ticker_col = FNSPID_CONFIG["ticker_col"]
        mask_ticker = chunk[ticker_col].isin(target_tickers)
        filtered_direct = chunk[mask_ticker].copy()

        # ─── PATH B: Articles tagged to OTHER tickers but mentioning our companies ───
        # This catches articles like "Apple and Tesla compete in EVs" tagged as TSLA,
        # which would otherwise be completely discarded.
        mask_other = ~mask_ticker
        other_articles = chunk[mask_other]
        filtered_keyword = pd.DataFrame()

        if not other_articles.empty:
            from features.article_enrichment import find_mentioned_tickers
            title_col = FNSPID_CONFIG["title_col"]
            # Scan titles for mentions of our companies (vectorized check first for speed)
            titles = other_articles[title_col].fillna("")
            # Quick pre-filter: only scan rows whose title contains at least one of our keywords
            # This avoids calling find_mentioned_tickers on every row
            has_potential = titles.str.contains(
                "|".join(["Apple", "Microsoft", "Google", "Amazon", "Facebook", "Meta",
                         "Nvidia", "JPMorgan", "Goldman", "Johnson", "UnitedHealth",
                         "Walmart", "Procter", "Exxon", "Caterpillar", "Disney",
                         "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "JNJ",
                         "UNH", "WMT", "XOM", "Big Tech", "FAANG", "Magnificent"]),
                case=False, na=False
            )
            candidates = other_articles[has_potential]

            if not candidates.empty:
                # For each candidate, find which of our tickers are mentioned
                keyword_rows = []
                for idx, row in candidates.iterrows():
                    title = str(row.get(title_col, ""))
                    mentioned = find_mentioned_tickers(title)
                    for t in mentioned:
                        if t in TICKERS:
                            row_copy = row.copy()
                            row_copy[ticker_col] = t  # Re-tag to matched ticker
                            row_copy["_keyword_matched"] = True
                            keyword_rows.append(row_copy)
                if keyword_rows:
                    filtered_keyword = pd.DataFrame(keyword_rows)

        # ─── Combine both paths ───
        parts = []
        if not filtered_direct.empty:
            filtered_direct["_keyword_matched"] = False
            parts.append(filtered_direct)
        if not filtered_keyword.empty:
            parts.append(filtered_keyword)

        if not parts:
            if chunk_num % 100 == 0:
                logger.info(f"  Chunk {chunk_num}: {total_read:,} rows read, {total_kept:,} kept")
            continue

        filtered = pd.concat(parts, ignore_index=True)

        # Parse dates
        date_col = FNSPID_CONFIG["date_col"]
        filtered[date_col] = filtered[date_col].apply(_parse_fnspid_date)
        filtered = filtered.dropna(subset=[date_col])

        # Filter by date range
        mask_date = (filtered[date_col] >= start_date) & (filtered[date_col] <= end_date)
        filtered = filtered[mask_date]

        if filtered.empty:
            continue

        # Normalize tickers and distribute to accumulators
        filtered[ticker_col] = filtered[ticker_col].apply(_normalize_ticker)

        for ticker in TICKERS:
            ticker_rows = filtered[filtered[ticker_col] == ticker]
            if not ticker_rows.empty:
                ticker_articles[ticker].append(ticker_rows)
                total_kept += len(ticker_rows)

        if chunk_num % 50 == 0:
            n_kw = len(filtered_keyword) if not filtered_keyword.empty else 0
            logger.info(f"  Chunk {chunk_num}: {total_read:,} rows read, "
                       f"{total_kept:,} kept (this chunk: {n_kw} keyword-matched)")

    logger.info(f"Finished reading: {total_read:,} total rows, {total_kept:,} kept")

    # Concatenate, deduplicate, and save per ticker
    result = {}
    for ticker in TICKERS:
        if ticker_articles[ticker]:
            df = pd.concat(ticker_articles[ticker], ignore_index=True)

            # Deduplicate: same title + same date = same article
            n_before = len(df)
            title_col_name = FNSPID_CONFIG["title_col"]
            date_col_name = FNSPID_CONFIG["date_col"]
            if title_col_name in df.columns and date_col_name in df.columns:
                # Keep direct-tagged articles over keyword-matched ones
                if "_keyword_matched" in df.columns:
                    df = df.sort_values("_keyword_matched", ascending=True)
                df = df.drop_duplicates(subset=[title_col_name, date_col_name], keep="first")
            n_dupes = n_before - len(df)

            df = df.sort_values(date_col_name).reset_index(drop=True)

            # Save
            out_path = output_dir / f"articles_{ticker}.parquet"
            df.to_parquet(out_path)
            result[ticker] = df

            n_kw = df["_keyword_matched"].sum() if "_keyword_matched" in df.columns else 0
            dedup_msg = f", {n_dupes} dupes removed" if n_dupes > 0 else ""
            logger.info(f"  {ticker}: {len(df):,} articles saved → {out_path} "
                       f"({n_kw:,} from keyword matching{dedup_msg})")
        else:
            logger.warning(f"  {ticker}: 0 articles found")

    return result


# ─────────────────────────────────────────────────────────────
# STAGE 2: FinBERT encoding
# ─────────────────────────────────────────────────────────────

def _prepare_text_for_finbert(
    title: str,
    article: str,
    max_article_chars: int = FINBERT_ARTICLE_MAX_CHARS,
) -> str:
    """
    Combine article title + first N characters of article body.

    The user correctly identified that combining title + body is more
    informative than title alone:
    - Title captures the headline signal ("earnings beat", "CEO resigns")
    - Article body adds context ("beat by 15%", "effective immediately")

    FinBERT has a 512-token limit (~2000 chars). Title (~64 chars) +
    first 500 chars of body (~564 chars total) ≈ ~140 tokens — safely
    within limits.
    """
    title = str(title).strip() if pd.notna(title) else ""
    article = str(article).strip() if pd.notna(article) else ""

    if article:
        # Take first N chars of article body
        article_snippet = article[:max_article_chars]
        combined = f"{title}. {article_snippet}"
    else:
        combined = title

    return combined


@timer
def encode_articles_finbert(
    articles_df: pd.DataFrame,
    ticker: str,
    model=None,
    tokenizer=None,
    batch_size: int = FINBERT_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Stage 2: Encode articles with FinBERT.

    For each article, produces:
    - 768-dim [CLS] embedding (from second-to-last hidden layer)
    - 3-class sentiment probabilities (positive, negative, neutral)
    - Confidence score (max of the 3 probabilities)

    Args:
        articles_df: DataFrame with Article_title, Article, Date columns.
        ticker: Ticker symbol (for logging).
        model: Pre-loaded FinBERT model (if None, loads automatically).
        tokenizer: Pre-loaded tokenizer.
        batch_size: Batch size for GPU inference.

    Returns:
        DataFrame with columns: date, embedding (768-dim np.array), confidence.
    """
    import torch

    if model is None or tokenizer is None:
        logger.info(f"  Loading FinBERT model: {FINBERT_MODEL_NAME}")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            FINBERT_MODEL_NAME, output_hidden_states=True
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
    else:
        device = next(model.parameters()).device

    title_col = FNSPID_CONFIG["title_col"]
    article_col = FNSPID_CONFIG["article_col"]
    date_col = FNSPID_CONFIG["date_col"]

    # Prepare texts
    texts = []
    for _, row in articles_df.iterrows():
        text = _prepare_text_for_finbert(row[title_col], row[article_col])
        texts.append(text)

    dates = pd.to_datetime(articles_df[date_col]).values
    n = len(texts)
    logger.info(f"  {ticker}: Encoding {n} articles with FinBERT...")

    embeddings = []
    confidences = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_texts = texts[i : i + batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)

            # Extract [CLS] embedding from second-to-last hidden layer
            # Shape: (batch, seq_len, 768)
            hidden_states = outputs.hidden_states[-2]
            cls_embeddings = hidden_states[:, 0, :].cpu().numpy()

            # Extract sentiment probabilities
            logits = outputs.logits.cpu()
            probs = torch.softmax(logits, dim=-1).numpy()
            batch_confidence = probs.max(axis=1)

            embeddings.append(cls_embeddings)
            confidences.append(batch_confidence)

            if (i // batch_size) % 20 == 0 and i > 0:
                logger.info(f"    Progress: {i}/{n} articles ({100*i/n:.0f}%)")

    embeddings = np.vstack(embeddings)
    confidences = np.concatenate(confidences)

    logger.info(f"  {ticker}: Encoding complete. Shape: {embeddings.shape}")

    result = pd.DataFrame({
        "date": dates,
        "confidence": confidences,
    })
    # Store embeddings as a list of arrays (for parquet compatibility)
    result["embedding"] = [embeddings[i] for i in range(len(embeddings))]

    return result


# ─────────────────────────────────────────────────────────────
# STAGE 3: Daily aggregation with confidence-weighted averaging
# ─────────────────────────────────────────────────────────────

def aggregate_daily_sentiment(
    encoded_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 3: Aggregate per-article embeddings to daily representations.

    For each trading day, compute confidence-weighted average embedding:
        e_bar_t = Σ(c_k * e_k) / Σ(c_k)
    where c_k is FinBERT's confidence for article k.

    Articles are assigned to trading days based on publication date.
    Articles published on weekends/holidays are assigned to the NEXT
    trading day (standard in financial NLP — weekend news affects
    Monday's market open).

    Days with no articles get a zero embedding (the model will learn
    a "no news" embedding during training).

    Args:
        encoded_df: DataFrame with date, embedding, confidence columns.
        trading_dates: DatetimeIndex of all trading dates.

    Returns:
        Tuple of:
            - embeddings: (n_days, 768) array of daily embeddings
            - counts: (n_days,) array of article counts per day
    """
    n_days = len(trading_dates)
    embeddings = np.zeros((n_days, FINBERT_DIM), dtype=np.float32)
    counts = np.zeros(n_days, dtype=np.float32)

    # Parse dates in the encoded DataFrame
    encoded_df = encoded_df.copy()
    encoded_df["date"] = pd.to_datetime(encoded_df["date"])
    encoded_df["trade_date"] = encoded_df["date"].dt.normalize()

    # Build sorted trading dates for mapping non-trading-day articles
    sorted_trading = trading_dates.normalize().sort_values()
    date_to_idx = {d: i for i, d in enumerate(sorted_trading)}

    # Map each article's trade_date to the nearest trading day:
    # - If it falls on a trading day, use it directly
    # - If it falls on a weekend/holiday, assign to the NEXT trading day
    #   (standard in financial NLP: weekend news affects Monday's open)
    def map_to_trading_day(dt):
        if dt in date_to_idx:
            return dt
        # Find the next trading day via searchsorted
        pos = sorted_trading.searchsorted(dt)
        if pos < len(sorted_trading):
            return sorted_trading[pos]
        # Article is after our last trading day — cannot assign
        return None

    encoded_df["mapped_date"] = encoded_df["trade_date"].apply(map_to_trading_day)

    # Count how many were remapped vs dropped
    n_original = len(encoded_df)
    n_remapped = (encoded_df["mapped_date"] != encoded_df["trade_date"]).sum()
    encoded_df = encoded_df.dropna(subset=["mapped_date"])
    n_dropped = n_original - len(encoded_df)

    for mapped_date, group in encoded_df.groupby("mapped_date"):
        if mapped_date in date_to_idx:
            idx = date_to_idx[mapped_date]
            embs = np.stack(group["embedding"].values)
            confs = group["confidence"].values

            # Confidence-weighted average
            weights = confs / (confs.sum() + 1e-10)
            embeddings[idx] = (embs * weights[:, None]).sum(axis=0)
            counts[idx] = len(group)

    pct_with_news = 100 * (counts > 0).mean()
    remap_msg = f", {n_remapped} weekend/holiday→next trading day" if n_remapped > 0 else ""
    drop_msg = f", {n_dropped} dropped (after study period)" if n_dropped > 0 else ""
    logger.info(
        f"  Daily aggregation: {n_days} days, "
        f"{pct_with_news:.1f}% have news, "
        f"mean articles/day={counts[counts>0].mean():.1f}"
        f"{remap_msg}{drop_msg}"
    )

    return embeddings, counts


# ─────────────────────────────────────────────────────────────
# VADER FALLBACK (no GPU required)
# ─────────────────────────────────────────────────────────────

def compute_vader_sentiment(
    articles_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Fallback: Compute VADER sentiment scores (5 daily features).

    Used when FinBERT is unavailable. Produces 5 features per day:
    mean_sentiment, max_sentiment, min_sentiment, sentiment_std, article_count.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
    except ImportError:
        logger.warning("VADER not installed. Returning zeros.")
        return pd.DataFrame(
            0.0, index=trading_dates,
            columns=["mean_sentiment", "max_sentiment", "min_sentiment",
                     "sentiment_std", "article_count"],
        )

    title_col = FNSPID_CONFIG["title_col"]
    article_col = FNSPID_CONFIG["article_col"]
    date_col = FNSPID_CONFIG["date_col"]

    # Score each article
    scores = []
    for _, row in articles_df.iterrows():
        text = _prepare_text_for_finbert(row[title_col], row[article_col])
        score = analyzer.polarity_scores(text)["compound"]
        scores.append({
            "date": pd.to_datetime(row[date_col]).normalize(),
            "sentiment": score,
        })

    if not scores:
        return pd.DataFrame(
            0.0, index=trading_dates,
            columns=["mean_sentiment", "max_sentiment", "min_sentiment",
                     "sentiment_std", "article_count"],
        )

    scored_df = pd.DataFrame(scores)

    daily = scored_df.groupby("date").agg(
        mean_sentiment=("sentiment", "mean"),
        max_sentiment=("sentiment", "max"),
        min_sentiment=("sentiment", "min"),
        sentiment_std=("sentiment", "std"),
        article_count=("sentiment", "count"),
    )
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
    daily = daily.reindex(trading_dates, fill_value=0.0)

    return daily


# ─────────────────────────────────────────────────────────────
# SYNTHETIC SENTIMENT FALLBACK (no external data required)
# ─────────────────────────────────────────────────────────────

def generate_synthetic_sentiment(
    price_df: pd.DataFrame,
    n_days: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic sentiment from price momentum as last-resort fallback.

    Returns zero embeddings (768-dim) and zero counts, effectively
    disabling the sentiment modality. The model's "no news" embedding
    handles this case.

    ⚠ EXPLICITLY SYNTHETIC. Provenance system records this.
    """
    logger.warning("⚠ Using synthetic (zero) sentiment — sentiment modality DISABLED")
    embeddings = np.zeros((n_days, FINBERT_DIM), dtype=np.float32)
    counts = np.zeros(n_days, dtype=np.float32)
    return embeddings, counts


# ─────────────────────────────────────────────────────────────
# TOP-LEVEL ORCHESTRATOR WITH PROVENANCE
# ─────────────────────────────────────────────────────────────

@timer
def fetch_all_sentiment(
    fnspid_path: str = None,
    trading_dates_per_ticker: Dict[str, pd.DatetimeIndex] = None,
    price_data: Dict[str, pd.DataFrame] = None,
    tickers: list = None,
    use_finbert: bool = True,
    cache: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, str]]:
    """
    Fetch and process sentiment for all tickers with provenance tracking.

    PRIORITY ORDER:
      1. FNSPID articles + FinBERT encoding (real, 769 features)
      2. FNSPID articles + VADER scoring (real but lower quality, 5 features)
      3. Synthetic zero embeddings (sentiment modality disabled)

    PROVENANCE: Every ticker is registered with its actual data source.

    Args:
        fnspid_path: Path to FNSPID CSV (None = skip FNSPID, use synthetic)
        trading_dates_per_ticker: Dict of ticker → DatetimeIndex
        price_data: Dict of ticker → price DataFrame (for synthetic fallback)
        tickers: List of tickers to process
        use_finbert: If True, use FinBERT; if False, use VADER on FNSPID articles
        cache: Whether to use cached data

    Returns:
        Tuple of:
          - sent_embeddings: Dict of ticker → (n_days, emb_dim) array
          - sent_counts: Dict of ticker → (n_days,) array
          - source_types: Dict of ticker → "real_finbert" | "real_vader" | "synthetic"
    """
    if tickers is None:
        tickers = TICKERS

    sent_embeddings = {}
    sent_counts = {}
    source_types = {}

    # Step 1: Load FNSPID articles (from cache or fresh extraction)
    articles_by_ticker = {}
    fnspid_cache_dir = RAW_DIR / "fnspid"

    if fnspid_path is not None:
        # Explicit FNSPID path provided — use cache if available, else extract
        all_cached = all(
            (fnspid_cache_dir / f"articles_{t}.parquet").exists() for t in tickers
        )

        if cache and all_cached:
            logger.info("Loading cached FNSPID articles...")
            for ticker in tickers:
                cache_path = fnspid_cache_dir / f"articles_{ticker}.parquet"
                articles_by_ticker[ticker] = pd.read_parquet(cache_path)
                logger.info(f"  {ticker}: {len(articles_by_ticker[ticker]):,} articles (cached)")
        else:
            logger.info("Extracting articles from FNSPID...")
            articles_by_ticker = extract_fnspid_articles(fnspid_path)
    else:
        # No --fnspid flag, but check if article caches exist from a previous run
        all_cached = all(
            (fnspid_cache_dir / f"articles_{t}.parquet").exists() for t in tickers
        )
        if cache and all_cached:
            logger.info("Loading cached FNSPID articles (no --fnspid flag, using previous extraction)...")
            for ticker in tickers:
                cache_path = fnspid_cache_dir / f"articles_{ticker}.parquet"
                articles_by_ticker[ticker] = pd.read_parquet(cache_path)
                logger.info(f"  {ticker}: {len(articles_by_ticker[ticker]):,} articles (cached)")
        else:
            logger.warning("⚠ No FNSPID path provided and no article caches found — "
                          "sentiment will use FinBERT caches if available, else SYNTHETIC")

    # Step 1b: Cross-company article enrichment + deduplication
    # FNSPID tags articles to only one ticker, but many mention multiple companies.
    # This step shares articles across tickers based on title/body keyword matching.
    # Results are saved back to cache so enrichment only runs ONCE.
    if articles_by_ticker:
        # Check if articles are already enriched (have _cross_matched column)
        sample_ticker = next(iter(articles_by_ticker))
        already_enriched = "_cross_matched" in articles_by_ticker[sample_ticker].columns

        if already_enriched:
            logger.info("Articles already enriched (loaded from cache) — skipping re-enrichment")
        else:
            from features.article_enrichment import enrich_articles_cross_company
            articles_by_ticker = enrich_articles_cross_company(articles_by_ticker)

            # Save enriched articles back to cache so future runs skip enrichment
            if cache:
                fnspid_cache_dir = RAW_DIR / "fnspid"
                fnspid_cache_dir.mkdir(parents=True, exist_ok=True)
                for ticker, df in articles_by_ticker.items():
                    cache_path = fnspid_cache_dir / f"articles_{ticker}.parquet"
                    df.to_parquet(cache_path)
                logger.info(f"  Enriched articles saved to cache ({len(articles_by_ticker)} tickers)")

    # Step 2: Process each ticker
    for ticker in tickers:
        if trading_dates_per_ticker and ticker in trading_dates_per_ticker:
            trading_dates = trading_dates_per_ticker[ticker]
        else:
            trading_dates = pd.bdate_range(DATA_START_DATE, DATA_END_DATE)
        n_days = len(trading_dates)

        # Check for cached FinBERT embeddings
        finbert_cache = PROCESSED_DIR / f"sentiment_finbert_{ticker}.npz"
        if cache and finbert_cache.exists():
            cached = np.load(finbert_cache)

            # Use n_articles_input if available (new format), fall back to counts.sum() (old format)
            if "n_articles_input" in cached:
                cached_article_count = int(cached["n_articles_input"])
            else:
                cached_article_count = int(cached["counts"].sum())

            # Staleness check: if article count changed (e.g., enrichment added articles),
            # the cache is stale and must be re-encoded.
            current_article_count = len(articles_by_ticker.get(ticker, []))
            if current_article_count > 0 and cached_article_count != current_article_count:
                logger.warning(
                    f"  {ticker}: FinBERT cache STALE — "
                    f"cache has {cached_article_count:,} articles, "
                    f"current articles: {current_article_count:,}. Re-encoding..."
                )
                # Fall through to re-encode below
            else:
                sent_embeddings[ticker] = cached["embeddings"]
                sent_counts[ticker] = cached["counts"]
                source_types[ticker] = "real_finbert"
                provenance.register(
                    ticker, "sentiment", "real",
                    f"FNSPID + FinBERT ({FINBERT_DIM}-dim, cached, {cached_article_count:,} articles)",
                    n_features=FINBERT_DIM + 1,
                )
                logger.info(f"  {ticker}: FinBERT embeddings loaded from cache ({cached_article_count:,} articles)")
                continue

        # Has FNSPID articles?
        if ticker in articles_by_ticker and len(articles_by_ticker[ticker]) > 0:
            articles = articles_by_ticker[ticker]
            n_articles = len(articles)

            if use_finbert:
                # Full FinBERT encoding
                try:
                    encoded = encode_articles_finbert(articles, ticker)
                    emb, cnt = aggregate_daily_sentiment(encoded, trading_dates)
                    sent_embeddings[ticker] = emb
                    sent_counts[ticker] = cnt
                    source_types[ticker] = "real_finbert"
                    provenance.register(
                        ticker, "sentiment", "real",
                        f"FNSPID ({n_articles:,} articles) + FinBERT ({FINBERT_DIM}-dim)",
                        n_features=FINBERT_DIM + 1,
                    )
                    # Cache for next time — include n_articles_input so
                    # staleness check can compare apples-to-apples
                    if cache:
                        np.savez(finbert_cache, embeddings=emb, counts=cnt,
                                 n_articles_input=np.array(n_articles))
                    continue
                except Exception as e:
                    logger.warning(f"  {ticker}: FinBERT encoding failed ({e}) — trying VADER")

            # VADER fallback (still using real FNSPID articles)
            try:
                vader_df = compute_vader_sentiment(articles, trading_dates)
                # VADER produces 5 features, not 768-dim embeddings
                sent_embeddings[ticker] = vader_df.values.astype(np.float32)
                sent_counts[ticker] = vader_df["article_count"].values.astype(np.float32)
                source_types[ticker] = "real_vader"
                provenance.register(
                    ticker, "sentiment", "real",
                    f"FNSPID ({n_articles:,} articles) + VADER (5 features) — "
                    f"lower quality than FinBERT",
                    n_features=5,
                )
                continue
            except Exception as e:
                logger.warning(f"  {ticker}: VADER also failed ({e}) — using synthetic")

        # Synthetic fallback (no articles available)
        emb, cnt = generate_synthetic_sentiment(
            price_data.get(ticker) if price_data else None, n_days,
        )
        sent_embeddings[ticker] = emb
        sent_counts[ticker] = cnt
        source_types[ticker] = "synthetic"
        provenance.register(
            ticker, "sentiment", "synthetic",
            f"⚠ Zero embeddings — sentiment modality DISABLED for this ticker. "
            f"No FNSPID articles found or FNSPID path not provided.",
            n_features=FINBERT_DIM + 1,
        )

    # Summary
    n_finbert = sum(1 for v in source_types.values() if v == "real_finbert")
    n_vader = sum(1 for v in source_types.values() if v == "real_vader")
    n_synth = sum(1 for v in source_types.values() if v == "synthetic")

    logger.info(
        f"Sentiment complete: {n_finbert} FinBERT, {n_vader} VADER, {n_synth} synthetic"
    )

    if n_synth > 0:
        synth_list = [t for t, s in source_types.items() if s == "synthetic"]
        logger.warning(
            f"⚠ SYNTHETIC SENTIMENT for: {', '.join(synth_list)}\n"
            f"  Sentiment modality is DISABLED for these tickers (zero embeddings).\n"
            f"  Modality ablation results for sentiment should note this."
        )

    return sent_embeddings, sent_counts, source_types


if __name__ == "__main__":
    # Test the text preparation function
    print("=== Text preparation test ===")
    title = "Interesting AAPL Put And Call Options For August 2024"
    article = "Investors in Apple Inc saw new options begin trading this week. " * 20
    text = _prepare_text_for_finbert(title, article)
    print(f"Title length: {len(title)}")
    print(f"Article length: {len(article)}")
    print(f"Combined length: {len(text)}")
    print(f"Combined preview: {text[:100]}...")
