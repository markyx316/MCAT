"""
features/article_enrichment.py — Cross-Company Article Matching.
==================================================================
FNSPID tags each article to only one stock symbol, but many articles
discuss multiple companies. For example, "Apple And Amazon Earnings"
is tagged only as AAPL, but is relevant to AMZN too.

This module scans article titles (and bodies when available) for
mentions of ALL our tickers' company names, then shares articles
across tickers. This dramatically improves sentiment coverage
for under-represented tickers.

From our analysis:
  - AAPL articles mention Amazon in 4.3% of titles
  - AAPL articles mention Google in 3.8% of titles
  - AAPL articles mention Goldman in 8.3% of titles (GS ticker match)

This enrichment can increase article counts by 20-50% for many tickers
and fill coverage gaps where FNSPID has limited direct tagging.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TICKERS, RAW_DIR
from utils import setup_logger

logger = setup_logger(__name__)

# ─────────────────────────────────────────────────────────────
# COMPANY NAME KEYWORDS FOR MATCHING
# ─────────────────────────────────────────────────────────────
# Each ticker maps to a list of keywords that identify the company.
# Order matters: more specific patterns first to avoid false positives.
# We use word-boundary matching to avoid partial matches
# (e.g., "CAT" matching "category").

COMPANY_KEYWORDS = {
    "AAPL": [r"\bApple\b", r"\bAAPL\b", r"\biPhone\b", r"\biPad\b", r"\bMacBook\b", r"\bApple Inc\b"],
    "MSFT": [r"\bMicrosoft\b", r"\bMSFT\b", r"\bAzure\b", r"\bWindows\b"],
    "GOOGL": [r"\bGoogle\b", r"\bAlphabet\b", r"\bGOOGL\b", r"\bGOOG\b", r"\bYouTube\b"],
    "AMZN": [r"\bAmazon\b", r"\bAMZN\b", r"\bAWS\b"],
    "META": [r"\bMeta Platforms\b", r"\bFacebook\b", r"\bMETA\b", r"\bInstagram\b", r"\bWhatsApp\b"],
    "NVDA": [r"\bNvidia\b", r"\bNVDA\b", r"\bGeForce\b", r"\bCUDA\b"],
    "JPM": [r"\bJPMorgan\b", r"\bJP Morgan\b", r"\bJPM\b", r"\bChase\b"],
    "GS":  [r"\bGoldman Sachs\b", r"\bGoldman\b"],  # Note: "GS" alone is too ambiguous
    "JNJ": [r"\bJohnson & Johnson\b", r"\bJohnson and Johnson\b", r"\bJNJ\b", r"\bJ&J\b"],
    "UNH": [r"\bUnitedHealth\b", r"\bUNH\b", r"\bUnited Health\b", r"\bOptum\b"],
    "WMT": [r"\bWalmart\b", r"\bWMT\b", r"\bWal-Mart\b"],
    "PG":  [r"\bProcter & Gamble\b", r"\bProcter and Gamble\b", r"\bP&G\b"],  # "PG" too ambiguous
    "XOM": [r"\bExxon\b", r"\bXOM\b", r"\bExxonMobil\b"],
    "CAT": [r"\bCaterpillar\b"],  # "CAT" alone matches too many words
    "DIS": [r"\bDisney\b", r"\bDIS\b", r"\bWalt Disney\b"],
}

# ─────────────────────────────────────────────────────────────
# SECTOR/GROUP KEYWORDS — map group terms to multiple tickers
# ─────────────────────────────────────────────────────────────
# Articles mentioning "Big Tech" or "FAANG" are relevant to all constituent stocks.
# From our data: 121 "Big Tech" + 81 "FAANG" + 40 "Magnificent Seven" articles.
GROUP_KEYWORDS = {
    r"\bBig Tech\b": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    r"\bFAANG\b": ["META", "AAPL", "AMZN", "NVDA", "GOOGL"],
    r"\bMagnificent Seven\b": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    r"\bMag 7\b": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    r"\bMega.?cap tech\b": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    r"\bWall Street banks?\b": ["JPM", "GS"],
    r"\bBig Banks?\b": ["JPM", "GS"],
}

# Pre-compile regex patterns for speed
_COMPILED_PATTERNS = {
    ticker: [re.compile(pat, re.IGNORECASE) for pat in patterns]
    for ticker, patterns in COMPANY_KEYWORDS.items()
}

_COMPILED_GROUP_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), tickers)
    for pat, tickers in GROUP_KEYWORDS.items()
]


def find_mentioned_tickers(text: str) -> List[str]:
    """
    Find all our tickers mentioned in a text string.
    Uses two-pass matching:
      1. Company-level: regex word-boundary matching per ticker
      2. Group-level: sector terms like "Big Tech", "FAANG" → all constituents

    Returns:
        List of unique ticker symbols found in the text.
    """
    if not text or not isinstance(text, str):
        return []

    mentioned = set()

    # Pass 1: Individual company keywords
    for ticker, patterns in _COMPILED_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                mentioned.add(ticker)
                break  # One match is enough for this ticker

    # Pass 2: Group/sector keywords (e.g., "Big Tech" → AAPL, MSFT, GOOGL, AMZN, META)
    for pattern, tickers in _COMPILED_GROUP_PATTERNS:
        if pattern.search(text):
            mentioned.update(tickers)

    return list(mentioned)


def enrich_articles_cross_company(
    articles_per_ticker: Dict[str, pd.DataFrame],
    use_body: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Enrich article collections by sharing articles across tickers.

    For each article, scans the title (and body if available) for mentions
    of other companies. If found, a copy of the article is added to that
    company's article list.

    Args:
        articles_per_ticker: Dict of ticker → DataFrame with Article_title, Article columns.
        use_body: If True, also scan article body (not just title). Slower but more thorough.

    Returns:
        Enriched dict of ticker → DataFrame (original articles + shared articles).
    """
    logger.info("Cross-company article enrichment...")

    # Collect all articles from all tickers into a single pool
    all_articles = []
    for ticker, df in articles_per_ticker.items():
        if df is not None and len(df) > 0:
            df_copy = df.copy()
            df_copy["_source_ticker"] = ticker
            all_articles.append(df_copy)

    if not all_articles:
        logger.warning("  No articles to enrich")
        return articles_per_ticker

    pool = pd.concat(all_articles, ignore_index=True)
    logger.info(f"  Total article pool: {len(pool):,} articles from {len(articles_per_ticker)} tickers")

    # For each article, find ALL mentioned tickers
    new_assignments = {ticker: [] for ticker in TICKERS}
    n_new = 0

    for idx, row in pool.iterrows():
        source_ticker = row["_source_ticker"]

        # Build text to scan: title + body (if available)
        title = str(row.get("Article_title", "")) if pd.notna(row.get("Article_title")) else ""
        body = ""
        if use_body and "Article" in row.index and pd.notna(row.get("Article")):
            # Only scan first 500 chars of body for speed
            body = str(row["Article"])[:500]

        text = f"{title} {body}"

        # Find all mentioned tickers
        mentioned = find_mentioned_tickers(text)

        # Share article with tickers that are mentioned but not the source
        for target_ticker in mentioned:
            if target_ticker != source_ticker:
                new_assignments[target_ticker].append(idx)
                n_new += 1

        if idx % 10000 == 0 and idx > 0:
            logger.info(f"    Scanned {idx:,}/{len(pool):,} articles, {n_new:,} new assignments")

    # Build enriched article sets with DEDUPLICATION
    enriched = {}
    n_total_dupes = 0

    for ticker in articles_per_ticker:
        original = articles_per_ticker[ticker]
        n_original = len(original) if original is not None else 0

        # Get newly matched articles
        new_indices = new_assignments.get(ticker, [])
        if new_indices:
            new_articles = pool.loc[new_indices].drop(columns=["_source_ticker"], errors="ignore")
            # Mark as cross-matched (for transparency)
            new_articles = new_articles.copy()
            new_articles["_cross_matched"] = True

            if original is not None and len(original) > 0:
                original_copy = original.copy()
                original_copy["_cross_matched"] = False
                combined = pd.concat([original_copy, new_articles], ignore_index=True)
            else:
                combined = new_articles
        else:
            if original is not None:
                combined = original.copy()
                combined["_cross_matched"] = False
            else:
                combined = pd.DataFrame()

        # ─── DEDUPLICATION ───
        # An article with the same title + date may appear multiple times:
        #   (a) already in original list AND matched by cross-company scan
        #   (b) matched from two different source tickers with same article
        # Deduplicate on (Article_title, Date), keeping the original over cross-matched.
        n_before_dedup = len(combined)
        if n_before_dedup > 0 and "Article_title" in combined.columns and "Date" in combined.columns:
            # Sort so originals (False) come before cross-matched (True)
            combined = combined.sort_values("_cross_matched", ascending=True)
            combined = combined.drop_duplicates(subset=["Article_title", "Date"], keep="first")
            combined = combined.reset_index(drop=True)

        n_dupes = n_before_dedup - len(combined)
        n_total_dupes += n_dupes

        enriched[ticker] = combined

        n_enriched = len(enriched[ticker])
        n_added = n_enriched - n_original
        if n_added > 0:
            dedup_msg = f", {n_dupes} duplicates removed" if n_dupes > 0 else ""
            logger.info(f"  {ticker}: {n_original:,} → {n_enriched:,} articles (+{n_added:,} cross-matched{dedup_msg})")
        else:
            logger.info(f"  {ticker}: {n_original:,} articles (no cross-matches)")

    total_original = sum(len(v) for v in articles_per_ticker.values() if v is not None)
    total_enriched = sum(len(v) for v in enriched.values())
    logger.info(f"  Enrichment complete: {total_original:,} → {total_enriched:,} total articles "
                f"(+{total_enriched - total_original:,}, {100*(total_enriched - total_original)/max(total_original,1):.1f}% increase, "
                f"{n_total_dupes:,} duplicates removed)")

    return enriched


if __name__ == "__main__":
    # Quick test with uploaded sample articles
    import os

    upload_dir = Path("/mnt/user-data/uploads")
    articles = {}
    for f in upload_dir.glob("articles_*.parquet"):
        ticker = f.stem.replace("articles_", "")
        if ticker in TICKERS:
            articles[ticker] = pd.read_parquet(f)
            print(f"  Loaded {ticker}: {len(articles[ticker])} articles")

    if articles:
        enriched = enrich_articles_cross_company(articles)
        print("\n  BEFORE vs AFTER:")
        for ticker in sorted(set(list(articles.keys()) + list(enriched.keys()))):
            before = len(articles.get(ticker, pd.DataFrame()))
            after = len(enriched.get(ticker, pd.DataFrame()))
            diff = after - before
            print(f"    {ticker}: {before:>6,} → {after:>6,}  (+{diff:,})")
