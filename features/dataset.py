"""
features/dataset.py — PyTorch Dataset for Multi-Modal Financial Data.
=====================================================================
Combines all 4 modalities (price, sentiment, fundamentals, macro) into
windowed samples ready for the Transformer model.

Each sample contains:
  - price_features:  (45, 39) — denoised, z-normalized price/technical features
  - sent_features:   (45, 769) — FinBERT embeddings + article count
  - fund_features:   (d_f,) — static fundamental vector
  - macro_features:  (45, d_m) — z-normalized macro features
  - label:           scalar — 3-day forward return in percentage points
  - stock_id:        int — stock index for [STOCK] embedding
  - date:            datetime — for walk-forward split filtering
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LOOKBACK_WINDOW, TICKERS, STOCK_TO_ID, FINBERT_DIM,
    DATA_START_DATE, DATA_END_DATE,
)
from features.normalize import normalize_window_zscore
from features.denoise import denoise_feature_matrix
from utils import setup_logger, timer

logger = setup_logger(__name__)


@dataclass
class Sample:
    """A single multi-modal sample."""
    price: np.ndarray       # (lookback, n_price_features)
    sentiment: np.ndarray   # (lookback, 769)
    fundamentals: np.ndarray  # (n_fund_features,)
    macro: np.ndarray       # (lookback, n_macro_features)
    label: float            # 3-day forward return in pp
    stock_id: int           # Index for stock embedding
    date: np.datetime64     # Date of the sample


class MultiModalDataset:
    """
    Dataset that creates windowed multi-modal samples for all tickers.

    Preprocessing per sample:
      1. Extract 45-day lookback windows from each modality
      2. Optionally denoise price features (wavelet DWT)
      3. Z-normalize temporal windows (per-window, per-feature)
      4. Pack into aligned arrays

    Samples are stored as pre-computed numpy arrays for fast iteration.
    """

    def __init__(
        self,
        price_features: Dict[str, pd.DataFrame],
        labels: Dict[str, pd.Series],
        sentiment_embeddings: Dict[str, np.ndarray] = None,
        sentiment_counts: Dict[str, np.ndarray] = None,
        fund_features: Dict[str, pd.DataFrame] = None,
        macro_features: pd.DataFrame = None,
        lookback: int = LOOKBACK_WINDOW,
        denoise: bool = True,
        tickers: list = None,
    ):
        """
        Build the dataset by extracting windowed samples from all tickers.

        Args:
            price_features: Dict of ticker → DataFrame (date-indexed, 39 features).
            labels: Dict of ticker → Series of forward returns.
            sentiment_embeddings: Dict of ticker → (n_days, 768) array.
            sentiment_counts: Dict of ticker → (n_days,) array.
            fund_features: Dict of ticker → DataFrame of fundamental features.
            macro_features: Single DataFrame of macro features (shared across stocks).
            lookback: Lookback window size.
            denoise: Whether to apply wavelet denoising.
            tickers: List of tickers to include (default: all).
        """
        if tickers is None:
            tickers = TICKERS

        self.lookback = lookback
        self.denoise = denoise

        # Build all samples
        (self.X_price, self.X_sent, self.X_fund, self.X_macro,
         self.y, self.stock_ids, self.dates) = self._build_samples(
            price_features, labels, sentiment_embeddings, sentiment_counts,
            fund_features, macro_features, tickers,
        )

        logger.info(
            f"Dataset built: {len(self)} samples | "
            f"Price: {self.X_price.shape} | Sent: {self.X_sent.shape} | "
            f"Fund: {self.X_fund.shape} | Macro: {self.X_macro.shape} | "
            f"Labels: mean={self.y.mean():.3f}pp, std={self.y.std():.3f}pp"
        )

    def _build_samples(
        self,
        price_features, labels, sentiment_embeddings, sentiment_counts,
        fund_features, macro_features, tickers,
    ) -> Tuple:
        """Extract all windowed samples across all tickers."""

        all_price = []
        all_sent = []
        all_fund = []
        all_macro = []
        all_labels = []
        all_stock_ids = []
        all_dates = []

        # ── Determine max fundamental dimension across all tickers ──
        # Real Alpha Vantage data has 10 features, synthetic has 7.
        # We must pad all to the same size for numpy stacking.
        max_fund_dim = 1  # Minimum fallback
        if fund_features is not None:
            for ticker in tickers:
                if ticker in fund_features:
                    n_cols = fund_features[ticker].shape[1]
                    max_fund_dim = max(max_fund_dim, n_cols)
        logger.info(f"  Fundamental dimension: {max_fund_dim} (max across all tickers)")

        for ticker in tickers:
            if ticker not in price_features or ticker not in labels:
                logger.warning(f"  {ticker}: Missing data — skipped")
                continue

            price_df = price_features[ticker]
            label_series = labels[ticker]

            # Align all modalities to the same date index
            common_dates = price_df.index.intersection(label_series.dropna().index)

            if macro_features is not None:
                common_dates = common_dates.intersection(macro_features.index)

            common_dates = common_dates.sort_values()

            if len(common_dates) < self.lookback + 1:
                logger.warning(f"  {ticker}: Only {len(common_dates)} aligned dates — skipped")
                continue

            # Convert to numpy for speed
            price_vals = price_df.reindex(common_dates).values.astype(np.float32)
            price_vals = np.nan_to_num(price_vals, nan=0.0, posinf=0.0, neginf=0.0)
            label_vals = label_series.reindex(common_dates).values.astype(np.float32)

            # Sentiment: align to common dates
            if sentiment_embeddings is not None and ticker in sentiment_embeddings:
                sent_emb = sentiment_embeddings[ticker]  # (n_days, emb_dim)
                sent_cnt = sentiment_counts[ticker]      # (n_days,)
                sent_emb_dim = sent_emb.shape[1]
            else:
                sent_emb_dim = FINBERT_DIM
                sent_emb = np.zeros((len(common_dates), sent_emb_dim), dtype=np.float32)
                sent_cnt = np.zeros(len(common_dates), dtype=np.float32)

            # Ensure sentiment is aligned (may need reindexing)
            if len(sent_emb) != len(common_dates):
                # Sentiment was built on full trading dates; reindex
                sent_emb_padded = np.zeros((len(common_dates), sent_emb_dim), dtype=np.float32)
                sent_cnt_padded = np.zeros(len(common_dates), dtype=np.float32)
                # Best-effort alignment by position
                n_copy = min(len(sent_emb), len(common_dates))
                sent_emb_padded[:n_copy] = sent_emb[:n_copy]
                sent_cnt_padded[:n_copy] = sent_cnt[:n_copy]
                sent_emb = sent_emb_padded
                sent_cnt = sent_cnt_padded

            # Fundamentals — pad to uniform max_fund_dim
            if fund_features is not None and ticker in fund_features:
                fund_df = fund_features[ticker].reindex(common_dates, method="ffill")
                fund_df = fund_df.ffill().bfill().fillna(0)  # Ensure no NaN
                fund_raw = fund_df.values.astype(np.float32)
                # Pad if this ticker has fewer features than max_fund_dim
                if fund_raw.shape[1] < max_fund_dim:
                    pad_width = max_fund_dim - fund_raw.shape[1]
                    fund_vals = np.pad(fund_raw, ((0, 0), (0, pad_width)),
                                       mode='constant', constant_values=0.0)
                else:
                    fund_vals = fund_raw
            else:
                fund_vals = np.zeros((len(common_dates), max_fund_dim), dtype=np.float32)
            # Final safety: replace any remaining NaN/inf
            fund_vals = np.nan_to_num(fund_vals, nan=0.0, posinf=0.0, neginf=0.0)

            # Macro
            if macro_features is not None:
                macro_reindexed = macro_features.reindex(common_dates, method="ffill")
                macro_reindexed = macro_reindexed.ffill().bfill().fillna(0)
                macro_vals = macro_reindexed.values.astype(np.float32)
            else:
                macro_vals = np.zeros((len(common_dates), 1), dtype=np.float32)
            macro_vals = np.nan_to_num(macro_vals, nan=0.0, posinf=0.0, neginf=0.0)

            stock_id = STOCK_TO_ID.get(ticker, 0)

            # Extract windows
            n_samples = len(common_dates) - self.lookback
            for i in range(self.lookback, len(common_dates)):
                # Check for valid label
                if np.isnan(label_vals[i]):
                    continue

                # Extract lookback windows
                price_window = price_vals[i - self.lookback : i]     # (60, 39)
                sent_window_emb = sent_emb[i - self.lookback : i]    # (60, 768)
                sent_window_cnt = sent_cnt[i - self.lookback : i]    # (60,)
                macro_window = macro_vals[i - self.lookback : i]     # (60, d_m)
                fund_snapshot = fund_vals[i]                         # (d_f,)

                # ─── Denoise price features ───
                if self.denoise:
                    price_window = denoise_feature_matrix(price_window)

                # ─── Normalize temporal windows ───
                price_window = normalize_window_zscore(price_window)
                macro_window = normalize_window_zscore(macro_window)
                # Sentiment embeddings: already normalized by FinBERT

                # ─── Combine sentiment: embeddings + count ───
                sent_combined = np.concatenate([
                    sent_window_emb,
                    sent_window_cnt[:, None],  # (60, 1)
                ], axis=1)  # (60, 769)

                # Accumulate
                all_price.append(price_window)
                all_sent.append(sent_combined)
                all_fund.append(fund_snapshot)
                all_macro.append(macro_window)
                all_labels.append(label_vals[i])
                all_stock_ids.append(stock_id)
                all_dates.append(common_dates[i])

            logger.info(f"  {ticker}: {len(all_labels) - sum(1 for _ in all_dates if _ < common_dates[0])} → {len([d for d in all_dates if d >= common_dates[self.lookback]])} valid samples")

        if not all_labels:
            raise ValueError("No valid samples created! Check data alignment.")

        return (
            np.array(all_price, dtype=np.float32),
            np.array(all_sent, dtype=np.float32),
            np.array(all_fund, dtype=np.float32),
            np.array(all_macro, dtype=np.float32),
            np.array(all_labels, dtype=np.float32),
            np.array(all_stock_ids, dtype=np.int64),
            np.array(all_dates),
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        """Return a single sample as dict of numpy arrays."""
        return {
            "price": self.X_price[idx],
            "sentiment": self.X_sent[idx],
            "fundamentals": self.X_fund[idx],
            "macro": self.X_macro[idx],
            "label": self.y[idx],
            "stock_id": self.stock_ids[idx],
            "date": self.dates[idx],
        }

    def get_subset_by_dates(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> "MultiModalSubset":
        """
        Return a subset of samples within a date range.
        Used for walk-forward train/val/test splitting.
        """
        dates_ts = pd.to_datetime(self.dates)
        mask = (dates_ts >= start_date) & (dates_ts < end_date)
        indices = np.where(mask)[0]
        return MultiModalSubset(self, indices)


class MultiModalSubset:
    """
    A subset of MultiModalDataset selected by indices.
    Implements __len__ and __getitem__ for PyTorch DataLoader.
    """

    def __init__(self, dataset: MultiModalDataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.dataset[real_idx]


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Converts list of sample dicts → dict of batched tensors.
    """
    import torch
    return {
        "price": torch.tensor(np.stack([s["price"] for s in batch]), dtype=torch.float32),
        "sentiment": torch.tensor(np.stack([s["sentiment"] for s in batch]), dtype=torch.float32),
        "fundamentals": torch.tensor(np.stack([s["fundamentals"] for s in batch]), dtype=torch.float32),
        "macro": torch.tensor(np.stack([s["macro"] for s in batch]), dtype=torch.float32),
        "label": torch.tensor(np.array([s["label"] for s in batch]), dtype=torch.float32),
        "stock_id": torch.tensor(np.array([s["stock_id"] for s in batch]), dtype=torch.long),
    }


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    dates = pd.bdate_range("2017-01-01", "2023-12-31")
    n = len(dates)

    # Synthetic price features (39 features)
    price_feats = {
        "AAPL": pd.DataFrame(
            np.random.randn(n, 39).astype(np.float32),
            index=dates, columns=[f"f{i}" for i in range(39)]
        ),
    }
    # Synthetic labels
    labels = {
        "AAPL": pd.Series(np.random.randn(n).astype(np.float32) * 3, index=dates, name="label"),
    }

    # Build dataset
    ds = MultiModalDataset(
        price_features=price_feats,
        labels=labels,
        denoise=False,  # Skip for speed in test
        tickers=["AAPL"],
    )

    print(f"\nDataset size: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

    # Test date subsetting
    sub = ds.get_subset_by_dates(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-07-01"))
    print(f"\nSubset 2020-H1: {len(sub)} samples")
