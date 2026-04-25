"""
config.py — Central Configuration for Multi-Modal Cross-Attention Transformer
==============================================================================
All hyperparameters, paths, and settings in one place.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR / "tables", RESULTS_DIR / "figures"]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 2. STOCK UNIVERSE
# ─────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",  # Technology
    "JPM", "GS",                                         # Financials
    "JNJ", "UNH",                                        # Healthcare
    "WMT", "PG",                                         # Consumer Staples
    "XOM",                                               # Energy
    "CAT",                                               # Industrials
    "DIS",                                               # Communication Services
]

# Mapping for FNSPID filtering (some tickers may appear differently)
TICKER_ALIASES = {
    "GOOGL": ["GOOGL", "GOOG"],
    "META": ["META", "FB"],  # Meta was Facebook before 2021
}

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Technology", "META": "Technology", "NVDA": "Technology",
    "JPM": "Financials", "GS": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare",
    "WMT": "Consumer Staples", "PG": "Consumer Staples",
    "XOM": "Energy", "CAT": "Industrials", "DIS": "Communication Services",
}

N_STOCKS = len(TICKERS)
STOCK_TO_ID = {t: i for i, t in enumerate(TICKERS)}

# ─────────────────────────────────────────────────────────────
# 3. DATE RANGES
# ─────────────────────────────────────────────────────────────
# Fetch more data before 2017 for indicator warmup
DATA_FETCH_START = "2015-01-01"
DATA_START_DATE = "2017-01-01"    # Actual start after warmup
DATA_END_DATE = "2023-12-31"

# ─────────────────────────────────────────────────────────────
# 3b. API KEYS (from environment variables or hardcoded)
# ─────────────────────────────────────────────────────────────
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "YOUR_AV_KEY_HERE")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# Alpha Vantage rate limits
AV_RATE_LIMIT_PER_MINUTE = 5     # Free tier: 5 calls/min
AV_RATE_LIMIT_PER_DAY = 25       # Free tier: 25 calls/day
AV_SLEEP_BETWEEN_CALLS = 13.0    # Seconds between calls (safe for 5/min)

# ─────────────────────────────────────────────────────────────
# 4. WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────

# --- Full rolling validation ---
INITIAL_TRAIN_YEARS = 3           # First fold trains on 3 years
VALIDATION_MONTHS = 6
TEST_MONTHS = 3
WALK_FORWARD_STEP_MONTHS = 3
EMBARGO_DAYS = 5                  # Gap between splits to prevent label overlap

# --- Focused 3-fold validation (default for experiments) ---
# Rationale: With only 3 folds, we shrink val (3 months — still plenty
# for early stopping at epoch ~3) and expand test (6 months — 2× more
# statistical power per fold). Test windows tile across Jul 2022 → Dec 2023,
# covering bear recovery, AI rally, and consolidation.
FOCUSED_FOLDS = [
    # (train_end, val_end, test_end) — train always starts from DATA_START_DATE
    # Fold 0: Train 5.3yr | Val 3mo | Test 6mo (bear recovery → early 2023)
    ("2022-04-01", "2022-07-01", "2023-01-01"),
    # Fold 1: Train 5.8yr | Val 3mo | Test 6mo (AI rally → summer 2023)
    ("2022-10-01", "2023-01-01", "2023-07-01"),
    # Fold 2: Train 6.3yr | Val 3mo | Test 6mo (consolidation → year-end 2023)
    ("2023-04-01", "2023-07-01", "2023-12-31"),
]

# ─────────────────────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
LOOKBACK_WINDOW = 45              # 45 trading days

# Technical indicator parameters
TECHNICAL_CONFIG = {
    "sma_periods": [5, 10, 20, 50],
    "ema_periods": [12, 26],
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_period": 20,
    "bollinger_std": 2,
    "atr_period": 14,
    "stoch_period": 14,
    "stoch_smooth": 3,
    "williams_period": 14,
    "cci_period": 20,
    "roc_period": 10,
    "adx_period": 14,
    "obv": True,
    "cmf_period": 20,
    "volume_sma_period": 20,
}

# Fourier trend components (new, inspired by Li & Xu 2025)
FOURIER_COMPONENTS = [3, 9]       # Low-frequency reconstructions

N_PRICE_FEATURES = 39             # 37 original + 2 Fourier

# Sentiment features
SENTIMENT_FEATURES = [
    "finbert_embedding",           # 768-dim FinBERT [CLS] embedding
    "article_count",               # Scalar: number of articles that day
]
FINBERT_DIM = 768
N_SENTIMENT_FEATURES = FINBERT_DIM + 1  # 769

# FinBERT text input: combine title + first N chars of article body
FINBERT_ARTICLE_MAX_CHARS = 500   # First 500 chars of article body
FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 64           # Batch size for FinBERT inference

# Fundamental features
FUNDAMENTAL_FEATURES_REAL = [
    # From EARNINGS endpoint (quarterly, time-varying)
    "reportedEPS", "estimatedEPS", "surprisePercentage",
    # From INCOME_STATEMENT endpoint (quarterly, time-varying)
    "totalRevenue", "netIncome", "grossProfit",
    # From BALANCE_SHEET endpoint (quarterly, time-varying)
    "totalShareholderEquity", "longTermDebt", "shortTermDebt", "totalAssets",
    # NOTE: OVERVIEW endpoint (PERatio, ProfitMargin, BookValue, etc.) is NOT called.
    # It returns only today's static snapshot — using it would be look-ahead bias.
    # Instead, fundamental_preprocessor.py computes time-varying ratios from the
    # above raw fields (e.g., profit_margin = netIncome / totalRevenue per quarter).
]
FUNDAMENTAL_FEATURES_SYNTHETIC = [
    "pe_proxy",          # price / SMA_200
    "volatility_60d",    # 60-day realized volatility
    "volume_trend_20d",  # 20-day volume slope
    "price_sma50_ratio", # price / SMA_50
    "momentum_90d",      # 90-day return
    "golden_cross",      # SMA_50 / SMA_200
    "log_mcap_proxy",    # log(price * avg_volume)
]
N_FUND_FEATURES_REAL = len(FUNDAMENTAL_FEATURES_REAL)
N_FUND_FEATURES_SYNTHETIC = len(FUNDAMENTAL_FEATURES_SYNTHETIC)

# Macro features
MACRO_YFINANCE = {
    "VIX": "^VIX",
    "TNX": "^TNX",        # 10Y Treasury yield
    "IRX": "^IRX",        # 13-week bill rate
    "DXY": "DX-Y.NYB",   # US Dollar Index
}
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLP", "XLE", "XLI", "XLC", "XLB", "XLRE", "XLU", "XLY"]
MACRO_FRED = {
    "FEDFUNDS": "Federal Funds Rate",
    "T10Y2Y": "10Y-2Y Yield Spread",
    "UNRATE": "Unemployment Rate",
}

# ─────────────────────────────────────────────────────────────
# 6. LABEL CONSTRUCTION
# ─────────────────────────────────────────────────────────────
FORWARD_HORIZON = 3               # 3-day forward return
RETURN_SCALE = 100                # Multiply returns by 100 → percentage points

# ─────────────────────────────────────────────────────────────
# 7. DENOISING
# ─────────────────────────────────────────────────────────────
WAVELET_CONFIG = {
    "wavelet": "db4",             # Daubechies-4
    "level": 3,                   # Decomposition level
    "mode": "soft",               # Soft thresholding
}

# ─────────────────────────────────────────────────────────────
# 8. MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "d_model": 64,                # Shared embedding dimension
    "n_heads": 4,                 # Attention heads (d_k = d_model / n_heads = 16)
    "d_ff": 128,                  # Feed-forward hidden dimension
    "dropout": 0.02885133826053316,

    # Per-modality encoder depth
    "price_encoder_layers": 2,
    "sentiment_encoder_layers": 1,
    "macro_encoder_layers": 1,

    # Cross-attention layers
    "cross_attention_layers": 2,  # Price←Sent, then Price←Macro

    # Causal convolution preprocessing (inspired by Wang 2023)
    "causal_conv_kernel": 5,

    # Positional encoding
    "max_seq_len": 61,            # Supports lookback up to 60 days + 1 [STOCK] token
    "pos_encoding": "learnable",  # "learnable" or "sinusoidal"

    # Output scaling: learnable scalar α that multiplies the raw prediction.
    # Initialized at 1.0.
    "output_scale_init": 1.0,
}

# ─────────────────────────────────────────────────────────────
# 9. TRAINING
# ─────────────────────────────────────────────────────────────
TRAINING_CONFIG = {
    "batch_size": 32,
    "max_epochs": 100,
    "learning_rate": 4.299400379693557e-05,
    "min_learning_rate": 1e-6,
    "weight_decay": 0.004229862158031256,
    "warmup_epochs": 6,
    "early_stopping_patience": 15,
    "gradient_clip_norm": 1.5,

    # Huber loss
    "huber_delta": 1.1,           # 1.1 percentage point threshold (tuned)

    # Optimizer
    "optimizer": "adamw",
    "betas": (0.9, 0.999),
}

# ─────────────────────────────────────────────────────────────
# 10. REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────
# 11. FNSPID PROCESSING
# ─────────────────────────────────────────────────────────────
FNSPID_CONFIG = {
    "chunk_size": 50_000,         # Read CSV in chunks of 50K rows
    "usecols": ["Date", "Article_title", "Stock_symbol", "Article"],
    "date_col": "Date",
    "ticker_col": "Stock_symbol",
    "title_col": "Article_title",
    "article_col": "Article",
}

# ─────────────────────────────────────────────────────────────
# 12. LOGGING
# ─────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
VERBOSE = True
