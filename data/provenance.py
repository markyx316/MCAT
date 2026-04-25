"""
data/provenance.py — Data Provenance Tracking System.
======================================================
Records and reports whether each data modality used REAL or SYNTHETIC
data for every ticker. This ensures complete transparency in the paper
and prevents accidental misrepresentation of results.

Every data fetcher registers its provenance status here. The provenance
report is:
  - Logged at dataset creation time
  - Saved alongside experiment results
  - Included in the paper's data description

Usage:
    from data.provenance import DataProvenance
    prov = DataProvenance()
    prov.register("AAPL", "fundamentals", "real", "Alpha Vantage API (10 features)")
    prov.register("MSFT", "fundamentals", "synthetic", "Price-derived proxies (7 features)")
    prov.report()
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Literal
from dataclasses import dataclass, field, asdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TICKERS, RESULTS_DIR
from utils import setup_logger

logger = setup_logger(__name__)

# Valid modality names
MODALITIES = ["price", "sentiment", "fundamentals", "macro"]

# Data source types
SourceType = Literal["real", "synthetic", "partial", "unavailable"]


@dataclass
class ModalityRecord:
    """Provenance record for a single ticker × modality combination."""
    ticker: str
    modality: str
    source_type: SourceType    # "real", "synthetic", "partial", "unavailable"
    source_detail: str         # Human-readable description
    n_features: int = 0        # Number of features from this source
    coverage_pct: float = 100.0  # Percentage of dates covered with real data
    timestamp: str = ""        # When this record was created

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class DataProvenance:
    """
    Tracks data provenance for all tickers and modalities.

    Ensures every modality is explicitly registered as real or synthetic
    before the dataset is used for training/evaluation.
    """

    def __init__(self):
        # {(ticker, modality): ModalityRecord}
        self._records: Dict[tuple, ModalityRecord] = {}

    def register(
        self,
        ticker: str,
        modality: str,
        source_type: SourceType,
        source_detail: str,
        n_features: int = 0,
        coverage_pct: float = 100.0,
    ):
        """
        Register the data source for a ticker × modality combination.

        Args:
            ticker: Stock ticker symbol.
            modality: One of "price", "sentiment", "fundamentals", "macro".
            source_type: "real", "synthetic", "partial", or "unavailable".
            source_detail: Human-readable description of the data source.
            n_features: Number of features from this source.
            coverage_pct: Percentage of dates with real data (for "partial").
        """
        assert modality in MODALITIES, f"Invalid modality '{modality}'. Must be one of {MODALITIES}"
        assert source_type in ("real", "synthetic", "partial", "unavailable"), \
            f"Invalid source_type '{source_type}'"

        record = ModalityRecord(
            ticker=ticker,
            modality=modality,
            source_type=source_type,
            source_detail=source_detail,
            n_features=n_features,
            coverage_pct=coverage_pct,
        )
        self._records[(ticker, modality)] = record

        # Log synthetic and partial data prominently
        if source_type == "synthetic":
            logger.warning(
                f"⚠ SYNTHETIC DATA: {ticker}/{modality} — {source_detail}"
            )
        elif source_type == "partial":
            logger.warning(
                f"⚠ PARTIAL DATA: {ticker}/{modality} — {source_detail} "
                f"({coverage_pct:.0f}% real)"
            )
        elif source_type == "unavailable":
            logger.warning(
                f"⚠ UNAVAILABLE: {ticker}/{modality} — {source_detail}"
            )
        else:
            logger.info(f"  ✓ REAL DATA: {ticker}/{modality} — {source_detail}")

    def register_bulk(
        self,
        tickers: list,
        modality: str,
        source_type: SourceType,
        source_detail: str,
        n_features: int = 0,
    ):
        """Register the same provenance for multiple tickers."""
        for ticker in tickers:
            self.register(ticker, modality, source_type, source_detail, n_features)

    def get_record(self, ticker: str, modality: str) -> Optional[ModalityRecord]:
        """Look up the provenance record for a ticker × modality."""
        return self._records.get((ticker, modality))

    def is_real(self, ticker: str, modality: str) -> bool:
        """Check if the data for this ticker × modality is real (not synthetic)."""
        record = self.get_record(ticker, modality)
        return record is not None and record.source_type == "real"

    def check_completeness(self, tickers: list = None, modalities: list = None):
        """
        Verify that all ticker × modality combinations have been registered.
        Raises ValueError if any are missing.
        """
        if tickers is None:
            tickers = TICKERS
        if modalities is None:
            modalities = MODALITIES

        missing = []
        for ticker in tickers:
            for modality in modalities:
                if (ticker, modality) not in self._records:
                    missing.append(f"{ticker}/{modality}")

        if missing:
            raise ValueError(
                f"Provenance not registered for {len(missing)} combinations: "
                f"{', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}"
            )

    def report(self) -> str:
        """
        Generate a human-readable provenance report.
        Prominently highlights any synthetic data usage.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("  DATA PROVENANCE REPORT")
        lines.append("=" * 70)

        # Summary counts
        n_real = sum(1 for r in self._records.values() if r.source_type == "real")
        n_synthetic = sum(1 for r in self._records.values() if r.source_type == "synthetic")
        n_partial = sum(1 for r in self._records.values() if r.source_type == "partial")
        n_total = len(self._records)

        lines.append(f"  Total records: {n_total}")
        lines.append(f"  Real: {n_real} | Synthetic: {n_synthetic} | Partial: {n_partial}")
        lines.append("")

        # Per-modality summary
        for modality in MODALITIES:
            mod_records = [r for r in self._records.values() if r.modality == modality]
            if not mod_records:
                continue

            n_mod_real = sum(1 for r in mod_records if r.source_type == "real")
            n_mod_synth = sum(1 for r in mod_records if r.source_type == "synthetic")

            status = "✓ ALL REAL" if n_mod_synth == 0 else f"⚠ {n_mod_synth} SYNTHETIC"
            lines.append(f"  {modality.upper():<15} {status}")

            # List synthetic tickers explicitly
            for r in mod_records:
                if r.source_type != "real":
                    lines.append(f"    ⚠ {r.ticker}: {r.source_type} — {r.source_detail}")

        lines.append("")

        # Synthetic data warning if any
        if n_synthetic > 0 or n_partial > 0:
            lines.append("  ╔══════════════════════════════════════════════════════╗")
            lines.append("  ║  WARNING: This experiment uses synthetic data for   ║")
            lines.append("  ║  some modalities. Results for those modalities      ║")
            lines.append("  ║  should be interpreted as testing architecture,     ║")
            lines.append("  ║  NOT the value of the underlying data source.       ║")
            lines.append("  ╚══════════════════════════════════════════════════════╝")

        lines.append("=" * 70)

        report_text = "\n".join(lines)
        logger.info("\n" + report_text)
        return report_text

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            "generated_at": datetime.now().isoformat(),
            "records": [asdict(r) for r in self._records.values()],
            "summary": {
                "n_total": len(self._records),
                "n_real": sum(1 for r in self._records.values() if r.source_type == "real"),
                "n_synthetic": sum(1 for r in self._records.values() if r.source_type == "synthetic"),
                "n_partial": sum(1 for r in self._records.values() if r.source_type == "partial"),
            },
        }

    def save(self, path: Path = None):
        """Save provenance report to JSON."""
        if path is None:
            path = RESULTS_DIR / "data_provenance.json"

        # Custom encoder to handle numpy int64/float64 from pandas/numpy
        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, cls=_NumpyEncoder)
        logger.info(f"Provenance saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "DataProvenance":
        """Load provenance from JSON."""
        prov = cls()
        with open(path) as f:
            data = json.load(f)
        for rec in data["records"]:
            prov.register(
                ticker=rec["ticker"],
                modality=rec["modality"],
                source_type=rec["source_type"],
                source_detail=rec["source_detail"],
                n_features=rec.get("n_features", 0),
                coverage_pct=rec.get("coverage_pct", 100.0),
            )
        return prov


# ─────────────────────────────────────────────────────────────
# GLOBAL PROVENANCE INSTANCE
# ─────────────────────────────────────────────────────────────
# Import this in all fetcher modules to register provenance
provenance = DataProvenance()


if __name__ == "__main__":
    # Demo
    prov = DataProvenance()
    prov.register("AAPL", "price", "real", "yfinance daily OHLCV", n_features=39)
    prov.register("AAPL", "sentiment", "real", "FNSPID + FinBERT (768-dim)", n_features=769)
    prov.register("AAPL", "fundamentals", "real", "Alpha Vantage API (10 features)", n_features=10)
    prov.register("AAPL", "macro", "real", "yfinance + FRED (20 features)", n_features=20)

    prov.register("MSFT", "price", "real", "yfinance daily OHLCV", n_features=39)
    prov.register("MSFT", "sentiment", "synthetic", "VADER scores (5-dim, no FinBERT)", n_features=5)
    prov.register("MSFT", "fundamentals", "synthetic", "Price-derived proxies (7 features)", n_features=7)
    prov.register("MSFT", "macro", "real", "yfinance + FRED (20 features)", n_features=20)

    prov.report()
    print("\nJSON output:")
    print(json.dumps(prov.to_dict()["summary"], indent=2))
