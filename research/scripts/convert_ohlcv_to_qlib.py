"""Convert OHLCV data to Qlib binary format for factor research.

Fetches 2 years of daily OHLCV for the MAS ticker universe via yfinance,
saves as CSV, then converts to Qlib binary format using qlib.data.dump_bin.

Usage:
    python research/scripts/convert_ohlcv_to_qlib.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

# MAS ticker universe (representative sample — extend as needed)
TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "BAC", "GS", "V", "MA", "UNH", "JNJ", "PFE",
    "XOM", "CVX", "COP", "LMT", "RTX", "GD",
    "DIS", "NFLX", "CMCSA", "AMD", "INTC", "AVGO",
    "HD", "LOW", "COST", "WMT",
]

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
CSV_DIR = DATA_DIR / "csv"
QLIB_DIR = DATA_DIR / "qlib_data"


def fetch_ohlcv() -> None:
    """Fetch 2 years of daily OHLCV and save per-ticker CSVs."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install -r research/requirements-research.txt")
        sys.exit(1)

    CSV_DIR.mkdir(parents=True, exist_ok=True)

    end_date = date.today()
    start_date = end_date - timedelta(days=730)

    print(f"Fetching OHLCV for {len(TICKERS)} tickers ({start_date} to {end_date})...")

    for ticker in TICKERS:
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            df = yf.download(
                ticker,
                start=str(start_date),
                end=str(end_date),
                progress=False,
            )
            if df.empty:
                print("SKIP (no data)")
                continue
            # Flatten multi-level columns from yfinance
            import pandas as pd
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Qlib expects: date, open, high, low, close, volume
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index.name = "date"
            csv_path = CSV_DIR / f"{ticker}.csv"
            df.to_csv(csv_path)
            print(f"OK ({len(df)} rows)")
        except Exception as e:
            print(f"ERROR ({e})")

    print(f"\nCSVs saved to {CSV_DIR}")


def convert_to_qlib() -> None:
    """Convert CSVs to Qlib binary format."""
    try:
        from qlib.scripts.dump_bin import DumpDataAll
    except ImportError:
        try:
            from qlib.utils.dump_bin import DumpDataAll  # type: ignore[no-redef]
        except ImportError:
            print("ERROR: qlib not installed. Run: pip install -r research/requirements-research.txt")
            sys.exit(1)

    QLIB_DIR.mkdir(parents=True, exist_ok=True)

    print("\nConverting CSVs to Qlib binary format...")
    DumpDataAll(
        csv_path=str(CSV_DIR),
        qlib_dir=str(QLIB_DIR),
        freq="day",
        max_workers=4,
        include_fields="open,high,low,close,volume",
        symbol_field_name="",
    ).dump()
    print(f"Qlib data saved to {QLIB_DIR}")


if __name__ == "__main__":
    fetch_ohlcv()
    convert_to_qlib()
