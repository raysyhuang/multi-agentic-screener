"""Tests for dataset health verification system."""

from datetime import date, timedelta

import pandas as pd

from src.data.dataset_verification import (
    verify_dataset,
    _business_days_between,
    DatasetHealthReport,
    MEGA_CAP_REFERENCE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n_bars: int = 200, last_date: date | None = None) -> pd.DataFrame:
    """Create a dummy OHLCV DataFrame."""
    if last_date is None:
        last_date = date.today()
    dates = [last_date - timedelta(days=i) for i in range(n_bars)]
    dates.reverse()
    return pd.DataFrame({
        "date": dates,
        "open": [100.0] * n_bars,
        "high": [105.0] * n_bars,
        "low": [95.0] * n_bars,
        "close": [102.0] * n_bars,
        "volume": [1_000_000] * n_bars,
    })


def _diverse_universe(n: int = 200) -> tuple[list[dict], list[str]]:
    """Build a diverse universe with tickers spanning A-Z and realistic dollar volume tiers."""
    import string
    tickers = []
    filtered = []
    for i in range(n):
        letter = string.ascii_uppercase[i % 26]
        ticker = f"{letter}{'BCDE'[i % 4]}{i:03d}"
        tickers.append(ticker)
        # Create a realistic spread of dollar volumes:
        #   top ~30: $100M+ (large-cap)
        #   next ~40: $50-100M
        #   next ~60: $10-50M
        #   rest: <$10M
        if i < 30:
            price, volume = 200, 2_000_000      # $400M dv
        elif i < 70:
            price, volume = 50, 1_200_000        # $60M dv
        elif i < 130:
            price, volume = 30, 800_000          # $24M dv
        else:
            price, volume = 15, 300_000          # $4.5M dv
        filtered.append({
            "symbol": ticker,
            "price": price,
            "volume": volume,
            "marketCap": (500 - i) * 1_000_000,
        })
    return filtered, tickers


def _biased_universe(n: int = 200) -> tuple[list[dict], list[str]]:
    """Build an alphabetically biased universe (A-C tickers only)."""
    tickers = []
    filtered = []
    letters = ["A", "B", "C"]
    for i in range(n):
        letter = letters[i % 3]
        ticker = f"{letter}{letter}{i:04d}"
        tickers.append(ticker)
        filtered.append({
            "symbol": ticker,
            "price": 10 + i * 0.1,
            "volume": 500_000 + i * 100,
            "marketCap": 0,  # No market cap (Polygon fallback)
        })
    return filtered, tickers


# ---------------------------------------------------------------------------
# verify_dataset tests
# ---------------------------------------------------------------------------

class TestVerifyDataset:
    def test_diverse_universe_passes(self):
        """A diverse universe with good data should pass all checks."""
        today = date.today()
        filtered, tickers = _diverse_universe(200)
        # Add mega-caps
        for mc in MEGA_CAP_REFERENCE:
            if mc not in tickers:
                tickers.append(mc)
                filtered.append({
                    "symbol": mc, "price": 200, "volume": 50_000_000,
                    "marketCap": 2_000_000_000_000,
                })

        price_data = {t: _make_ohlcv(200, today) for t in tickers}
        qualified = tickers[:]

        report = verify_dataset(filtered, tickers, price_data, qualified, today)

        assert isinstance(report, DatasetHealthReport)
        assert report.passed is True
        assert report.passed_count == report.total_checks
        assert len(report.warnings) == 0

    def test_biased_universe_warns(self):
        """An A-C only universe should fail alphabet diversity and mega-cap coverage."""
        today = date.today()
        filtered, tickers = _biased_universe(200)
        price_data = {t: _make_ohlcv(200, today) for t in tickers}
        qualified = tickers[:]

        report = verify_dataset(filtered, tickers, price_data, qualified, today)

        assert report.passed is False
        assert len(report.warnings) > 0

        check_names = {c.name for c in report.checks if not c.passed}
        assert "alphabet_diversity" in check_names
        assert "mega_cap_coverage" in check_names

    def test_stale_data_warns(self):
        """Data that's too old should trigger a freshness warning."""
        today = date.today()
        filtered, tickers = _diverse_universe(100)
        for mc in MEGA_CAP_REFERENCE:
            if mc not in tickers:
                tickers.append(mc)
                filtered.append({
                    "symbol": mc, "price": 200, "volume": 50_000_000,
                    "marketCap": 2_000_000_000_000,
                })

        # Create OHLCV data that's 10 business days old
        stale_date = today - timedelta(days=14)
        price_data = {t: _make_ohlcv(200, stale_date) for t in tickers}
        qualified = tickers[:]

        report = verify_dataset(filtered, tickers, price_data, qualified, today)

        freshness_check = next(c for c in report.checks if c.name == "data_freshness")
        assert freshness_check.passed is False

    def test_small_universe_warns(self):
        """A universe with fewer than 50 tickers should trigger a size warning."""
        today = date.today()
        filtered, tickers = _diverse_universe(30)
        for mc in MEGA_CAP_REFERENCE:
            if mc not in tickers:
                tickers.append(mc)
                filtered.append({
                    "symbol": mc, "price": 200, "volume": 50_000_000,
                    "marketCap": 2_000_000_000_000,
                })

        price_data = {t: _make_ohlcv(200, today) for t in tickers}
        qualified = tickers[:30]  # Only 30 qualified

        report = verify_dataset(filtered, tickers, price_data, qualified, today)

        size_check = next(c for c in report.checks if c.name == "universe_size")
        assert size_check.passed is False

    def test_low_ohlcv_completeness_warns(self):
        """Tickers with few bars should fail completeness check."""
        today = date.today()
        filtered, tickers = _diverse_universe(100)
        for mc in MEGA_CAP_REFERENCE:
            if mc not in tickers:
                tickers.append(mc)
                filtered.append({
                    "symbol": mc, "price": 200, "volume": 50_000_000,
                    "marketCap": 2_000_000_000_000,
                })

        # Most tickers have only 20 bars (below MIN_OHLCV_BARS=100)
        price_data = {t: _make_ohlcv(20, today) for t in tickers}
        qualified = tickers[:]

        report = verify_dataset(filtered, tickers, price_data, qualified, today)

        comp_check = next(c for c in report.checks if c.name == "ohlcv_completeness")
        assert comp_check.passed is False

    def test_to_dict_serializable(self):
        """Report should be JSON-serializable."""
        import json
        today = date.today()
        filtered, tickers = _diverse_universe(60)
        for mc in MEGA_CAP_REFERENCE:
            if mc not in tickers:
                tickers.append(mc)
                filtered.append({
                    "symbol": mc, "price": 200, "volume": 50_000_000,
                    "marketCap": 2_000_000_000_000,
                })
        price_data = {t: _make_ohlcv(200, today) for t in tickers}
        qualified = tickers[:]

        report = verify_dataset(filtered, tickers, price_data, qualified, today)
        d = report.to_dict()

        # Should not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        assert "passed" in d
        assert "checks" in d
