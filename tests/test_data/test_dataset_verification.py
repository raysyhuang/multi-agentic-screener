"""Tests for dataset health verification system."""

from datetime import date, timedelta

import pandas as pd

from src.data.dataset_verification import (
    verify_dataset,
    verify_cross_engine,
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


# ---------------------------------------------------------------------------
# verify_cross_engine tests
# ---------------------------------------------------------------------------

class TestVerifyCrossEngine:
    def test_empty_engines(self):
        """No engine results should return a pass."""
        report = verify_cross_engine([], {"regime": "bull"})
        assert report.passed is True

    def test_consensus_pass(self):
        """All engines agreeing on regime should pass consensus check."""
        engines = [
            {
                "engine_name": "koocore_d",
                "regime": "bull",
                "picks": [{"ticker": "AAPL", "entry_price": 200}],
                "candidates_screened": 100,
            },
            {
                "engine_name": "gemini_stst",
                "regime": "bull",
                "picks": [{"ticker": "AAPL", "entry_price": 201}],
                "candidates_screened": 120,
            },
        ]
        report = verify_cross_engine(engines, {"regime": "bull"})
        consensus = next((c for c in report.checks if c.name == "regime_consensus"), None)
        assert consensus is not None
        assert consensus.passed is True

    def test_regime_divergence_warns(self):
        """Engines disagreeing on regime should trigger a warning."""
        engines = [
            {
                "engine_name": "koocore_d",
                "regime": "bull",
                "picks": [{"ticker": "AAPL", "entry_price": 200}],
                "candidates_screened": 100,
            },
            {
                "engine_name": "gemini_stst",
                "regime": "bear",
                "picks": [{"ticker": "MSFT", "entry_price": 400}],
                "candidates_screened": 120,
            },
        ]
        report = verify_cross_engine(engines, {"regime": "bull"})
        consensus = next((c for c in report.checks if c.name == "regime_consensus"), None)
        assert consensus is not None
        assert consensus.passed is False

    def test_pick_overlap_detected(self):
        """Overlapping tickers should be detected."""
        engines = [
            {
                "engine_name": "engine_a",
                "regime": "bull",
                "picks": [
                    {"ticker": "AAPL", "entry_price": 200},
                    {"ticker": "MSFT", "entry_price": 400},
                ],
                "candidates_screened": 100,
            },
            {
                "engine_name": "engine_b",
                "regime": "bull",
                "picks": [
                    {"ticker": "AAPL", "entry_price": 201},
                    {"ticker": "TSLA", "entry_price": 300},
                ],
                "candidates_screened": 100,
            },
        ]
        report = verify_cross_engine(engines, {"regime": "bull"})
        overlap = next((c for c in report.checks if c.name == "pick_overlap"), None)
        assert overlap is not None
        assert overlap.passed is True
        assert overlap.value == 1  # AAPL in both

    def test_price_inconsistency_detected(self):
        """Tickers with >2% price spread should trigger warning."""
        engines = [
            {
                "engine_name": "engine_a",
                "regime": "bull",
                "picks": [{"ticker": "AAPL", "entry_price": 200}],
                "candidates_screened": 100,
            },
            {
                "engine_name": "engine_b",
                "regime": "bull",
                "picks": [{"ticker": "AAPL", "entry_price": 220}],  # 10% off
                "candidates_screened": 100,
            },
        ]
        report = verify_cross_engine(engines, {"regime": "bull"})
        price = next((c for c in report.checks if c.name == "price_consistency"), None)
        assert price is not None
        assert price.passed is False


# ---------------------------------------------------------------------------
# Sort fix test
# ---------------------------------------------------------------------------

class TestUniverseSort:
    def test_dollar_volume_sort_produces_diverse_tickers(self):
        """Sorting by dollar volume should surface large-cap tickers regardless of alphabetical order."""
        # Simulate Polygon fallback: no marketCap, alphabetically sorted
        universe = []
        # Add some small-caps from A-C with low dollar volume
        for i, ticker in enumerate(["AA01", "AB02", "AC03", "BA04", "BB05", "BC06", "CA07", "CB08"]):
            universe.append({
                "symbol": ticker,
                "price": 10 + i,
                "volume": 500_000,
                "marketCap": 0,
            })

        # Add mega-caps from later alphabet with high dollar volume
        mega = [
            ("MSFT", 400, 30_000_000),
            ("NVDA", 800, 40_000_000),
            ("TSLA", 250, 60_000_000),
            ("JPM", 200, 15_000_000),
            ("AAPL", 200, 50_000_000),
        ]
        for ticker, price, vol in mega:
            universe.append({
                "symbol": ticker,
                "price": price,
                "volume": vol,
                "marketCap": 0,  # Polygon fallback: no market cap
            })

        # Apply the same sort logic as main.py
        universe.sort(
            key=lambda s: (
                s.get("marketCap") or 0,
                (s.get("price") or s.get("lastPrice") or 0) * (s.get("volume") or 0),
            ),
            reverse=True,
        )

        top_5 = [s["symbol"] for s in universe[:5]]

        # All mega-caps should be in top 5 because their dollar volume is much higher
        for mc in ["MSFT", "NVDA", "TSLA", "AAPL", "JPM"]:
            assert mc in top_5, f"{mc} should be in top 5, got {top_5}"

        # First letter diversity: top 5 should span multiple letters
        first_letters = set(t[0] for t in top_5)
        assert len(first_letters) >= 3, f"Expected >= 3 first letters, got {first_letters}"

    def test_market_cap_primary_sort_still_works(self):
        """When marketCap is available, it should be the primary sort key."""
        universe = [
            {"symbol": "ZZZ", "price": 10, "volume": 100_000_000, "marketCap": 1_000_000_000_000},
            {"symbol": "AAA", "price": 500, "volume": 50_000_000, "marketCap": 500_000_000_000},
        ]

        universe.sort(
            key=lambda s: (
                s.get("marketCap") or 0,
                (s.get("price") or s.get("lastPrice") or 0) * (s.get("volume") or 0),
            ),
            reverse=True,
        )

        assert universe[0]["symbol"] == "ZZZ"  # Higher market cap wins


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestBusinessDays:
    def test_same_day(self):
        d = date(2026, 2, 16)  # Monday
        assert _business_days_between(d, d) == 0

    def test_weekdays(self):
        # Monday to Friday = 4 business days
        assert _business_days_between(date(2026, 2, 16), date(2026, 2, 20)) == 4

    def test_over_weekend(self):
        # Friday to Monday = 1 business day
        assert _business_days_between(date(2026, 2, 13), date(2026, 2, 16)) == 1
