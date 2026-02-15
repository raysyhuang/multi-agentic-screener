"""Integration tests — cross-provider data consistency.

Requires live API keys. Run with:
    pytest tests/test_data/test_provider_variance.py -v -m integration

These are excluded from CI by default.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import pytest

from src.data.polygon_client import PolygonClient
from src.data.fmp_client import FMPClient
from src.data.yfinance_client import YFinanceClient

logger = logging.getLogger(__name__)

# Use a well-known, highly-liquid ticker for consistency tests
TEST_TICKER = "AAPL"
# Go back far enough that all providers have settled data
TEST_TO = date.today() - timedelta(days=7)
TEST_FROM = TEST_TO - timedelta(days=30)


def _has_required_columns(df: pd.DataFrame) -> bool:
    """Check that OHLCV schema is consistent across providers."""
    required = {"open", "high", "low", "close", "volume"}
    return required.issubset(set(c.lower() for c in df.columns))


@pytest.mark.integration
class TestProviderVariance:
    """Cross-provider consistency validation."""

    @pytest.fixture(autouse=True)
    async def setup_clients(self):
        self.polygon = PolygonClient()
        self.fmp = FMPClient()
        self.yfinance = YFinanceClient()

    async def _fetch_all(self) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV from all three providers."""
        results = {}
        try:
            results["polygon"] = await self.polygon.get_ohlcv(TEST_TICKER, TEST_FROM, TEST_TO)
        except Exception as e:
            pytest.skip(f"Polygon unavailable: {e}")
        try:
            results["fmp"] = await self.fmp.get_daily_prices(TEST_TICKER, TEST_FROM, TEST_TO)
        except Exception as e:
            pytest.skip(f"FMP unavailable: {e}")
        try:
            results["yfinance"] = await self.yfinance.get_ohlcv(TEST_TICKER, TEST_FROM, TEST_TO)
        except Exception as e:
            pytest.skip(f"yfinance unavailable: {e}")
        return results

    @pytest.mark.asyncio
    async def test_column_schema_consistency(self):
        """All providers should return the same OHLCV column schema."""
        data = await self._fetch_all()
        for provider, df in data.items():
            assert not df.empty, f"{provider} returned empty DataFrame"
            assert _has_required_columns(df), (
                f"{provider} missing required columns. Got: {list(df.columns)}"
            )

    @pytest.mark.asyncio
    async def test_close_prices_within_50bps(self):
        """Close prices should match within 50 basis points across providers."""
        data = await self._fetch_all()
        providers = list(data.keys())
        if len(providers) < 2:
            pytest.skip("Need at least 2 providers for comparison")

        # Normalize date columns and align
        dfs = {}
        for name, df in data.items():
            df = df.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df = df.set_index("date")
            dfs[name] = df

        # Compare each pair
        for i, p1 in enumerate(providers):
            for p2 in providers[i + 1:]:
                common_dates = dfs[p1].index.intersection(dfs[p2].index)
                if len(common_dates) == 0:
                    continue

                close1 = dfs[p1].loc[common_dates, "close"].values
                close2 = dfs[p2].loc[common_dates, "close"].values

                pct_diff = abs(close1 - close2) / close1 * 100
                max_diff = pct_diff.max()

                assert max_diff < 0.50, (
                    f"{p1} vs {p2}: max close price diff = {max_diff:.4f}% "
                    f"(exceeds 50bps threshold)"
                )
                logger.info(
                    "%s vs %s: max close diff = %.4f%% across %d dates",
                    p1, p2, max_diff, len(common_dates),
                )

    @pytest.mark.asyncio
    async def test_volume_within_order_of_magnitude(self):
        """Volume should be within 1 order of magnitude across providers."""
        data = await self._fetch_all()
        providers = list(data.keys())
        if len(providers) < 2:
            pytest.skip("Need at least 2 providers for comparison")

        dfs = {}
        for name, df in data.items():
            df = df.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df = df.set_index("date")
            dfs[name] = df

        for i, p1 in enumerate(providers):
            for p2 in providers[i + 1:]:
                common_dates = dfs[p1].index.intersection(dfs[p2].index)
                if len(common_dates) == 0:
                    continue

                vol1 = dfs[p1].loc[common_dates, "volume"].values.astype(float)
                vol2 = dfs[p2].loc[common_dates, "volume"].values.astype(float)

                # Avoid division by zero
                mask = (vol1 > 0) & (vol2 > 0)
                if mask.sum() == 0:
                    continue

                ratio = vol1[mask] / vol2[mask]
                max_ratio = max(ratio.max(), 1.0 / ratio.min())

                assert max_ratio < 10.0, (
                    f"{p1} vs {p2}: volume ratio = {max_ratio:.1f}x "
                    f"(exceeds 10x threshold)"
                )
                logger.info(
                    "%s vs %s: max volume ratio = %.1fx across %d dates",
                    p1, p2, max_ratio, mask.sum(),
                )

    @pytest.mark.asyncio
    async def test_date_coverage_overlap(self):
        """Providers should have at least 80% overlap in trading dates."""
        data = await self._fetch_all()
        providers = list(data.keys())
        if len(providers) < 2:
            pytest.skip("Need at least 2 providers")

        date_sets = {}
        for name, df in data.items():
            if "date" in df.columns:
                date_sets[name] = set(pd.to_datetime(df["date"]).dt.date)
            else:
                date_sets[name] = set(df.index)

        for i, p1 in enumerate(providers):
            for p2 in providers[i + 1:]:
                overlap = date_sets[p1] & date_sets[p2]
                union = date_sets[p1] | date_sets[p2]
                pct = len(overlap) / len(union) * 100 if union else 0

                assert pct >= 80, (
                    f"{p1} vs {p2}: date overlap = {pct:.0f}% "
                    f"(below 80% threshold). "
                    f"{p1}={len(date_sets[p1])}, {p2}={len(date_sets[p2])}, "
                    f"overlap={len(overlap)}"
                )
