"""FRED / macro data client — VIX, yield curve, breadth indicators."""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import date, timedelta

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# FRED data is free, no key needed for basic series via yfinance / public endpoints.
# We use yfinance for VIX (^VIX) and treasury yields since FRED API requires registration.
# For broader macro data we hit the FRED public JSON endpoint.

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Bounded retry config for transient FRED failures (5xx + network errors).
# Worst-case added latency before final fail-closed: ~3s with jitter.
_FRED_RETRY_MAX_ATTEMPTS = 3
_FRED_RETRY_BASE_DELAY = 0.5
_FRED_RETRY_BACKOFF_FACTOR = 2.0
_FRED_RETRY_JITTER = 0.25
_FRED_TIMEOUT_S = 30


async def _fetch_fred_observations(
    url: str, params: dict, series_id: str
) -> httpx.Response:
    """GET FRED observations with bounded retries.

    Retries on httpx.TransportError (incl. ConnectError, ReadError,
    NetworkError) and httpx.TimeoutException, plus HTTP 5xx responses.
    Never retries 4xx — those are real client errors. Final attempt
    raises the underlying exception so callers' fail-closed semantics
    are preserved.
    """
    for attempt in range(1, _FRED_RETRY_MAX_ATTEMPTS + 1):
        is_last = attempt == _FRED_RETRY_MAX_ATTEMPTS
        try:
            async with httpx.AsyncClient(timeout=_FRED_TIMEOUT_S) as client:
                resp = await client.get(url, params=params)

            if 500 <= resp.status_code < 600 and not is_last:
                logger.warning(
                    "FRED %s attempt %d/%d got HTTP %d — retrying",
                    series_id, attempt, _FRED_RETRY_MAX_ATTEMPTS, resp.status_code,
                )
            else:
                # 2xx success, 4xx (no retry), or final-attempt 5xx (raise).
                resp.raise_for_status()
                if attempt > 1:
                    logger.info(
                        "FRED %s succeeded on attempt %d/%d",
                        series_id, attempt, _FRED_RETRY_MAX_ATTEMPTS,
                    )
                return resp
        except (httpx.TransportError, httpx.TimeoutException) as e:
            if is_last:
                logger.error(
                    "FRED %s exhausted %d attempts (%s: %s)",
                    series_id, _FRED_RETRY_MAX_ATTEMPTS, type(e).__name__, e,
                )
                raise
            logger.warning(
                "FRED %s attempt %d/%d %s: %s — retrying",
                series_id, attempt, _FRED_RETRY_MAX_ATTEMPTS, type(e).__name__, e,
            )

        # Reached only when retrying. Exponential backoff with jitter.
        delay = _FRED_RETRY_BASE_DELAY * (_FRED_RETRY_BACKOFF_FACTOR ** (attempt - 1))
        delay = max(0.0, delay + random.uniform(-_FRED_RETRY_JITTER, _FRED_RETRY_JITTER))
        await asyncio.sleep(delay)

    # Unreachable: the loop either returns on success or raises on exhaustion.
    raise RuntimeError("FRED retry loop exited unexpectedly")

# Key series IDs
SERIES = {
    "vix": "VIXCLS",
    "yield_10y": "DGS10",
    "yield_2y": "DGS2",
    "yield_spread": "T10Y2Y",  # 10Y-2Y spread
    "fed_funds": "FEDFUNDS",
    "initial_claims": "ICSA",
}


class FREDClient:
    def __init__(self, api_key: str | None = None):
        self._api_key = api_key  # Optional — we also use yfinance fallback

    async def get_series(
        self,
        series_id: str,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> pd.DataFrame:
        """Fetch a FRED series. Falls back to empty if no API key."""
        if not self._api_key:
            return await self._yfinance_fallback(series_id, from_date, to_date)

        if from_date is None:
            from_date = date.today() - timedelta(days=365)
        if to_date is None:
            to_date = date.today()

        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": str(from_date),
            "observation_end": str(to_date),
        }
        resp = await _fetch_fred_observations(FRED_BASE, params, series_id)
        data = resp.json()

        observations = data.get("observations", [])
        if not observations:
            return pd.DataFrame()

        df = pd.DataFrame(observations)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", "value"]].dropna().reset_index(drop=True)
        return df

    async def _yfinance_fallback(
        self, series_id: str, from_date: date | None, to_date: date | None
    ) -> pd.DataFrame:
        """Use yfinance for VIX when no FRED key available."""
        import yfinance as yf
        from concurrent.futures import ThreadPoolExecutor

        yf_map = {
            "VIXCLS": "^VIX",
            "DGS10": "^TNX",
            "DGS2": "^IRX",
        }
        yf_ticker = yf_map.get(series_id)
        if not yf_ticker:
            return pd.DataFrame()

        start = str(from_date or date.today() - timedelta(days=365))
        end = str(to_date or date.today())

        def _fetch():
            tk = yf.Ticker(yf_ticker)
            df = tk.history(start=start, end=end)
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["Date"]).dt.date
            df["value"] = df["Close"]
            return df[["date", "value"]].reset_index(drop=True)

        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        return await loop.run_in_executor(executor, _fetch)

    async def get_vix(self, lookback_days: int = 90) -> pd.DataFrame:
        from_date = date.today() - timedelta(days=lookback_days)
        return await self.get_series(SERIES["vix"], from_date)

    async def get_yield_curve(self, lookback_days: int = 90) -> pd.DataFrame:
        """Return 10Y-2Y spread."""
        from_date = date.today() - timedelta(days=lookback_days)
        return await self.get_series(SERIES["yield_spread"], from_date)

    async def get_macro_snapshot(self) -> dict:
        """Fetch latest values for key macro indicators."""
        vix = await self.get_vix(lookback_days=5)
        yield_curve = await self.get_yield_curve(lookback_days=5)
        return {
            "vix": float(vix["value"].iloc[-1]) if not vix.empty else None,
            "yield_spread_10y2y": float(yield_curve["value"].iloc[-1]) if not yield_curve.empty else None,
        }
