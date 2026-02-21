"""Financial Modeling Prep client — fundamentals, earnings, insider transactions."""

from __future__ import annotations

import asyncio
import logging
from datetime import date

import httpx
import pandas as pd

from src.config import get_settings
from src.data.circuit_breaker import RateLimitError

logger = logging.getLogger(__name__)

BASE_URL = "https://financialmodelingprep.com/stable"
NON_RETRYABLE_STATUS = {401, 402, 403}

MAX_RETRIES = 3
BACKOFF_BASE = 0.5  # seconds: 0.5, 1.0, 2.0


class FMPFatalError(Exception):
    """Raised for non-retryable FMP auth/plan/access errors."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"FMP fatal error {status_code}: {message}")


class FMPDisabledError(Exception):
    """Raised when FMP client has been disabled for this process."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"FMP disabled: {reason}")


def _extract_error_message(resp: httpx.Response) -> str:
    """Best-effort extraction of FMP error detail."""
    try:
        payload = resp.json()
    except Exception:
        payload = None

    if isinstance(payload, dict):
        for key in ("Error Message", "error", "message"):
            value = payload.get(key)
            if value:
                return str(value)

    text = (resp.text or "").strip()
    if text:
        return text[:240]
    return "access denied by provider"


async def _request_with_backoff(
    client: httpx.AsyncClient, url: str, params: dict,
) -> httpx.Response:
    """Make an HTTP GET with exponential backoff on 429 responses."""
    for attempt in range(MAX_RETRIES):
        resp = await client.get(url, params=params)
        if resp.status_code in NON_RETRYABLE_STATUS:
            raise FMPFatalError(resp.status_code, _extract_error_message(resp))
        if resp.status_code == 429:
            if attempt < MAX_RETRIES - 1:
                delay = BACKOFF_BASE * (2 ** attempt)
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                logger.warning(
                    "FMP 429 rate limited (attempt %d/%d) — retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
                continue
            raise RateLimitError("fmp", retry_after=None)
        resp.raise_for_status()
        return resp
    raise RateLimitError("fmp")


def _ensure_list(data, context: str = "") -> list[dict]:
    """Ensure FMP response is a list. FMP sometimes returns error dicts instead of lists."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "Error Message" in data or "error" in data:
            logger.warning("FMP returned error%s: %s", f" for {context}" if context else "", data)
        return []
    logger.warning("FMP returned unexpected type%s: %s", f" for {context}" if context else "", type(data).__name__)
    return []


class FMPClient:
    def __init__(self):
        self._api_key = get_settings().fmp_api_key
        self._disabled_reason: str | None = None

    def _params(self, **kwargs) -> dict:
        return {"apikey": self._api_key, **kwargs}

    def _ensure_enabled(self) -> None:
        if self._disabled_reason:
            raise FMPDisabledError(self._disabled_reason)

    async def _request(self, url: str, params: dict) -> httpx.Response:
        self._ensure_enabled()
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                return await _request_with_backoff(client, url, params)
            except FMPFatalError as exc:
                reason = f"{exc.status_code} {exc.message}"
                if not self._disabled_reason:
                    logger.error("Disabling FMP client for this process: %s", reason)
                self._disabled_reason = reason
                raise

    async def get_earnings_calendar(
        self, from_date: date, to_date: date
    ) -> list[dict]:
        """Upcoming earnings dates."""
        url = f"{BASE_URL}/earnings-calendar"
        params = self._params(**{"from": str(from_date), "to": str(to_date)})
        resp = await self._request(url, params)
        return _ensure_list(resp.json(), "earnings_calendar")

    async def get_earnings_surprise(self, ticker: str) -> list[dict]:
        """Historical earnings data (actual vs estimate)."""
        url = f"{BASE_URL}/earnings"
        resp = await self._request(url, self._params(symbol=ticker))
        return _ensure_list(resp.json(), f"earnings_surprise/{ticker}")

    async def get_insider_trading(self, ticker: str, limit: int = 50) -> list[dict]:
        """Recent insider transactions."""
        url = f"{BASE_URL}/insider-trading/search"
        params = self._params(symbol=ticker, limit=limit, page=0)
        resp = await self._request(url, params)
        return _ensure_list(resp.json(), f"insider_trading/{ticker}")

    async def get_institutional_holders(self, ticker: str) -> list[dict]:
        """Institutional ownership data."""
        url = f"{BASE_URL}/institutional-ownership/symbol-positions-summary"
        resp = await self._request(url, self._params(symbol=ticker))
        return _ensure_list(resp.json(), f"institutional_holders/{ticker}")

    async def get_company_profile(self, ticker: str) -> dict:
        """Company profile with sector, market cap, etc."""
        url = f"{BASE_URL}/profile"
        resp = await self._request(url, self._params(symbol=ticker))
        data = resp.json()
        return data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})

    async def get_stock_screener(
        self,
        market_cap_more_than: int = 300_000_000,
        volume_more_than: int = 500_000,
        price_more_than: float = 5.0,
        exchange: str = "NYSE,NASDAQ",
        limit: int = 5000,
    ) -> list[dict]:
        """Screen stocks by basic criteria — used for universe construction."""
        url = f"{BASE_URL}/company-screener"
        params = self._params(
            marketCapMoreThan=market_cap_more_than,
            volumeMoreThan=volume_more_than,
            priceMoreThan=price_more_than,
            exchange=exchange,
            limit=limit,
        )
        resp = await self._request(url, params)
        return _ensure_list(resp.json(), "stock_screener")

    async def get_key_metrics(self, ticker: str, period: str = "annual") -> list[dict]:
        """Key financial metrics (P/E, EV/EBITDA, etc.)."""
        url = f"{BASE_URL}/key-metrics"
        params = self._params(symbol=ticker, period=period)
        resp = await self._request(url, params)
        return _ensure_list(resp.json(), f"key_metrics/{ticker}")

    async def get_daily_prices(
        self, ticker: str, from_date: date, to_date: date
    ) -> pd.DataFrame:
        """Historical daily prices as DataFrame."""
        url = f"{BASE_URL}/historical-price-eod/full"
        params = self._params(symbol=ticker, **{"from": str(from_date), "to": str(to_date)})
        resp = await self._request(url, params)
        data = resp.json()

        if not isinstance(data, dict):
            logger.warning("FMP daily_prices/%s returned unexpected type: %s", ticker, type(data).__name__)
            return pd.DataFrame()
        historical = data.get("historical", [])
        if not historical:
            return pd.DataFrame()

        df = pd.DataFrame(historical)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").reset_index(drop=True)
        return df
