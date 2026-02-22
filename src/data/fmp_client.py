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
V3_URL = "https://financialmodelingprep.com/api/v3"
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
        self._settings = get_settings()
        self._api_key = self._settings.fmp_api_key
        self._disabled_reason: str | None = None
        self._call_day = date.today()
        self._call_count = 0
        self._warned_budget = False
        self._analyst_estimates_enabled = True
        self._analyst_estimates_status = "supported"
        self._stock_news_v3_enabled = True
        self._stock_news_v3_status = "supported"
        self._stock_news_stable_enabled = True
        self._stock_news_stable_status = "supported"

    def _params(self, **kwargs) -> dict:
        return {"apikey": self._api_key, **kwargs}

    def _ensure_enabled(self) -> None:
        if self._disabled_reason:
            raise FMPDisabledError(self._disabled_reason)

    def _record_call(self, endpoint: str) -> None:
        today = date.today()
        if self._call_day != today:
            self._call_day = today
            self._call_count = 0
            self._warned_budget = False

        self._call_count += 1
        budget = max(0, int(self._settings.fmp_daily_call_budget))
        warn_pct = float(self._settings.fmp_budget_warn_threshold_pct or 0.8)

        if budget > 0:
            used_pct = self._call_count / budget
            if used_pct >= warn_pct and not self._warned_budget:
                logger.warning(
                    "FMP daily call budget at %.0f%% (%d/%d).",
                    used_pct * 100, self._call_count, budget,
                )
                self._warned_budget = True

            if self._call_count > budget and self._settings.fmp_enforce_daily_budget:
                raise RateLimitError("fmp", retry_after=None)

        if self._call_count % 100 == 0:
            logger.info("FMP call usage today: %d calls (last endpoint: %s)", self._call_count, endpoint)

    def get_budget_status(self) -> dict:
        budget = max(0, int(self._settings.fmp_daily_call_budget))
        remaining = max(0, budget - self._call_count) if budget > 0 else None
        used_pct = (self._call_count / budget) if budget > 0 else 0.0
        return {
            "date": str(self._call_day),
            "calls_used": self._call_count,
            "daily_budget": budget,
            "calls_remaining": remaining,
            "used_pct": round(used_pct, 4),
            "enforce_budget": bool(self._settings.fmp_enforce_daily_budget),
        }

    def get_endpoint_status(self) -> dict:
        """Expose endpoint availability state for dashboard diagnostics."""
        base_status = "supported" if self._disabled_reason is None else "disabled"
        return {
            "client_enabled": self._disabled_reason is None,
            "disabled_reason": self._disabled_reason,
            "calls_used": self._call_count,
            "daily_budget": max(0, int(self._settings.fmp_daily_call_budget)),
            "endpoints": {
                "profile": base_status,
                "earnings": base_status,
                "insider_trading": base_status,
                "screener": base_status,
                "ratios": base_status,
                "analyst_estimates": self._analyst_estimates_status,
                "stock_news_bulk_v3": self._stock_news_v3_status,
                "stock_news_bulk_stable": self._stock_news_stable_status,
            },
        }

    async def _request(self, url: str, params: dict) -> httpx.Response:
        self._ensure_enabled()
        endpoint = url.rstrip("/").split("/")[-1] or "unknown"
        self._record_call(endpoint)
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

    async def get_analyst_estimates(
        self,
        ticker: str,
        period: str = "quarter",
        limit: int = 8,
    ) -> list[dict]:
        """Analyst EPS/revenue estimates for upcoming periods."""
        if not self._analyst_estimates_enabled:
            return []
        url = f"{BASE_URL}/analyst-estimates"
        params = self._params(symbol=ticker, period=period)
        # Endpoint availability varies by plan. Never disable the whole FMP client
        # just because analyst-estimates is unavailable.
        try:
            self._ensure_enabled()
            self._record_call("analyst-estimates")
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await _request_with_backoff(client, url, params)
            return _ensure_list(resp.json(), f"analyst_estimates/{ticker}")[:limit]
        except FMPFatalError as exc:
            if exc.status_code in {402, 403}:
                logger.info(
                    "FMP analyst-estimates plan-gated (%d); disabling endpoint for this run.",
                    exc.status_code,
                )
                self._analyst_estimates_enabled = False
                self._analyst_estimates_status = "plan_gated"
                return []
            if exc.status_code == 401:
                self._analyst_estimates_status = "auth_error"
                reason = f"{exc.status_code} {exc.message}"
                if not self._disabled_reason:
                    logger.error("Disabling FMP client for this process: %s", reason)
                self._disabled_reason = reason
            raise
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code in {400, 404}:
                logger.info(
                    "FMP analyst-estimates unavailable (%d); disabling endpoint for this run.",
                    exc.response.status_code,
                )
                self._analyst_estimates_enabled = False
                self._analyst_estimates_status = "unsupported"
                return []
            raise

    async def get_ratios(self, ticker: str, period: str = "annual") -> dict:
        """Financial ratios (P/E, P/B, debt/equity, etc.)."""
        url = f"{BASE_URL}/ratios"
        params = self._params(symbol=ticker, period=period)
        resp = await self._request(url, params)
        data = resp.json()
        if isinstance(data, list):
            return data[0] if data else {}
        return data if isinstance(data, dict) else {}

    async def get_stock_news_bulk(self, tickers: list[str], limit: int = 200) -> list[dict]:
        """Bulk stock news fetch for multiple tickers in one request."""
        clean = [t.strip().upper() for t in tickers if t and t.strip()]
        if not clean:
            return []

        joined = ",".join(dict.fromkeys(clean))

        # Prefer v3 bulk endpoint for multi-ticker queries when available.
        data: list[dict] = []
        if self._stock_news_v3_enabled:
            url = f"{V3_URL}/stock_news"
            params = {"tickers": joined, "limit": limit, "apikey": self._api_key}
            try:
                self._ensure_enabled()
                self._record_call("stock_news_bulk_v3")
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await _request_with_backoff(client, url, params)
                data = _ensure_list(resp.json(), "stock_news_bulk")
            except FMPFatalError as exc:
                # Legacy v3 access is often plan-gated; disable this route only.
                logger.info(
                    "FMP v3 bulk stock_news unavailable (%d): %s — falling back to stable endpoint.",
                    exc.status_code, exc.message,
                )
                if exc.status_code == 401:
                    self._stock_news_v3_status = "auth_error"
                    self._stock_news_v3_enabled = False
                elif exc.status_code in {402, 403}:
                    self._stock_news_v3_status = "plan_gated"
                    self._stock_news_v3_enabled = False
            except Exception as exc:
                logger.warning("FMP v3 bulk stock_news failed: %s", exc)

        if data:
            return data

        # Fallback to stable endpoint shape.
        if not self._stock_news_stable_enabled:
            return []
        stable_url = f"{BASE_URL}/stock-news"
        stable_params = self._params(symbol=joined, limit=limit)
        try:
            stable_resp = await self._request(stable_url, stable_params)
            return _ensure_list(stable_resp.json(), "stock_news_bulk_stable")
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code in {400, 404}:
                self._stock_news_stable_enabled = False
                self._stock_news_stable_status = "unsupported"
                logger.info(
                    "FMP stable stock-news unavailable (%d); disabling endpoint for this run.",
                    exc.response.status_code,
                )
                return []
            raise

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
