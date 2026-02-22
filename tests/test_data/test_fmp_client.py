"""Tests for FMP client with mocked HTTP responses."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

from src.data.fmp_client import FMPClient, FMPDisabledError, FMPFatalError


@pytest.fixture
def client():
    with patch("src.data.fmp_client.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            fmp_api_key="test_key",
            fmp_daily_call_budget=750,
            fmp_budget_warn_threshold_pct=0.8,
            fmp_enforce_daily_budget=False,
        )
        return FMPClient()


@pytest.mark.asyncio
async def test_get_earnings_calendar(client):
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"symbol": "AAPL", "date": "2024-01-25", "eps": 2.10},
        {"symbol": "MSFT", "date": "2024-01-30", "eps": 2.93},
    ]
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await client.get_earnings_calendar(date(2024, 1, 1), date(2024, 2, 1))

    assert len(result) == 2
    assert result[0]["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_get_stock_screener(client):
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"symbol": "AAPL", "price": 195.0, "volume": 50000000, "exchangeShortName": "NASDAQ"},
        {"symbol": "MSFT", "price": 380.0, "volume": 30000000, "exchangeShortName": "NASDAQ"},
    ]
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await client.get_stock_screener()

    assert len(result) == 2


@pytest.mark.asyncio
async def test_get_insider_trading(client):
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"symbol": "AAPL", "transactionType": "P-Purchase", "securitiesTransacted": 1000, "price": 190},
    ]
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await client.get_insider_trading("AAPL")

    assert len(result) == 1


@pytest.mark.asyncio
async def test_get_daily_prices(client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "historical": [
            {"date": "2024-01-02", "open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000000},
            {"date": "2024-01-03", "open": 103, "high": 107, "low": 102, "close": 106, "volume": 1200000},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        df = await client.get_daily_prices("AAPL", date(2024, 1, 1), date(2024, 1, 5))

    assert not df.empty
    assert len(df) == 2
    assert "close" in df.columns


@pytest.mark.asyncio
async def test_get_analyst_estimates(client):
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"estimatedEpsAvg": 2.1, "estimatedRevenueAvg": 100_000_000},
    ]
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await client.get_analyst_estimates("AAPL")

    assert len(result) == 1
    assert result[0]["estimatedEpsAvg"] == 2.1
    _, kwargs = mock_client.get.call_args
    assert kwargs["params"]["period"] == "quarter"


@pytest.mark.asyncio
async def test_get_ratios(client):
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"priceEarningsRatio": 15.2, "priceToBookRatio": 2.0, "debtEquityRatio": 0.5},
    ]
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await client.get_ratios("AAPL")

    assert isinstance(result, dict)
    assert result["priceEarningsRatio"] == 15.2


@pytest.mark.asyncio
async def test_get_stock_news_bulk(client):
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"symbol": "AAPL", "title": "AAPL headline"},
        {"symbol": "MSFT", "title": "MSFT headline"},
    ]
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await client.get_stock_news_bulk(["AAPL", "MSFT"], limit=50)

    assert len(result) == 2
    assert result[0]["symbol"] == "AAPL"


def test_get_endpoint_status_defaults(client):
    status = client.get_endpoint_status()
    assert status["client_enabled"] is True
    assert status["endpoints"]["analyst_estimates"] == "supported"
    assert status["endpoints"]["stock_news_bulk_v3"] == "supported"
    assert status["endpoints"]["stock_news_bulk_stable"] == "supported"


@pytest.mark.asyncio
async def test_stock_news_bulk_legacy_403_does_not_disable_client(client):
    stable_response = MagicMock()
    stable_response.json.return_value = [{"symbol": "AAPL", "title": "fallback"}]
    stable_response.raise_for_status = MagicMock()

    with (
        patch("httpx.AsyncClient") as mock_client_cls,
        patch("src.data.fmp_client._request_with_backoff", side_effect=FMPFatalError(403, "Legacy Endpoint")),
    ):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        client._request = AsyncMock(return_value=stable_response)
        result = await client.get_stock_news_bulk(["AAPL"], limit=20)

    assert len(result) == 1
    assert client._disabled_reason is None
    assert client.get_endpoint_status()["endpoints"]["stock_news_bulk_v3"] == "plan_gated"


@pytest.mark.asyncio
async def test_analyst_estimates_400_disables_only_endpoint(client):
    bad_response = MagicMock()
    bad_response.status_code = 400
    bad_response.headers = {}
    bad_response.json.return_value = {"error": "bad request"}
    bad_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "bad request",
        request=httpx.Request("GET", "https://financialmodelingprep.com/stable/analyst-estimates"),
        response=httpx.Response(400, request=httpx.Request("GET", "https://financialmodelingprep.com/stable/analyst-estimates")),
    )

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = bad_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        first = await client.get_analyst_estimates("AAPL")
        second = await client.get_analyst_estimates("MSFT")

    assert first == []
    assert second == []
    assert client._analyst_estimates_enabled is False
    assert mock_client.get.call_count == 1


@pytest.mark.asyncio
async def test_analyst_estimates_402_disables_only_endpoint(client):
    with (
        patch("httpx.AsyncClient") as mock_client_cls,
        patch("src.data.fmp_client._request_with_backoff", side_effect=FMPFatalError(402, "plan gated")),
    ):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        first = await client.get_analyst_estimates("AAPL")
        second = await client.get_analyst_estimates("MSFT")

    assert first == []
    assert second == []
    assert client._analyst_estimates_enabled is False
    assert client._disabled_reason is None
    assert client.get_endpoint_status()["endpoints"]["analyst_estimates"] == "plan_gated"


@pytest.mark.asyncio
async def test_fmp_disables_after_non_retryable_error(client):
    """402/401 class errors should disable FMP for the process."""
    blocked = MagicMock()
    blocked.status_code = 402
    blocked.headers = {}
    blocked.json.return_value = {"Error Message": "Payment Required"}

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = blocked
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(FMPFatalError):
            await client.get_stock_screener()

        with pytest.raises(FMPDisabledError):
            await client.get_stock_screener()

    assert mock_client.get.call_count == 1
