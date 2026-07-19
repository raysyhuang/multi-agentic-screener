"""Tests for Polygon.io client with mocked HTTP responses."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.data.polygon_client import PolygonClient


@pytest.fixture
def client():
    with patch("src.data.polygon_client.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(polygon_api_key="test_key")
        return PolygonClient()


@pytest.mark.asyncio
async def test_get_ohlcv_parses_response(client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"o": 100, "h": 105, "l": 99, "c": 103, "v": 1000000, "vw": 102, "t": 1704067200000, "n": 5000},
            {"o": 103, "h": 107, "l": 102, "c": 106, "v": 1200000, "vw": 104, "t": 1704153600000, "n": 6000},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        df = await client.get_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 2))

    assert not df.empty
    assert "open" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns
    assert "date" in df.columns
    assert len(df) == 2


@pytest.mark.asyncio
async def test_get_ohlcv_empty_results(client):
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        df = await client.get_ohlcv("FAKE", date(2024, 1, 1), date(2024, 1, 2))

    assert df.empty


@pytest.mark.asyncio
async def test_get_news_parses_response(client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"title": "AAPL beats earnings", "published_utc": "2024-01-01T10:00:00Z"},
            {"title": "AAPL announces buyback", "published_utc": "2024-01-02T10:00:00Z"},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        news = await client.get_news("AAPL")

    assert len(news) == 2
    assert news[0]["title"] == "AAPL beats earnings"


@pytest.mark.asyncio
async def test_get_intraday_aggs_preserves_datetime(client):
    """Intraday bars must keep the intraday time (many rows per date)."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"o": 100, "h": 101, "l": 99, "c": 100.5, "v": 5000, "vw": 100.2, "t": 1704114000000, "n": 50},
            {"o": 100.5, "h": 102, "l": 100, "c": 101.5, "v": 6000, "vw": 101.1, "t": 1704114060000, "n": 60},
        ],
        "next_url": None,
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        df = await client.get_intraday_aggs("AAPL", date(2024, 1, 1), date(2024, 1, 1))

    assert not df.empty
    assert "datetime" in df.columns and "date" in df.columns
    assert len(df) == 2
    # Two distinct minute timestamps on the same date
    assert df["datetime"].nunique() == 2
    assert df["date"].nunique() == 1


@pytest.mark.asyncio
async def test_get_intraday_aggs_paginates(client):
    """next_url must be followed until exhausted, accumulating all bars."""
    page1 = MagicMock()
    page1.json.return_value = {
        "results": [{"o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "vw": 1, "t": 1704114000000, "n": 1}],
        "next_url": "https://api.polygon.io/next?cursor=abc",
    }
    page1.raise_for_status = MagicMock()
    page2 = MagicMock()
    page2.json.return_value = {
        "results": [{"o": 2, "h": 2, "l": 2, "c": 2, "v": 2, "vw": 2, "t": 1704114060000, "n": 2}],
        "next_url": None,
    }
    page2.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.side_effect = [page1, page2]
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        df = await client.get_intraday_aggs("AAPL", date(2024, 1, 1), date(2024, 1, 1))

    assert len(df) == 2  # both pages accumulated
    assert mock_client.get.call_count == 2
