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
