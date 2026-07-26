"""Tests for FRED client retry behavior on transient failures."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.data.fred_client import (
    FREDClient,
    _FRED_RETRY_MAX_ATTEMPTS,
    _fetch_fred_observations,
)


def _build_response(status_code: int, observations: list | None = None) -> MagicMock:
    """Construct a mock httpx.Response with optional observations payload."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = {
        "observations": observations or [{"date": "2026-04-24", "value": "0.53"}]
    }
    if 400 <= status_code:
        # Real httpx raises HTTPStatusError on raise_for_status() for 4xx/5xx
        request = httpx.Request("GET", "https://api.stlouisfed.org/test")
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=request, response=resp
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


def _patch_async_client(responses_or_exceptions: list):
    """Build a patcher for httpx.AsyncClient that yields a sequence of
    responses (or raises exceptions) across successive .get() calls.
    """
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=responses_or_exceptions)

    mock_cls = MagicMock(return_value=mock_client)
    return patch("src.data.fred_client.httpx.AsyncClient", mock_cls), mock_client


@pytest.fixture(autouse=True)
def _no_sleep():
    """Skip the real backoff sleep so tests stay fast."""
    with patch("src.data.fred_client.asyncio.sleep", new=AsyncMock(return_value=None)):
        yield


@pytest.mark.asyncio
async def test_fetch_succeeds_first_attempt():
    """Happy path: 200 on first call, no retries triggered."""
    patcher, mock_client = _patch_async_client([_build_response(200)])
    with patcher:
        resp = await _fetch_fred_observations(
            "https://example/test", {"series_id": "T10Y2Y"}, "T10Y2Y"
        )
    assert resp.status_code == 200
    assert mock_client.get.call_count == 1


@pytest.mark.asyncio
async def test_fetch_retries_then_succeeds_on_5xx():
    """500 first call, 200 second call → success after one retry."""
    responses = [_build_response(500), _build_response(200)]
    patcher, mock_client = _patch_async_client(responses)
    with patcher:
        resp = await _fetch_fred_observations(
            "https://example/test", {"series_id": "T10Y2Y"}, "T10Y2Y"
        )
    assert resp.status_code == 200
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_fetch_retries_then_succeeds_on_network_error():
    """ConnectError first call, 200 second call → success after one retry."""
    responses = [
        httpx.ConnectError("connection refused"),
        _build_response(200),
    ]
    patcher, mock_client = _patch_async_client(responses)
    with patcher:
        resp = await _fetch_fred_observations(
            "https://example/test", {"series_id": "T10Y2Y"}, "T10Y2Y"
        )
    assert resp.status_code == 200
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_fetch_exhausts_retries_on_repeated_5xx():
    """502 on every attempt → final call raises HTTPStatusError."""
    responses = [_build_response(502)] * _FRED_RETRY_MAX_ATTEMPTS
    patcher, mock_client = _patch_async_client(responses)
    with patcher:
        with pytest.raises(httpx.HTTPStatusError):
            await _fetch_fred_observations(
                "https://example/test", {"series_id": "T10Y2Y"}, "T10Y2Y"
            )
    assert mock_client.get.call_count == _FRED_RETRY_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_fetch_does_not_retry_on_4xx():
    """404 → raised immediately, no retries."""
    patcher, mock_client = _patch_async_client([_build_response(404)])
    with patcher:
        with pytest.raises(httpx.HTTPStatusError):
            await _fetch_fred_observations(
                "https://example/test", {"series_id": "T10Y2Y"}, "T10Y2Y"
            )
    assert mock_client.get.call_count == 1


@pytest.mark.asyncio
async def test_get_series_uses_retry_path():
    """Regression: FREDClient.get_series raises after final failure,
    preserving fail-closed semantics for callers (the morning pipeline).
    """
    client = FREDClient(api_key="test_key")
    responses = [_build_response(503)] * _FRED_RETRY_MAX_ATTEMPTS
    patcher, mock_client = _patch_async_client(responses)
    with patcher:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_series(
                "T10Y2Y", from_date=date(2026, 4, 22), to_date=date(2026, 4, 27)
            )
    assert mock_client.get.call_count == _FRED_RETRY_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_macro_snapshot_hy_oas_boundary_lengths(monkeypatch):
    """Regression: hy_oas_chg20 uses iloc[-21], which needs 21 obs — a >=20 guard
    raised IndexError at exactly 20 obs and crashed the macro fetch."""
    import pandas as pd
    from datetime import date, timedelta

    client = FREDClient(api_key="test_key")

    def _series(n):
        d0 = date(2026, 6, 1)
        return pd.DataFrame({
            "date": [d0 + timedelta(days=i) for i in range(n)],
            "value": [3.0 + 0.01 * i for i in range(n)],
        })

    for n, expect_chg in ((20, False), (21, True), (5, False)):
        async def fake_get_series(series_id, from_date=None, to_date=None, _n=n):
            if series_id == "BAMLH0A0HYM2":
                return _series(_n)
            return _series(5)

        monkeypatch.setattr(client, "get_series", fake_get_series)
        snap = await client.get_macro_snapshot()  # must not raise at any length
        assert snap["hy_oas"] is not None
        assert (snap["hy_oas_chg20"] is not None) == expect_chg
        if n >= 20:
            assert snap["hy_oas_stress"] is not None
