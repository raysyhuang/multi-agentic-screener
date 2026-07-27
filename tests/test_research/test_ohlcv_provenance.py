"""OHLCV provenance tracking (2026-07-27).

A silent vendor fallback or silently-dropped tickers let a run be labelled
"Polygon-backed" while containing yfinance bars or a quietly shrunken universe.
That mixed provenance is what invalidates a backtest claim — the discipline was
adopted after a cross-agent review where the other checkout got this right and
this one did not. These tests pin the guarantees:

  * per-ticker Polygon failures are RECORDED with a reason, never swallowed
  * a provenance manifest travels with the cache
  * strict=True refuses to silently fall back to yfinance
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.research import signal_backtest as sb


def _fake_df():
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3).date,
        "open": [1.0, 1.0, 1.0], "high": [1.0, 1.0, 1.0],
        "low": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0], "volume": [10, 10, 10],
    })


class _Client:
    """Polygon stub: AAA succeeds, BBB raises, CCC returns empty."""

    async def get_ohlcv(self, sym, start, end):
        if sym == "AAA":
            return _fake_df()
        if sym == "BBB":
            raise ConnectionError("boom")
        return pd.DataFrame()


@pytest.fixture
def _stub(monkeypatch, tmp_path):
    import src.data.polygon_client as pc

    monkeypatch.setattr(pc, "PolygonClient", lambda *a, **k: _Client())
    monkeypatch.setattr(sb, "_cache_key", lambda t, y: tmp_path / "ohlcv_test.parquet")
    return tmp_path


def test_polygon_failures_are_recorded_not_swallowed(_stub):
    out = sb.fetch_ohlcv_polygon(["AAA", "BBB", "CCC"], years=1.0, no_cache=True)
    assert set(out) == {"AAA"}

    prov = sb.get_last_ohlcv_provenance()
    assert prov["provider"] == "polygon"
    assert prov["requested"] == 3 and prov["returned"] == 1
    assert sorted(prov["missing"]) == ["BBB", "CCC"]
    # Each failure carries a REASON — that is the whole point.
    assert "ConnectionError" in prov["failures"]["BBB"]
    assert prov["failures"]["CCC"] == "empty_or_malformed_response"


def test_provenance_manifest_is_written_next_to_the_cache(_stub):
    sb.fetch_ohlcv_polygon(["AAA", "BBB"], years=1.0, no_cache=True)
    manifest = _stub / "ohlcv_poly_test.provenance.json"
    assert manifest.exists(), "cached dataset must be self-describing"
    data = json.loads(manifest.read_text())
    assert data["provider"] == "polygon"
    assert data["missing"] == ["BBB"]
    assert "window" in data


def test_cache_hit_restores_provenance_rather_than_faking_a_clean_fetch(_stub):
    sb.fetch_ohlcv_polygon(["AAA", "BBB"], years=1.0, no_cache=True)
    sb._LAST_PROVENANCE = {}  # simulate a fresh process
    sb.fetch_ohlcv_polygon(["AAA", "BBB"], years=1.0)  # cache hit
    prov = sb.get_last_ohlcv_provenance()
    assert prov["from_cache"] is True
    assert prov["missing"] == ["BBB"], "cache hit must not hide the earlier failure"


def test_strict_refuses_silent_yfinance_fallback(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("polygon down")

    monkeypatch.setattr(sb, "fetch_ohlcv_polygon", _boom)
    with pytest.raises(RuntimeError, match="strict=True"):
        sb.fetch_ohlcv(["AAA"], years=1.0, source="polygon", strict=True)
