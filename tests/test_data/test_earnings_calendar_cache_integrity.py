"""A cache entry is untrusted input, exercised against the real DataAggregator.

`get_upcoming_earnings` decoded cached JSON unguarded, and set
`last_earnings_fetch_error = None` BEFORE decoding. A corrupt row therefore
raised straight out of the method — crashing the run while reporting a healthy
feed — which defeats the whole point of separating:

    valid empty calendar  |  provider failure  |  unhealthy calendar input

These drive the actual object rather than a mirror of its predicate, because the
defect lived in the cache path, not in the health-stage arithmetic.
"""

from __future__ import annotations

import json

import pytest

from src.data.aggregator import DataAggregator

pytestmark = pytest.mark.asyncio


@pytest.fixture
def agg(monkeypatch):
    a = DataAggregator()
    monkeypatch.setattr(a, "_cache_enabled", True)
    return a


def _stub_cache(monkeypatch, agg, value):
    monkeypatch.setattr(agg._cache, "get", lambda key: value)
    monkeypatch.setattr(agg._cache, "put", lambda *a, **k: None)


def _stub_provider(monkeypatch, agg, result):
    async def fake(from_date, to_date):
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(agg.fmp, "get_earnings_calendar", fake)


async def test_malformed_cached_json_does_not_raise_and_refetches(agg, monkeypatch):
    """The reported defect: corrupt cache crashed the run mid-pipeline."""
    _stub_cache(monkeypatch, agg, "{not json at all")
    _stub_provider(monkeypatch, agg, [{"symbol": "AAPL", "date": "2026-08-20"}])

    out = await agg.get_upcoming_earnings()

    assert out == [{"symbol": "AAPL", "date": "2026-08-20"}], "should degrade to a live fetch"
    assert agg.last_earnings_fetch_error is None, "a successful refetch is healthy"


async def test_cached_json_of_the_wrong_shape_is_treated_as_a_miss(agg, monkeypatch):
    """A decoded dict sails past a bare json.loads and fails much later."""
    _stub_cache(monkeypatch, agg, json.dumps({"symbol": "AAPL"}))
    _stub_provider(monkeypatch, agg, [])

    out = await agg.get_upcoming_earnings()

    assert out == []
    assert agg.last_earnings_fetch_error is None


async def test_malformed_cache_plus_failing_provider_reports_the_failure(agg, monkeypatch):
    """Degrading to a miss must not hide a provider that is also down."""
    _stub_cache(monkeypatch, agg, "<<<corrupt>>>")
    _stub_provider(monkeypatch, agg, RuntimeError("boom"))

    out = await agg.get_upcoming_earnings()

    assert out == []
    assert agg.last_earnings_fetch_error == "RuntimeError", (
        "a crash after a bad cache must surface as a provider failure, not health"
    )


async def test_a_provider_payload_of_the_wrong_type_is_not_reported_healthy(agg, monkeypatch):
    """Not a valid empty calendar — must not read as a working feed."""
    _stub_cache(monkeypatch, agg, None)
    _stub_provider(monkeypatch, agg, {"unexpected": "dict"})

    out = await agg.get_upcoming_earnings()

    assert out == []
    assert agg.last_earnings_fetch_error == "MalformedProviderPayload"


async def test_a_valid_cached_calendar_is_used_and_reported_healthy(agg, monkeypatch):
    """The working path still works — guards must not break the normal case."""
    rows = [{"symbol": "MSFT", "date": "2026-08-18"}]
    _stub_cache(monkeypatch, agg, json.dumps(rows))

    async def explode(*a, **k):
        raise AssertionError("provider must not be called when the cache is valid")
    monkeypatch.setattr(agg.fmp, "get_earnings_calendar", explode)

    out = await agg.get_upcoming_earnings()

    assert out == rows
    assert agg.last_earnings_fetch_error is None


async def test_a_valid_empty_cached_calendar_is_distinguishable_from_failure(agg, monkeypatch):
    """An empty list is a real answer: healthy fetch, zero rows. The health
    stage flags it on row count, NOT by pretending the fetch failed."""
    _stub_cache(monkeypatch, agg, json.dumps([]))

    out = await agg.get_upcoming_earnings()

    assert out == []
    assert agg.last_earnings_fetch_error is None
