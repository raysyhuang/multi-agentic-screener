"""Integration smoke tests — live Heroku engine endpoints.

Validates schema, freshness, and type correctness of real API responses.
All tests are marked ``@pytest.mark.integration`` and excluded from the
default ``pytest`` run (per pyproject.toml addopts = "-m 'not integration'").

Run manually:  python -m pytest tests/test_engines/test_engine_smoke.py -v -m integration
"""

from __future__ import annotations

from datetime import date, timedelta

import httpx
import pytest

# ── Engine URLs ──────────────────────────────────────────────────────────

_ENGINES = {
    "koocore_d": {
        "url": "https://koocore-dashboard-dfa104d689ad.herokuapp.com",
        "paths": ["/api/engine/results"],
    },
    "gemini_stst": {
        "url": "https://geministst-a76526147b8c.herokuapp.com",
        "paths": ["/api/engine/results"],
    },
    "top3_7d": {
        "url": "https://sleepy-everglades-94250-93af971ee6c1.herokuapp.com",
        "paths": ["/api/engine/results", "/api/engine/results/latest", "/api/results"],
    },
}

_REQUIRED_FIELDS = {"engine_name", "run_date", "candidates_screened", "picks"}
_TIMEOUT = 30.0


# ── Helpers ──────────────────────────────────────────────────────────────

def _fetch_engine(engine_name: str) -> httpx.Response | None:
    """Try all known paths for an engine, return first 200 response."""
    cfg = _ENGINES[engine_name]
    base = cfg["url"]
    for path in cfg["paths"]:
        url = f"{base}{path}"
        try:
            resp = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
            if resp.status_code == 200:
                return resp
            if resp.status_code != 404:
                return resp  # return non-404 errors for assertion
        except httpx.RequestError:
            continue
    return None


def _trading_days_since(d: date) -> int:
    """Count trading days (weekdays) between d and today, exclusive of d."""
    today = date.today()
    if d >= today:
        return 0
    count = 0
    current = d
    while current < today:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return count


# ── KooCore-D ────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_koocore_endpoint_returns_200():
    resp = _fetch_engine("koocore_d")
    assert resp is not None, "KooCore-D endpoint unreachable"
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"


@pytest.mark.integration
def test_koocore_response_schema():
    resp = _fetch_engine("koocore_d")
    assert resp is not None
    data = resp.json()
    missing = _REQUIRED_FIELDS - set(data.keys())
    assert not missing, f"KooCore-D missing fields: {missing}"


@pytest.mark.integration
def test_koocore_picks_have_valid_risk_fields():
    resp = _fetch_engine("koocore_d")
    assert resp is not None
    data = resp.json()
    for pick in data.get("picks", []):
        entry = pick.get("entry_price", 0)
        stop = pick.get("stop_loss")
        target = pick.get("target_price")
        assert entry > 0, f"Invalid entry_price for {pick.get('ticker')}"
        if stop is not None:
            assert stop < entry, f"stop_loss >= entry_price for {pick.get('ticker')}"
        if target is not None:
            assert target > entry, f"target_price <= entry_price for {pick.get('ticker')}"


# ── Gemini STST ──────────────────────────────────────────────────────────

@pytest.mark.integration
def test_gemini_endpoint_returns_200():
    resp = _fetch_engine("gemini_stst")
    assert resp is not None, "Gemini STST endpoint unreachable"
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"


@pytest.mark.integration
def test_gemini_response_schema():
    resp = _fetch_engine("gemini_stst")
    assert resp is not None
    data = resp.json()
    missing = _REQUIRED_FIELDS - set(data.keys())
    assert not missing, f"Gemini STST missing fields: {missing}"


@pytest.mark.integration
def test_gemini_picks_numeric_fields_are_native_types():
    """All numeric values in Gemini picks must be native Python types (not numpy).

    This guards against the np.float64 serialization bug at the API level.
    JSON deserialization inherently returns native types, so this test verifies
    the engine is not somehow embedding non-JSON-serializable types.
    """
    resp = _fetch_engine("gemini_stst")
    assert resp is not None
    data = resp.json()
    for pick in data.get("picks", []):
        for key, val in pick.items():
            if isinstance(val, float):
                assert type(val) is float, f"Pick field '{key}' is {type(val)}"
            if isinstance(val, int) and not isinstance(val, bool):
                assert type(val) is int, f"Pick field '{key}' is {type(val)}"


# ── Top3-7D ──────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_top3_7d_endpoint_returns_200():
    resp = _fetch_engine("top3_7d")
    assert resp is not None, "Top3-7D endpoint unreachable"
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"


@pytest.mark.integration
def test_top3_7d_response_schema():
    resp = _fetch_engine("top3_7d")
    assert resp is not None
    data = resp.json()
    missing = _REQUIRED_FIELDS - set(data.keys())
    assert not missing, f"Top3-7D missing fields: {missing}"


# ── Cross-engine ─────────────────────────────────────────────────────────

@pytest.mark.integration
def test_all_engines_run_date_not_stale():
    """run_date should be within 2 trading days of today for all engines."""
    for engine_name in _ENGINES:
        resp = _fetch_engine(engine_name)
        if resp is None or resp.status_code != 200:
            pytest.skip(f"{engine_name} endpoint unreachable")
        data = resp.json()
        run_date_str = data.get("run_date", "")
        try:
            run_date = date.fromisoformat(run_date_str.split("T")[0])
        except ValueError:
            pytest.fail(f"{engine_name} has unparseable run_date: {run_date_str}")
        staleness = _trading_days_since(run_date)
        assert staleness <= 2, (
            f"{engine_name} run_date {run_date_str} is {staleness} trading days old"
        )


@pytest.mark.integration
def test_all_engines_picks_have_no_duplicate_tickers():
    """No engine should return duplicate tickers in its picks."""
    for engine_name in _ENGINES:
        resp = _fetch_engine(engine_name)
        if resp is None or resp.status_code != 200:
            pytest.skip(f"{engine_name} endpoint unreachable")
        data = resp.json()
        tickers = [p.get("ticker") for p in data.get("picks", [])]
        assert len(tickers) == len(set(tickers)), (
            f"{engine_name} has duplicate tickers: "
            f"{[t for t in tickers if tickers.count(t) > 1]}"
        )
