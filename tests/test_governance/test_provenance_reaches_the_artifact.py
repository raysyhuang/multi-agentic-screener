"""Gate attribution and provenance must survive all the way to the stored payload.

The funnels were always computed — they were logged and dropped. A counter that
reaches a log line and not the artifact is only marginally better than no
counter, because answering "why did this run pick nothing" then still means
locating that run's log output. These tests pin the serialization boundary the
data has to cross: `GovernanceRecord.to_dict()`, which is what
`main.py` writes into `pipeline_artifacts.payload`.

Round-trip equality is asserted against the funnel's own `to_dict()` rather than
a fixed set of keys, so a counter added to a funnel later reaches the artifact
without anyone having to remember this file. That is deliberate: the ETF gate's
`unrecognized_type_flags` (PR #63) is exactly such a counter.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields

from src.governance.artifacts import GovernanceContext
from src.signals.filter import FilterFunnel, OHLCVFunnel, filter_universe

SPY_ETF = {
    "symbol": "SPY",
    "price": 500.0,
    "volume": 70_000_000,
    "exchangeShortName": "NYSE",
    "type": "ETF",
}
AAPL = {
    "symbol": "AAPL",
    "price": 220.0,
    "volume": 50_000_000,
    "exchangeShortName": "NASDAQ",
    "type": "",
}

# A realistic run's provenance. Its key set is pinned against the aggregator's
# real output below, so this cannot quietly rot into a shape nothing produces.
PROVENANCE = {
    "ohlcv_by_source": {"polygon": 800, "yfinance": 41},
    "ohlcv_cache_hits": 141,
    "ohlcv_failed_tickers": ["ZZZZ"],
    "universe_source": "fmp",
    "universe_cache_hit": False,
    "universe_errors": [],
    "macro_source": "live",
    "macro_cache_hit": False,
    "circuits_opened_during_run": ["polygon"],
}


def test_etf_rejection_reaches_the_persisted_record() -> None:
    funnel = FilterFunnel()
    passed = filter_universe([SPY_ETF, AAPL], funnel=funnel)
    assert [s["symbol"] for s in passed] == ["AAPL"]

    with GovernanceContext(run_id="r1", run_date="2026-08-11") as gov:
        gov.set_funnels(universe=funnel)

    payload = gov.record.to_dict()
    assert payload["universe_funnel"]["failed_type"] == 1
    assert payload["universe_funnel"]["passed"] == 1


def test_every_dataclass_field_reaches_the_payload() -> None:
    """Compared against `asdict`, deliberately — not the funnel's own to_dict.

    Comparing to `funnel.to_dict()` would be tautological when the persistence
    path also calls `to_dict()`: a counter added to the dataclass and forgotten
    in that hand-written method would be missing from both sides and the test
    would still pass, while the artifact silently lost the field. `asdict`
    enumerates the dataclass itself, so omission fails here.
    """
    funnel = FilterFunnel()
    filter_universe([SPY_ETF, AAPL], funnel=funnel)

    with GovernanceContext(run_id="r2", run_date="2026-08-11") as gov:
        gov.set_funnels(universe=funnel, ohlcv=OHLCVFunnel(total_input=9, passed=4))

    payload = gov.record.to_dict()
    assert payload["universe_funnel"] == asdict(funnel)
    assert set(payload["universe_funnel"]) == {f.name for f in fields(FilterFunnel)}
    assert payload["ohlcv_funnel"] == asdict(OHLCVFunnel(total_input=9, passed=4))


def test_the_provenance_fixture_matches_the_real_shape() -> None:
    """Hand-written fixtures drift from the thing they stand in for.

    `circuits_open` was renamed to `circuits_opened_during_run` when it stopped
    being a point-in-time sample, and the fixtures below kept the retired key —
    passing happily while asserting against a shape the aggregator no longer
    produces. Pin the fixture to the real one so the next rename cannot leave a
    fossil behind.
    """
    from src.data.aggregator import DataAggregator

    real = DataAggregator().get_data_provenance()
    assert set(PROVENANCE) == set(real), (
        "fixture keys have drifted from get_data_provenance(); "
        f"missing={set(real) - set(PROVENANCE)}, stale={set(PROVENANCE) - set(real)}"
    )


def test_data_provenance_reaches_the_persisted_record() -> None:
    with GovernanceContext(run_id="r3", run_date="2026-08-11") as gov:
        gov.set_data_provenance(PROVENANCE)

    assert gov.record.to_dict()["data_provenance"] == PROVENANCE


def test_the_payload_is_json_serializable() -> None:
    """It is written to a JSONB column — a non-serializable value fails the run."""
    funnel = FilterFunnel()
    filter_universe([SPY_ETF, AAPL], funnel=funnel)

    with GovernanceContext(run_id="r4", run_date="2026-08-11") as gov:
        gov.set_funnels(universe=funnel, ohlcv=OHLCVFunnel())
        gov.set_data_provenance(
            {"ohlcv_by_source": {"polygon": 1}, "circuits_opened_during_run": []}
        )

    json.dumps(gov.record.to_dict())


def test_absent_provenance_defaults_to_empty_not_missing() -> None:
    """A run that never set them must still produce the keys, not KeyError."""
    with GovernanceContext(run_id="r5", run_date="2026-08-11") as gov:
        pass

    payload = gov.record.to_dict()
    assert payload["data_provenance"] == {}
    assert payload["universe_funnel"] == {}
    assert payload["ohlcv_funnel"] == {}
