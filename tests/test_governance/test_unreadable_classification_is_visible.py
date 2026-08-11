"""An admitted-but-unclassified row must be visible outside the log file.

The ETF gate deliberately fails open: when a row's isEtf/isFund classification
arrives in a shape it cannot read, the row is admitted rather than dropped,
because failing closed on an unknown provider encoding would cost the entire
day's universe. That trade is only defensible while the condition is visible.

A count that reaches nothing but a `logger.warning` is not visible in any
operational sense — a provider schema drift could admit ETFs, ETNs and funds for
weeks and be noticed only if someone happened to read the right run's log. These
tests pin the two surfaces that make it observable: pipeline health (which
reaches the dashboard) and the governance artifact persisted to
`pipeline_artifacts.payload`.
"""

from __future__ import annotations

import json

from src.governance.artifacts import GovernanceContext
from src.signals.filter import FilterFunnel, filter_universe
from src.validation.stage_validator import Severity, validate_universe

# FMP shape — no `type` key — with a classification the gate cannot read.
UNREADABLE = {
    "symbol": "XXXX",
    "price": 50.0,
    "volume": 5_000_000,
    "exchangeShortName": "NASDAQ",
    "isEtf": "maybe",
    "isFund": None,
}
CLEAN = {
    "symbol": "AAPL",
    "price": 220.0,
    "volume": 50_000_000,
    "exchangeShortName": "NASDAQ",
    "isEtf": False,
    "isFund": False,
}


def _funnel_for(rows: list[dict]) -> FilterFunnel:
    funnel = FilterFunnel()
    filter_universe(rows, funnel=funnel)
    return funnel


def test_unreadable_classification_raises_a_health_warning() -> None:
    funnel = _funnel_for([UNREADABLE, CLEAN])
    assert funnel.unrecognized_type_flags == 1

    sv = validate_universe(
        raw_count=2,
        filtered_count=funnel.passed,
        filtered=[],
        unrecognized_type_flags=funnel.unrecognized_type_flags,
    )

    check = next(c for c in sv.checks if c.name == "security_type_classification")
    assert not check.passed
    assert check.severity is Severity.WARN, "must not block the run"
    assert check.value == 1
    assert "admitted unchecked" in check.message


def test_a_clean_run_passes_the_check() -> None:
    funnel = _funnel_for([CLEAN])
    sv = validate_universe(
        raw_count=1,
        filtered_count=funnel.passed,
        filtered=[],
        unrecognized_type_flags=funnel.unrecognized_type_flags,
    )

    check = next(c for c in sv.checks if c.name == "security_type_classification")
    assert check.passed
    assert check.value == 0


def test_the_warning_is_never_severe_enough_to_stop_the_book() -> None:
    """Failing closed here would hand a provider hiccup the power to halt the book."""
    sv = validate_universe(
        raw_count=500, filtered_count=400, filtered=[], unrecognized_type_flags=400
    )
    check = next(c for c in sv.checks if c.name == "security_type_classification")
    assert check.severity is Severity.WARN


def test_the_count_reaches_the_persisted_governance_artifact() -> None:
    """The other half: it must survive into pipeline_artifacts.payload."""
    funnel = _funnel_for([UNREADABLE, CLEAN])

    with GovernanceContext(run_id="r1", run_date="2026-08-11") as gov:
        gov.set_funnels(universe=funnel)

    payload = gov.record.to_dict()
    assert payload["universe_funnel"]["unrecognized_type_flags"] == 1
    # And the payload is written to a JSONB column.
    assert json.loads(json.dumps(payload))["universe_funnel"]["unrecognized_type_flags"] == 1
