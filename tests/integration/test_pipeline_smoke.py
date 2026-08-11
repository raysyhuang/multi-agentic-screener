"""Run the whole morning pipeline against stubbed providers and a real database.

Nothing else does. The unit suite covers components; every bug that reached
production this month lived in the wiring between them:

  * PR #64 called `aggregator.get_data_provenance()` ~900 lines after the
    aggregator was released, dereferencing None on every run. Five review rounds
    examined the provenance logic — which was correct in isolation — and CI had
    nothing to fail on, because no test executes the pipeline.
  * The alembic chain could not build a schema from empty for six months, for
    the same reason: nothing ran it.

This test injects a sentinel provenance at the aggregator boundary and asserts
it reaches the persisted governance artifact. That is the end-to-end link the
source-level guard in `test_pipeline_aggregator_lifetime.py` cannot prove: a
future edit could snapshot correctly and then persist `{}`, passing every unit
test while writing an empty record.

Requires PostgreSQL (the models use JSONB) with the schema migrated, so it is
marked `integration` and excluded from the default run.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.db.models import DailyRun, PipelineArtifact
from src.db.session import get_session

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

# Injected at the aggregator boundary; asserted in the persisted payload.
SENTINEL_PROVENANCE = {
    "ohlcv_by_source": {"smoke-test-provider": 42},
    "ohlcv_cache_hits": 7,
    "ohlcv_failed_tickers": ["SENTINEL"],
    "universe_source": "smoke-test-universe",
    "universe_cache_hit": False,
    "universe_errors": [],
    "macro_source": "live",
    "macro_cache_hit": False,
    "circuits_opened_during_run": ["smoke-test-circuit"],
}

_TICKERS = [
    "AAAA", "BBBB", "CCCC", "DDDD", "EEEE",
    "FFFF", "GGGG", "HHHH", "IIII", "JJJJ",
]


def _bars(seed: int, days: int = 300) -> pd.DataFrame:
    """A plausible daily series: gentle drift plus noise, deterministic."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0006, 0.02, days))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.01, days)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.01, days)))
    open_ = (high + low) / 2.0
    start = date.today() - timedelta(days=days)
    return pd.DataFrame({
        "date": [start + timedelta(days=i) for i in range(days)],
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.integers(2_000_000, 20_000_000, days).astype(float),
    })


class FakeAggregator:
    """Stands in for DataAggregator across the whole pipeline surface.

    Every method the pipeline calls is implemented — that list is enumerable
    with `grep -o 'aggregator\\.[a-z_]*(' src/main.py`, and a missing one fails
    this test loudly rather than silently skipping a stage.
    """

    def __init__(self) -> None:
        self.closed = False
        self.provenance_reset = False
        self._bars = {t: _bars(seed=i) for i, t in enumerate(_TICKERS)}

    async def get_universe(self) -> list[dict]:
        return [
            {
                "symbol": t,
                "price": 100.0,
                "volume": 5_000_000,
                "marketCap": 5_000_000_000,
                "exchangeShortName": "NASDAQ",
                "isEtf": False,
                "isFund": False,
            }
            for t in _TICKERS
        ]

    async def get_bulk_ohlcv(self, tickers, from_date, to_date, **kw) -> dict:
        return {t: self._bars.get(t, pd.DataFrame()) for t in tickers}

    async def get_macro_context(self) -> dict:
        return {
            "vix": 15.0,
            "yield_10y": 4.1,
            "spy_prices": _bars(seed=901),
            "qqq_prices": _bars(seed=902),
        }

    async def get_ticker_fundamentals(self, ticker: str) -> dict:
        return {
            "earnings_surprises": [],
            "insider_transactions": [],
            "profile": {"symbol": ticker},
            "analyst_estimates": [],
            "ratios": {},
        }

    async def get_upcoming_earnings(self, days_ahead: int = 14) -> list[dict]:
        return []

    def get_fmp_budget_status(self) -> dict:
        return {
            "date": str(date.today()), "calls_used": 0, "daily_budget": 750,
            "calls_remaining": 750, "used_pct": 0.0,
        }

    def get_fmp_endpoint_status(self) -> dict:
        return {"endpoints": {}, "health_checked_endpoints": []}

    def get_data_provenance(self) -> dict:
        # Refuses once released, mirroring the real lifetime. Returning the
        # sentinel here regardless would let the test pass on code that reads
        # provenance AFTER teardown — the #64 defect itself — proving only that
        # the value propagated, not that it was captured while it could be.
        if self.closed:
            raise AssertionError(
                "get_data_provenance() called after close(); the real aggregator "
                "is None by this point and this is exactly how the 2026-08-11 "
                "outage happened"
            )
        return dict(SENTINEL_PROVENANCE)

    def reset_data_provenance(self) -> None:
        self.provenance_reset = True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def pinned_run_id(monkeypatch, request) -> str:
    """Force a per-test run_id so assertions target this invocation only.

    `run_morning_pipeline` generates its own uuid, so without this every query
    has to guess ("the newest row") — and a run that wrote nothing is then
    indistinguishable from one that wrote correctly, because an earlier test's
    rows satisfy the query anyway. Both the failure test and the provenance test
    had that hole: the provenance assertion was passing against the artifact
    written by the FIRST test.

    Derived from the test name, so each test gets a distinct stable id and their
    rows cannot be mistaken for one another.
    """
    import hashlib
    import uuid

    digest = hashlib.sha256(request.node.name.encode()).digest()
    fixed = uuid.UUID(bytes=digest[:16])
    monkeypatch.setattr(uuid, "uuid4", lambda: fixed)
    return fixed.hex[:12]


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    """Patch every outbound edge: providers, alerts, benchmark fetches."""
    from src import main as main_mod

    fake = FakeAggregator()
    monkeypatch.setattr(main_mod, "DataAggregator", lambda: fake)
    # Telegram is already a no-op with empty credentials, but make it explicit
    # so a configured environment cannot make this test post to a real chat.
    monkeypatch.setattr(main_mod, "send_alert", AsyncMock(return_value=False))
    return fake


async def test_pipeline_runs_end_to_end(stubbed_pipeline, pinned_run_id) -> None:
    """The whole thing executes without an unhandled exception.

    This alone would have caught the #64 outage: the pipeline fails closed, so
    the `None` dereference surfaced as a NoTrade rather than a crash, and only
    an end-to-end execution reaches that line at all.
    """
    from src.main import run_morning_pipeline

    await run_morning_pipeline()

    assert stubbed_pipeline.closed, "the aggregator should be released after Step 4"
    assert stubbed_pipeline.provenance_reset, "provenance should be reset at run start"

    async with get_session() as session:
        artifacts = (await session.execute(
            select(PipelineArtifact).where(PipelineArtifact.run_id == pinned_run_id)
        )).scalars().all()
        run = (await session.execute(
            select(DailyRun).where(DailyRun.run_date == date.today())
        )).scalar_one_or_none()

    assert artifacts, f"run {pinned_run_id} persisted no artifacts"
    assert run is not None, "the run must persist a DailyRun row"
    assert not any(a.status == "failed" for a in artifacts), (
        f"a clean run wrote failed artifacts: "
        f"{[(a.stage, a.status) for a in artifacts if a.status == 'failed']}"
    )


async def test_injected_provenance_reaches_the_governance_artifact(
    stubbed_pipeline, pinned_run_id
) -> None:
    """The link the source-level guard cannot prove.

    A future edit could snapshot provenance correctly before teardown and then
    persist something else entirely — `provenance = {}` between the two lines
    passes every unit test we have. Only following a sentinel from the
    aggregator boundary into the stored payload rules that out.
    """
    from src.main import run_morning_pipeline

    await run_morning_pipeline()

    async with get_session() as session:
        artifact = (await session.execute(
            select(PipelineArtifact).where(
                PipelineArtifact.run_id == pinned_run_id,
                PipelineArtifact.stage == "governance",
            )
        )).scalars().first()

    assert artifact is not None, (
        f"run {pinned_run_id} persisted no governance artifact of its own"
    )
    payload = artifact.payload or {}

    assert payload.get("data_provenance") == SENTINEL_PROVENANCE, (
        "the provenance captured from the aggregator did not reach the stored "
        f"governance record; got {payload.get('data_provenance')!r}"
    )
    # The funnels travel the same path and are the other half of #64's purpose.
    assert payload.get("universe_funnel"), "universe funnel must be persisted"
    assert payload.get("ohlcv_funnel"), "ohlcv funnel must be persisted"


async def test_a_failed_run_records_its_own_failure(
    monkeypatch, stubbed_pipeline, pinned_run_id
) -> None:
    """Fail-closed must be verified on THIS invocation, not on any row existing.

    Asserting merely that some DailyRun exists is far too weak: the successful
    smoke tests above already create one, so an invocation that persisted
    nothing at all would still pass. The run id is pinned so every assertion
    below is about this run and no other.

    The #64 outage produced exactly this shape — an exception deep in the run
    becoming a NoTrade — so the harness must reproduce it deliberately.
    """
    from src import main as main_mod

    async def boom(*a, **k):
        raise RuntimeError("smoke-test induced failure")

    monkeypatch.setattr(stubbed_pipeline, "get_universe", boom)

    await main_mod.run_morning_pipeline()  # fail-closed: must not raise

    async with get_session() as session:
        artifacts = (await session.execute(
            select(PipelineArtifact).where(PipelineArtifact.run_id == pinned_run_id)
        )).scalars().all()
        run = (await session.execute(
            select(DailyRun).where(DailyRun.run_date == date.today())
        )).scalar_one_or_none()

    assert artifacts, f"run {pinned_run_id} persisted nothing at all"

    final = [a for a in artifacts if a.stage == "final_output"]
    assert final, "a failed run must persist a final_output artifact"
    assert final[0].status == "failed", f"expected failed, got {final[0].status!r}"

    codes = {e.get("code") for e in (final[0].errors or [])}
    assert "PIPELINE_CRASH" in codes, f"crash marker missing; got {codes!r}"
    assert final[0].payload.get("decision") == "NoTrade"

    assert run is not None, "the failed run must still leave a DailyRun row"
    assert run.regime == "unknown", "a crashed run must not claim a regime"
    assert "smoke-test induced failure" in str(run.regime_details)
