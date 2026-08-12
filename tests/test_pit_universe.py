"""Adversarial PIT fixtures — contract §8, the acceptance gate.

Each test constructs a tiny synthetic vintage and asserts the loader does NOT
leak the future. They are written to fail on the specific mistakes the contract
was written to prevent, not to confirm that a correct build looks correct:

  1. a ticker eligible on D+1 but not D is absent from D
  2. a ticker delisted at D+5 is PRESENT at D — survivorship
  3. a market-cap crossing at D+3 does not qualify the name at D  (Phase B)
  4. bars after D are unreachable when constructing D
  5. an alias introduced later does not retro-resolve at D
  6. a classification known only today does not backfill onto D
  7. a UTC-evening timestamp maps to the correct ET market date

Fixture 3 is asserted at the contract level here because Phase A does not apply
market cap at all — the constraint is Phase B's, and its absence from Phase A
output is itself the property worth pinning.
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pit_universe_phase_a as pit  # noqa: E402


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(json.dumps(payload).encode())


def _bar(ticker: str, close: float = 100.0, volume: float = 1_000_000) -> dict:
    return {"T": ticker, "c": close, "h": close * 1.02, "l": close * 0.98, "v": volume}


@pytest.fixture
def vintage(tmp_path, monkeypatch) -> str:
    """An isolated synthetic vintage rooted in tmp_path."""
    monkeypatch.setattr(pit, "ROOT", tmp_path)
    return "fixture"


def _seed(tmp_path, vintage: str, bars_by_date: dict[str, list[dict]],
          labels_by_month: dict[str, list[dict]]) -> None:
    for d, bars in bars_by_date.items():
        _write(tmp_path / vintage / "raw" / "grouped" / f"{d}.json.gz", {"results": bars})
    for month, rows in labels_by_month.items():
        _write(
            tmp_path / vintage / "raw" / "reference" / month / "page-1.json.gz",
            {"results": rows},
        )


def _ref(ticker: str, type_: str = "CS", exchange: str = "XNYS") -> dict:
    return {"ticker": ticker, "type": type_, "primary_exchange": exchange}


# ── 1. no future membership ──────────────────────────────────────────────────

def test_a_ticker_that_starts_trading_later_is_absent_earlier(tmp_path, vintage):
    _seed(
        tmp_path, vintage,
        {"2024-03-01": [_bar("AAA")], "2024-03-04": [_bar("AAA"), _bar("NEW")]},
        {"2024-03": [_ref("AAA"), _ref("NEW")]},
    )
    m = pit.build_membership(vintage)

    assert "NEW" not in m[date(2024, 3, 1)]["traded"], "a later arrival leaked backwards"
    assert "NEW" in m[date(2024, 3, 4)]["eligible_pre_mcap"]


# ── 2. survivorship ──────────────────────────────────────────────────────────

def test_a_ticker_delisted_later_is_still_present_before_it_delisted(tmp_path, vintage):
    """The failure that makes a backtest look better than reality."""
    _seed(
        tmp_path, vintage,
        {"2024-03-01": [_bar("AAA"), _bar("DEAD")], "2024-03-08": [_bar("AAA")]},
        {"2024-03": [_ref("AAA"), _ref("DEAD")]},
    )
    m = pit.build_membership(vintage)

    assert "DEAD" in m[date(2024, 3, 1)]["eligible_pre_mcap"], (
        "a name that traded on this date was dropped because it later delisted"
    )
    assert "DEAD" not in m[date(2024, 3, 8)]["traded"]


# ── 3. market cap is Phase B ─────────────────────────────────────────────────

def test_phase_a_does_not_apply_market_cap_at_all(tmp_path, vintage):
    """Phase A must not silently approximate the constraint it defers.

    A name below $300M is eligible in Phase A output because market cap has not
    been evaluated. Pinning this stops a future edit from quietly applying a
    current-value cap — which would be backfilled data, and worse than omission.
    """
    _seed(
        tmp_path, vintage,
        {"2024-03-01": [_bar("TINY", close=6.0, volume=900_000)]},
        {"2024-03": [_ref("TINY")]},
    )
    m = pit.build_membership(vintage)

    assert "TINY" in m[date(2024, 3, 1)]["eligible_pre_mcap"]
    assert "mcap" not in json.dumps(m[date(2024, 3, 1)]["exclusions"]).lower()


# ── 4. no future bars ────────────────────────────────────────────────────────

def test_membership_for_a_date_uses_only_that_date(tmp_path, vintage):
    """A name failing volume on D must not be rescued by D+1's volume."""
    _seed(
        tmp_path, vintage,
        {
            "2024-03-01": [_bar("THIN", volume=1_000)],       # fails on D
            "2024-03-04": [_bar("THIN", volume=5_000_000)],   # passes on D+1
        },
        {"2024-03": [_ref("THIN")]},
    )
    m = pit.build_membership(vintage)

    assert "THIN" not in m[date(2024, 3, 1)]["eligible_pre_mcap"]
    assert "THIN" in m[date(2024, 3, 4)]["eligible_pre_mcap"]


# ── 5 & 6. no backfilled labels ──────────────────────────────────────────────

def test_a_label_that_appears_later_does_not_backfill(tmp_path, vintage):
    """March has no label for LATE; April does. March must not borrow it."""
    _seed(
        tmp_path, vintage,
        {"2024-03-01": [_bar("LATE")], "2024-04-01": [_bar("LATE")]},
        {"2024-03": [], "2024-04": [_ref("LATE")]},
    )
    m = pit.build_membership(vintage)

    assert "LATE" not in m[date(2024, 3, 1)]["eligible_pre_mcap"]
    assert m[date(2024, 3, 1)]["exclusions"].get("type_unknown") == 1, (
        "an unresolvable label must be counted as unknown, not silently dropped"
    )
    assert "LATE" in m[date(2024, 4, 1)]["eligible_pre_mcap"]


def test_a_reclassification_does_not_apply_retroactively(tmp_path, vintage):
    """ETF in March, common stock in April: March stays excluded."""
    _seed(
        tmp_path, vintage,
        {"2024-03-01": [_bar("CONV")], "2024-04-01": [_bar("CONV")]},
        {"2024-03": [_ref("CONV", type_="ETF")], "2024-04": [_ref("CONV", type_="CS")]},
    )
    m = pit.build_membership(vintage)

    assert "CONV" not in m[date(2024, 3, 1)]["eligible_pre_mcap"]
    assert "CONV" in m[date(2024, 4, 1)]["eligible_pre_mcap"]


def test_the_forward_held_label_is_the_most_recent_prior_snapshot(tmp_path, vintage):
    """Mid-month dates use the month's snapshot — never the next one."""
    _seed(
        tmp_path, vintage,
        {"2024-03-20": [_bar("MID")]},
        {"2024-03": [_ref("MID", type_="ETF")], "2024-04": [_ref("MID", type_="CS")]},
    )
    m = pit.build_membership(vintage)

    assert "MID" not in m[date(2024, 3, 20)]["eligible_pre_mcap"], (
        "a mid-March date used April's label"
    )


# ── 7. ET boundary ───────────────────────────────────────────────────────────

def test_a_utc_evening_timestamp_maps_to_the_previous_et_market_date():
    """§0 — the error that produced a future-dated probe and a CI false-red."""
    utc_evening = datetime(2026, 8, 12, 0, 30, tzinfo=timezone.utc)
    et = utc_evening.astimezone(ZoneInfo("America/New_York")).date()

    assert et == date(2026, 8, 11), "a UTC-evening instant is the PREVIOUS ET date"


def test_the_session_range_never_includes_an_unstarted_session():
    """Today's bars do not exist until the session closes."""
    sessions = pit.et_sessions(0.05)

    assert sessions, "expected at least one session"
    assert sessions[-1] < pit._today_et(), (
        "the range reaches today's ET date, whose bars are absent or partial"
    )


# ── audit sampling determinism ───────────────────────────────────────────────

def test_the_audit_sample_is_reproducible(tmp_path, vintage):
    """Same vintage plus same seed and sampler version ⇒ identical pair set."""
    bars = {f"2024-03-{d:02d}": [_bar(f"T{i}") for i in range(40)] for d in (1, 4, 5)}
    _seed(tmp_path, vintage, bars, {"2024-03": [_ref(f"T{i}") for i in range(40)]})

    m = pit.build_membership(vintage)
    first = pit.audit_sample(vintage, m)
    second = pit.audit_sample(vintage, m)

    assert first == second, "the sampler is not deterministic across runs"


def test_the_audit_population_is_drawn_before_classification(tmp_path, vintage):
    """§3b — an ETF-labelled name must still be auditable.

    If the sample came from the eligible set, a name wrongly labelled ETF would
    be excluded before sampling and its false exclusion never detected.
    """
    _seed(
        tmp_path, vintage,
        {"2024-03-01": [_bar("STOCK"), _bar("LABELLED_ETF")]},
        {"2024-03": [_ref("STOCK"), _ref("LABELLED_ETF", type_="ETF")]},
    )
    m = pit.build_membership(vintage)
    sample = pit.audit_sample(vintage, m)

    sampled_tickers = {t for pairs in sample.values() for (_b, _d, t) in pairs}
    assert "LABELLED_ETF" in sampled_tickers, (
        "an ETF-labelled name was not auditable, so false exclusion is undetectable"
    )
    assert "STOCK" in sampled_tickers


# ── repairs demanded by the Phase A review ───────────────────────────────────

def _membership(monthly: list[tuple[int, int, int, int]]) -> dict:
    """[(year, month, type_unknown, exchange_unknown)] -> membership, 1000/month."""
    out = {}
    for y, m, tu, eu in monthly:
        out[date(y, m, 1)] = {
            "pre_classification": [f"T{i}" for i in range(1000)],
            "traded": [], "eligible_pre_mcap": [],
            "exclusions": {"type_unknown": tu, "exchange_unknown": eu},
        }
    return out


def _months(n: int, tu: int = 3, eu: int = 3) -> list[tuple[int, int, int, int]]:
    return [(2023 + i // 12, i % 12 + 1, tu, eu) for i in range(n)]


def test_an_elevated_type_month_halts_on_the_trailing_median_rule():
    """§A.5: monthly rate > 2x the trailing-12m MEDIAN for that metric.

    0.3% baseline for twelve months, then 0.7% — under the 1% absolute gate, so
    only the relative rule can catch it. This is the test my first
    implementation could not have passed: it compared a pooled trailing rate to
    a fixed 1% and would have stayed silent.
    """
    from pit_universe_report import unknown_rate_gates

    rows = _months(12, tu=3, eu=3) + [(2024, 1, 7, 3)]
    gates, halts = unknown_rate_gates(_membership(rows))

    assert gates["per_month"]["2024-01"]["type_unknown_trailing_median_pct"] == 0.3
    rel = [h for h in halts if "relative" in h and "type_unknown" in h]
    assert rel, f"an elevated type month did not halt on the relative rule; {halts}"
    assert not any("absolute" in h for h in halts), "0.7% must not trip the 1% absolute gate"


def test_an_elevated_exchange_month_halts_on_the_trailing_median_rule():
    """The metric my first implementation omitted entirely."""
    from pit_universe_report import unknown_rate_gates

    rows = _months(12, tu=3, eu=3) + [(2024, 1, 3, 7)]
    gates, halts = unknown_rate_gates(_membership(rows))

    rel = [h for h in halts if "relative" in h and "exchange_unknown" in h]
    assert rel, f"an elevated exchange month did not halt; {halts}"


def test_a_stable_month_does_not_halt():
    """The gate must have an off state, or it certifies nothing."""
    from pit_universe_report import unknown_rate_gates

    gates, halts = unknown_rate_gates(_membership(_months(24, tu=3, eu=3)))

    assert halts == [], f"a flat, healthy dataset halted: {halts}"


def test_a_bad_month_cannot_raise_the_baseline_it_is_judged_against():
    """The trailing window is strictly PRIOR to the month under test."""
    from pit_universe_report import unknown_rate_gates

    rows = _months(12, tu=3, eu=3) + [(2024, 1, 40, 3)]
    gates, _ = unknown_rate_gates(_membership(rows))

    assert gates["per_month"]["2024-01"]["type_unknown_trailing_median_pct"] == 0.3, (
        "the month under test contaminated its own baseline"
    )


def test_the_pit_live_count_divergence_gate_is_evaluated(tmp_path, monkeypatch):
    """§A.5: median PIT-vs-live eligible COUNT divergence > 15% halts.

    Distinct from the per-name check: a universe uniformly half the size of live
    can contain every live candidate and still be the wrong population.
    """
    import pit_universe_report as rep

    monkeypatch.setattr(rep, "ROOT", tmp_path)
    live_dir = tmp_path / "v" / "raw" / "live"
    live_dir.mkdir(parents=True)
    _write(live_dir / "dashboard.json.gz", {"run_history": [
        {"date": "2024-03-01", "universe": 1000},
        {"date": "2024-03-04", "universe": 1000},
    ]})
    membership = {
        date(2024, 3, 1): {"eligible_pre_mcap": ["A"] * 500, "traded": []},
        date(2024, 3, 4): {"eligible_pre_mcap": ["A"] * 500, "traded": []},
    }

    result, halts = rep.live_count_divergence("v", membership)

    assert result["median_divergence_pct"] == 50.0
    assert halts, "a universe half the size of live did not halt"

    ok, ok_halts = rep.live_count_divergence("v", {
        d: {"eligible_pre_mcap": ["A"] * 950, "traded": []} for d in membership
    })
    assert ok_halts == [], f"a 5% divergence must pass; {ok_halts}"


def test_a_failed_fetch_is_never_written_into_the_raw_tree(tmp_path):
    """A hole must stay a hole, or resume logic skips it forever.

    `_get` returns a `_failed` sentinel instead of raising so one bad shard
    cannot void 8,000 calls. That only holds if the sentinel is not persisted:
    resume keys on file existence, so a written sentinel is permanently
    indistinguishable from a legitimately empty response.
    """
    path = tmp_path / "grouped" / "2024-03-01.json.gz"

    assert pit._write_raw(path, {"results": None, "_failed": True, "_reason": "http_503"}) is None
    assert not path.exists(), "a failed fetch was persisted and will be skipped on resume"

    assert pit._write_raw(path, {"results": []}) is not None
    assert path.exists(), "a legitimately empty response must still be recorded"


def test_the_ledger_counts_across_runs_so_the_ceiling_cannot_be_reset(tmp_path, monkeypatch):
    """A per-invocation counter is not a budget when the build is resumable."""
    monkeypatch.setattr(pit, "ROOT", tmp_path)

    first = pit._open_ledger("v")
    for _ in range(5):
        first.record("https://api.polygon.io/v3/x", {"a": 1}, 200, 1)
    first.record_failure("https://api.polygon.io/v3/x", {"a": 1}, "http_503", 6)
    first.close()

    second = pit._open_ledger("v")

    assert second.calls == 5, (
        f"resumed ledger lost prior spend ({second.calls}); the ceiling would be "
        "reset by every restart"
    )
    assert second.calls == 5, "a failure record must not inflate the request count"


def test_the_ledger_never_records_request_parameters_verbatim(tmp_path, monkeypatch):
    """Ledgers get attached to artifacts; artifacts on this repo are public."""
    monkeypatch.setattr(pit, "ROOT", tmp_path)

    ledger = pit._open_ledger("v")
    ledger.record("https://api.polygon.io/v3/reference/tickers", {"secret": "sentinel"}, 200, 1)
    ledger.close()

    written = (tmp_path / "v" / "request_ledger.jsonl").read_text()
    assert "sentinel" not in written, "raw parameters reached the ledger"
    assert "params_sha256" in written


def test_get_refuses_to_issue_a_request_with_no_ledger(monkeypatch):
    """An unmetered request path must be impossible, not merely absent."""
    import asyncio

    monkeypatch.setattr(pit, "_LEDGER", None)

    with pytest.raises(RuntimeError, match="ledger"):
        asyncio.run(pit._get(None, "https://api.polygon.io/v3/x", {}))


def test_known_market_holidays_are_not_sessions():
    """Pin calendar BEHAVIOUR, not just the pinned version string.

    The calendar decides which dates enter a vintage, so a library upgrade that
    revised a historical holiday would silently change the dataset. The version
    is pinned in pyproject and recorded in the manifest; this asserts the dates
    that pin is protecting, so an upgrade fails here rather than in a diff of
    8,000 raw files.
    """
    sessions = set(pit.et_sessions(3.0))
    covered = [d for d in sessions if d.year == 2025]
    assert covered, "fixture needs 2025 inside the range"

    for holiday in (date(2025, 1, 1), date(2025, 7, 4), date(2025, 12, 25)):
        assert holiday not in sessions, f"{holiday} is a market holiday, not a session"
    assert date(2025, 7, 3) in sessions, "a half day is still a session"


def test_an_adr_the_live_book_trades_is_reported_as_pit_being_stricter():
    """§4 — the attribution that inverts the conclusion if you get it wrong.

    `src/signals/filter.py` gates exchange, ETF/fund flags, price, volume and
    market cap. It never requires common stock, so PIT's `type == "CS"` is a
    constraint live does not have, and it silently removes ADRs the book
    actually picks. The first version of this check bucketed ADRs with ETFs and
    reported a PIT over-restriction as a live defect.
    """
    from pit_universe_report import live_divergence

    rows = [
        {"ticker": "ADR1", "run_date": "2024-03-01", "picked": True, "rank": 1, "model": "mr"},
        {"ticker": "ETF1", "run_date": "2024-03-01", "picked": False, "rank": 5, "model": "mr"},
    ]
    membership = {date(2024, 3, 1): {"eligible_pre_mcap": [], "traded": ["ADR1", "ETF1"]}}
    labels = {(2024, 3): {
        "ADR1": {"type": "ADRC", "exchange": "NYSE"},
        "ETF1": {"type": "ETF", "exchange": "NASDAQ"},
    }}

    import pit_universe_report as rep

    orig = rep._read_raw
    rep._read_raw = lambda p: {"candidates": rows}
    try:
        import pit_universe_phase_a as mod
        real_m, real_l = mod.build_membership, mod._classification_by_month
        mod.build_membership = lambda v: membership
        mod._classification_by_month = lambda v: labels
        (rep.ROOT / "stub" / "raw" / "live").mkdir(parents=True, exist_ok=True)
        (rep.ROOT / "stub" / "raw" / "live" / "dashboard.json.gz").write_bytes(b"x")
        try:
            result, halts = live_divergence("stub")
        finally:
            mod.build_membership, mod._classification_by_month = real_m, real_l
            import shutil
            shutil.rmtree(rep.ROOT / "stub", ignore_errors=True)
    finally:
        rep._read_raw = orig

    assert result["pit_stricter_than_live_count"] == 1, "the ADR was not attributed to PIT"
    assert result["pit_stricter_picked_count"] == 1, "a PICKED exclusion must be counted"
    assert result["explained_by_live_gates_count"] == 1, "the ETF belongs to live's dead gate"
    assert result["pit_false_exclusion_count"] == 0
    assert any("PICKED" in h for h in halts), f"a picked exclusion must halt; got {halts}"


def test_a_retry_storm_cannot_walk_through_the_call_ceiling(tmp_path, monkeypatch):
    """Neo's reproduction: 11,999/12,000 + repeated 503 ended at 12,005.

    The budget was checked once per logical request, then the retry loop issued
    up to six more. A ceiling a retry storm can overrun is not a ceiling, and it
    fails worst precisely when the vendor is unhealthy and retries are dense.
    Asserts on requests actually SENT, not on the ledger count, because the
    contract is about outbound calls.
    """
    import asyncio

    monkeypatch.setattr(pit, "ROOT", tmp_path)
    monkeypatch.setattr(pit, "PHASE_A_CALL_CEILING", 12_000)
    monkeypatch.setattr(pit, "REQUEST_DELAY_S", 0)
    monkeypatch.setattr(pit.asyncio, "sleep", AsyncMock())

    ledger = pit._open_ledger("v")
    ledger.calls = 11_999

    sent = []

    class _Resp:
        status_code = 503
        def raise_for_status(self): raise AssertionError("unreachable")
        def json(self): return {}

    class _Client:
        async def get(self, url, **kw):
            sent.append(url)
            return _Resp()

    with pytest.raises(pit.BudgetExceeded):
        asyncio.run(pit._get(_Client(), "https://api.polygon.io/v3/x", {"a": 1}))

    assert len(sent) == 1, (
        f"{len(sent)} outbound requests were sent with 1 unit of budget left; "
        "the ceiling was checked once instead of before every attempt"
    )
    assert ledger.calls <= 12_000, f"ledger overran the ceiling: {ledger.calls}"
