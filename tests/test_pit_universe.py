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
