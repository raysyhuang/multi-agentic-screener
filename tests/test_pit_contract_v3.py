"""PIT universe contract v3 (rulings R1-R8, 2026-10-06), on a synthetic vintage.

R1 transition resolution, R2 zero-tolerance exchange drift, R3 ADR eligibility,
R6 the zero-median ratio rule, and the version stamp that keeps v2 vintages
replaying under v2 rules. No network: probes are served from a fake `_get`.
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pit_universe_phase_a as pa  # noqa: E402
import pit_universe_report as pr  # noqa: E402

VINTAGE = "2099-01-01"
FEB = [date(2024, 2, d) for d in (1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 20, 21, 22, 23, 26, 27, 28, 29)]
MAR1 = date(2024, 3, 1)


def _w(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(json.dumps(payload).encode())


@pytest.fixture
def vintage(tmp_path, monkeypatch):
    """Feb-2024 grouped bars + Feb/Mar snapshots. LNG moves XASE->XNYS on 02-05;
    PBR is an ADR throughout; NEWCO lists on 02-12 (absent from the Feb snapshot)."""
    monkeypatch.setattr(pa, "ROOT", tmp_path)
    monkeypatch.setattr(pr, "ROOT", tmp_path)
    base = tmp_path / VINTAGE
    (base).mkdir(parents=True)
    (base / "contract.json").write_text(json.dumps({"version": "v3"}))
    for d in FEB + [MAR1]:
        rows = [{"T": "LNG", "c": 150.0, "v": 2e6}, {"T": "PBR", "c": 15.0, "v": 9e6}]
        if d >= date(2024, 2, 12):
            rows.append({"T": "NEWCO", "c": 20.0, "v": 1e6})
        _w(base / "raw" / "grouped" / f"{d}.json.gz", {"results": rows})
    _w(base / "raw" / "reference" / "2024-02" / "page-1.json.gz", {"results": [
        {"ticker": "LNG", "type": "CS", "primary_exchange": "XASE"},
        {"ticker": "PBR", "type": "ADRC", "primary_exchange": "XNYS"},
    ]})
    _w(base / "raw" / "reference" / "2024-03" / "page-1.json.gz", {"results": [
        {"ticker": "LNG", "type": "CS", "primary_exchange": "XNYS"},
        {"ticker": "PBR", "type": "ADRC", "primary_exchange": "XNYS"},
        {"ticker": "NEWCO", "type": "CS", "primary_exchange": "XNAS"},
    ]})
    return base


def _fake_get_factory(truth):
    calls = []

    async def fake_get(client, url, params, allow_404=False):
        t = url.rsplit("/", 1)[-1]
        d = date.fromisoformat(params["date"])
        calls.append((t, d))
        label = truth(t, d)
        if label is None:
            return {"results": None, "_not_found": True}
        return {"results": {"ticker": t, "type": label[0], "primary_exchange": label[1]}}
    return fake_get, calls


def _truth(t, d):
    if t == "LNG":
        return ("CS", "XASE") if d < date(2024, 2, 5) else ("CS", "XNYS")
    if t == "NEWCO":
        return None if d < date(2024, 2, 12) else ("CS", "XNAS")
    return ("ADRC", "XNYS")


def test_adr_common_is_eligible_under_v3(vintage):
    m = pa.build_membership(VINTAGE)
    assert "PBR" in m[FEB[0]]["eligible_pre_mcap"]


def test_v2_vintage_keeps_cs_only_and_no_overrides(vintage):
    (vintage / "contract.json").unlink()
    m = pa.build_membership(VINTAGE)
    assert "PBR" not in m[FEB[0]]["eligible_pre_mcap"]
    assert m[FEB[0]]["exclusions"].get("not_common_stock") == 1


def test_transition_candidates_are_membership_relevant_only(vintage):
    cands = {(c["ticker"], c["month"]) for c in pa.transition_candidates(VINTAGE)}
    assert cands == {("LNG", (2024, 2)), ("NEWCO", (2024, 2))}   # PBR unchanged: not resolved


@pytest.mark.asyncio
async def test_binary_search_finds_the_exact_transition_day(vintage, monkeypatch):
    fake_get, calls = _fake_get_factory(_truth)
    monkeypatch.setattr(pa, "_get", fake_get)
    await pa.resolve_transitions(VINTAGE)
    assert len(calls) <= 2 * 5                       # <= ceil(log2(20)) probes per candidate
    m = pa.build_membership(VINTAGE)
    assert "LNG" not in m[date(2024, 2, 2)]["eligible_pre_mcap"]      # still on AMEX
    assert "LNG" in m[date(2024, 2, 5)]["eligible_pre_mcap"]          # NYSE from the move
    assert "NEWCO" in m[date(2024, 2, 12)]["eligible_pre_mcap"]       # listed mid-month
    # Resume is free: a second run issues no probes.
    calls.clear()
    await pa.resolve_transitions(VINTAGE)
    assert calls == []


@pytest.mark.asyncio
async def test_ambiguous_probe_marks_the_rest_of_the_window_unknown(vintage, monkeypatch):
    def weird(t, d):
        if t == "LNG" and d >= date(2024, 2, 5):
            return ("ETF", "XNYS")                   # neither old nor new
        return _truth(t, d)
    fake_get, _ = _fake_get_factory(weird)
    monkeypatch.setattr(pa, "_get", fake_get)
    await pa.resolve_transitions(VINTAGE)
    m = pa.build_membership(VINTAGE)
    assert "LNG" not in m[date(2024, 2, 20)]["eligible_pre_mcap"]
    assert m[date(2024, 2, 20)]["exclusions"].get("type_unknown", 0) >= 1


def test_spine_refuses_to_extend_a_vintage_under_another_contract(vintage):
    (vintage / "contract.json").write_text(json.dumps({"version": "v2"}))
    with pytest.raises(RuntimeError, match="start a new vintage"):
        pa._stamp_contract(VINTAGE)


def test_zero_median_uses_the_absolute_gate_only():
    """R6: 13 months of 0% exchange_unknown then 0.5% must not halt (absolute is 1%)."""
    membership = {}
    for i in range(14):
        d = date(2023 + (i // 12), (i % 12) + 1, 3)
        unknown = 1 if i == 13 else 0
        membership[d] = {"pre_classification": ["X"] * 200,
                         "exclusions": {"exchange_unknown": unknown, "type_unknown": 0}}
    _, halts = pr.unknown_rate_gates(membership)
    assert halts == []


def test_any_exchange_disagreement_halts(vintage):
    """R2: zero tolerance — one disagreement in a month is a halt."""
    _w(vintage / "raw" / "audit" / "2024-02" / "LNG_2024-02-22.json.gz", {
        "results": {"ticker": "LNG", "type": "CS", "primary_exchange": "XNYS"},
        "_audit": {"bucket": "common_stock", "date": "2024-02-22", "ticker": "LNG"},
    })
    res = pr._audit_results(VINTAGE)
    assert res["per_month"]["2024-02"]["exchange_disagree"] == 1   # held AMEX, unresolved
    assert pr.HALT_DRIFT_EXCHANGE_DISAGREEMENTS == 0
