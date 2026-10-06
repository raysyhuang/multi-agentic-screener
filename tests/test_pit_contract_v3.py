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


@pytest.fixture(autouse=True)
def _no_warmup_history(monkeypatch):
    """The synthetic vintage has no warm-up bars; history is tested separately."""
    monkeypatch.setattr(pa, "MIN_PRIOR_BARS", 0)


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
    (base / "contract.json").write_text(json.dumps({
        "version": "v3", "start": "2024-02-01", "end": "2024-03-01", "warmup_sessions": 0}))
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
        pa._stamp_contract(VINTAGE, pa.DEFAULT_START, date(2026, 10, 5))


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
    (vintage / "audit_plan.json").write_text(json.dumps({
        "overrides_sha256": pa.overrides_fingerprint(VINTAGE),
        "pairs": ["2024-02/LNG_2024-02-22"]}))
    res = pr._audit_results(VINTAGE)
    assert res["per_month"]["2024-02"]["exchange_disagree"] == 1   # held AMEX, unresolved
    assert pr.HALT_DRIFT_EXCHANGE_DISAGREEMENTS == 0


def test_price_and_volume_thresholds_are_strict_under_v3(vintage):
    _w(vintage / "raw" / "grouped" / f"{FEB[1]}.json.gz", {"results": [
        {"T": "LNG", "c": 5.0, "v": 2e6}, {"T": "PBR", "c": 15.0, "v": 500_000}]})
    m = pa.build_membership(VINTAGE)[FEB[1]]
    assert m["exclusions"]["failed_price"] == 1 and m["exclusions"]["failed_volume"] == 1


def test_history_rule_excludes_until_200_prior_bars(vintage, monkeypatch):
    monkeypatch.setattr(pa, "MIN_PRIOR_BARS", 5)
    m = pa.build_membership(VINTAGE)
    assert "PBR" not in m[FEB[4]]["eligible_pre_mcap"]          # 4 prior bars
    assert "PBR" in m[FEB[5]]["eligible_pre_mcap"]              # 5 prior bars
    assert m[FEB[0]]["exclusions"]["insufficient_history"] >= 1


def test_audit_and_report_refuse_incomplete_transitions(vintage):
    with pytest.raises(RuntimeError, match="§3a-v2 incomplete"):
        pa.require_transitions_complete(VINTAGE)


def test_stamp_freezes_range_and_refuses_to_restamp_a_v2_vintage(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "ROOT", tmp_path)
    (tmp_path / "v2v" / "raw" / "grouped").mkdir(parents=True)
    (tmp_path / "v2v" / "raw" / "grouped" / "x.json.gz").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="refusing to re-stamp"):
        pa._stamp_contract("v2v", pa.DEFAULT_START, date(2026, 10, 5))
    with pytest.raises(RuntimeError, match="fixes the range start"):
        pa._stamp_contract("fresh", date(2020, 1, 2), date(2026, 10, 5))
    rec = pa._stamp_contract("fresh", pa.DEFAULT_START, date(2026, 10, 5))
    assert rec["end"] == "2026-10-05"
    assert pa._stamp_contract("fresh", pa.DEFAULT_START, date(2027, 1, 4))["end"] == "2026-10-05"


def test_live_count_gate_defers_before_phase_b_and_without_60_clean_runs(vintage):
    m = pa.build_membership(VINTAGE)
    res, halts = pr.live_count_gate_v3(VINTAGE, m, boundaries=[date(2026, 10, 5)])
    assert res["status"] == "DEFERRED" and "Phase B" in res["reason"] and halts == []
    with_mcap = {d: {**rec, "eligible": rec["eligible_pre_mcap"]} for d, rec in m.items()}
    res, _ = pr.live_count_gate_v3(VINTAGE, with_mcap, boundaries=[date(2026, 10, 5)])
    assert res["status"] == "DEFERRED" and res["min_required"] == 60


def test_package_never_emits_an_upload_to_the_public_repo(vintage, monkeypatch, capsys):
    monkeypatch.delenv("PIT_RELEASE_REPO", raising=False)
    archive = pr.package(VINTAGE)
    out = json.loads(capsys.readouterr().out)
    assert out["upload"] is None and "PRIVATE" in out["note"]
    import tarfile
    with tarfile.open(archive) as tar:
        assert f"{VINTAGE}/contract.json" in tar.getnames()
    monkeypatch.setenv("PIT_RELEASE_REPO", "raysyhuang/mas-data")
    pr.package(VINTAGE)
    assert "--repo raysyhuang/mas-data" in json.loads(capsys.readouterr().out)["upload"]


def test_audit_results_refuse_a_sample_drawn_under_other_transition_results(vintage):
    (vintage / "raw" / "audit").mkdir(parents=True)
    (vintage / "audit_plan.json").write_text(json.dumps({"overrides_sha256": "stale", "pairs": []}))
    res = pr._audit_results(VINTAGE)
    assert res["ran"] is False and res.get("stale") is True
