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
    mar = {"results": [
        {"ticker": "LNG", "type": "CS", "primary_exchange": "XNYS"},
        {"ticker": "PBR", "type": "ADRC", "primary_exchange": "XNYS"},
        {"ticker": "NEWCO", "type": "CS", "primary_exchange": "XNAS"},
    ]}
    _w(base / "raw" / "reference" / "2024-03" / "page-1.json.gz", mar)
    # Trailing comparison snapshot: the session after the frozen end (2024-03-01).
    _w(base / "raw" / "reference_trailing" / "2024-03-04" / "page-1.json.gz", mar)
    pa._EXPECTED_CACHE.clear()
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
    # A repo GitHub does not confirm as private gets no command, whatever its name.
    monkeypatch.setenv("PIT_RELEASE_REPO", "someone/public-data")
    monkeypatch.setattr(pr, "_repo_is_private", lambda r: False)
    pr.package(VINTAGE)
    assert json.loads(capsys.readouterr().out)["upload"] is None
    monkeypatch.setenv("PIT_RELEASE_REPO", "raysyhuang/multi-agentic-screener; rm -rf /")
    pr.package(VINTAGE)
    assert json.loads(capsys.readouterr().out)["upload"] is None
    monkeypatch.setenv("PIT_RELEASE_REPO", "raysyhuang/mas-data")
    monkeypatch.setattr(pr, "_repo_is_private", lambda r: True)
    pr.package(VINTAGE)
    cmd = json.loads(capsys.readouterr().out)["upload"]
    assert isinstance(cmd, list) and cmd[cmd.index("--repo") + 1] == "raysyhuang/mas-data"


def test_audit_results_refuse_a_sample_drawn_under_other_transition_results(vintage):
    (vintage / "raw" / "audit").mkdir(parents=True)
    (vintage / "audit_plan.json").write_text(json.dumps({"overrides_sha256": "stale", "pairs": []}))
    res = pr._audit_results(VINTAGE)
    assert res["ran"] is False and res.get("stale") is True


def test_final_month_transition_is_discovered_against_the_trailing_snapshot(vintage):
    trailing = {"results": [
        {"ticker": "LNG", "type": "CS", "primary_exchange": "XNYS"},
        {"ticker": "PBR", "type": "ETF", "primary_exchange": "XNYS"},       # changes in March
        {"ticker": "NEWCO", "type": "CS", "primary_exchange": "XNAS"},
    ]}
    _w(vintage / "raw" / "reference_trailing" / "2024-03-04" / "page-1.json.gz", trailing)
    pa._EXPECTED_CACHE.clear()
    cands = {(c["ticker"], c["month"]) for c in pa.transition_candidates(VINTAGE)}
    assert ("PBR", (2024, 3)) in cands


def test_missing_trailing_snapshot_refuses_candidate_generation(vintage):
    import shutil
    shutil.rmtree(vintage / "raw" / "reference_trailing")
    pa._EXPECTED_CACHE.clear()
    with pytest.raises(RuntimeError, match="trailing comparison snapshot"):
        pa.transition_candidates(VINTAGE)


def test_stray_grouped_file_outside_the_frozen_range_is_refused(vintage):
    _w(vintage / "raw" / "grouped" / "2024-03-04.json.gz", {"results": []})
    with pytest.raises(RuntimeError, match="extra"):
        pa.build_membership(VINTAGE)


def test_warmup_bars_count_toward_history_but_emit_no_membership(vintage, monkeypatch):
    stamp = json.loads((vintage / "contract.json").read_text())
    stamp["warmup_sessions"] = 3
    (vintage / "contract.json").write_text(json.dumps(stamp))
    for d in ("2024-01-29", "2024-01-30", "2024-01-31"):
        _w(vintage / "raw" / "grouped" / f"{d}.json.gz", {"results": [{"T": "PBR", "c": 15.0, "v": 9e6}]})
    monkeypatch.setattr(pa, "MIN_PRIOR_BARS", 3)
    m = pa.build_membership(VINTAGE)
    assert date(2024, 1, 31) not in m                     # warm-up emits nothing
    assert "PBR" in m[FEB[0]]["eligible_pre_mcap"]          # 3 warm-up bars suffice
    assert "LNG" not in m[FEB[0]]["eligible_pre_mcap"]      # no warm-up history


@pytest.mark.asyncio
async def test_orphan_transition_result_makes_the_set_incomplete_and_is_never_applied(vintage, monkeypatch):
    fake_get, _ = _fake_get_factory(_truth)
    monkeypatch.setattr(pa, "_get", fake_get)
    await pa.resolve_transitions(VINTAGE)
    assert pa.transition_status(VINTAGE)["complete"]
    pa._write_raw_unchecked(vintage / "raw" / "transitions" / "2024-02" / "PBR.result.json.gz",
                            {"ticker": "PBR", "month": "2024-02", "old": None, "new": None,
                             "effective": None, "ambiguous_from": "2024-02-02", "probes": []})
    st = pa.transition_status(VINTAGE)
    assert not st["complete"] and st["extra"] == [("2024-02", "PBR")]
    assert "PBR" in pa.build_membership(VINTAGE)[date(2024, 2, 20)]["eligible_pre_mcap"]


def test_incomplete_audit_plan_is_not_reported_as_run(vintage):
    _w(vintage / "raw" / "audit" / "2024-02" / "LNG_2024-02-22.json.gz", {
        "results": {"ticker": "LNG", "type": "CS", "primary_exchange": "XNYS"},
        "_audit": {"bucket": "common_stock", "date": "2024-02-22", "ticker": "LNG"}})
    (vintage / "audit_plan.json").write_text(json.dumps({
        "overrides_sha256": pa.overrides_fingerprint(VINTAGE),
        "pairs": ["2024-02/LNG_2024-02-22", "2024-02/PBR_2024-02-22"]}))
    res = pr._audit_results(VINTAGE)
    assert res["ran"] is False and res["incomplete"] and res["missing_count"] == 1


def test_live_count_gate_evaluates_with_60_clean_paired_runs_and_defers_on_unknown_history(vintage, monkeypatch):
    m = pa.build_membership(VINTAGE)
    with_mcap = {d: {**rec, "eligible": rec["eligible_pre_mcap"]} for d, rec in m.items()}
    rows = [{"date": "2024-02-%02d" % (d.day), "universe": len(with_mcap[d]["eligible"])}
            for d in FEB]
    monkeypatch.setattr(pr, "LIVE_GATE_MIN_CLEAN_OBS", 5)
    monkeypatch.setattr(pr, "_read_raw", lambda p: {"run_history": rows})
    (vintage / "raw" / "live").mkdir(parents=True, exist_ok=True)
    (vintage / "raw" / "live" / "dashboard.json.gz").write_bytes(b"x")
    res, halts = pr.live_count_gate_v3(VINTAGE, with_mcap, boundaries=[date(2024, 1, 15)])
    assert res["status"] == "EVALUATED" and res["paired"] >= 5 and halts == []
    partial = dict(with_mcap)
    partial[FEB[0]] = m[FEB[0]]                                   # one session lacks Phase B
    res, _ = pr.live_count_gate_v3(VINTAGE, partial, boundaries=[date(2024, 1, 15)])
    assert res["status"] == "DEFERRED"
    def boom():
        raise RuntimeError("no origin/main")
    monkeypatch.setattr(pr, "universe_definition_boundaries", boom)
    res, _ = pr.live_count_gate_v3(VINTAGE, with_mcap)
    assert res["status"] == "DEFERRED" and "history unavailable" in res["reason"]


def test_verify_rejects_a_manifest_that_drops_the_v3_identity(vintage):
    (vintage / "manifest.json").write_text(json.dumps({"raw_hashes": {"x": "y"}}))
    with pytest.raises(SystemExit, match="local contract v3 != manifest contract v2"):
        pr.verify(VINTAGE)
