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


def _mark(snap_dir: Path, as_of: str) -> None:
    """Completion marker for a one-page synthetic snapshot."""
    pages = sorted(snap_dir.glob("page-*.json.gz"))
    (snap_dir / pa.SNAPSHOT_MARKER).write_text(json.dumps({
        "as_of": as_of, "pages": len(pages), "page_sha256": [pa._sha256_file(p) for p in pages]}))


def _snap(path: Path, payload, as_of: str) -> None:
    """A one-page snapshot with request provenance and a completion marker."""
    _w(path, {**payload, "_request": {"as_of": as_of, "cursor": None}})
    _mark(path.parent, as_of)


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
        # R9: unadjusted bars (no splits in this fixture, so identical).
        _w(base / "raw" / "grouped_raw" / f"{d}.json.gz", {"results": rows})
    _snap(base / "raw" / "splits" / "page-1.json.gz", {"results": []}, "2024-03-01")
    _snap(base / "raw" / "reference" / "2024-02" / "page-1.json.gz", {"results": [
        {"ticker": "LNG", "type": "CS", "primary_exchange": "XASE"},
        {"ticker": "PBR", "type": "ADRC", "primary_exchange": "XNYS"},
    ]}, "2024-02-01")
    mar = {"results": [
        {"ticker": "LNG", "type": "CS", "primary_exchange": "XNYS"},
        {"ticker": "PBR", "type": "ADRC", "primary_exchange": "XNYS"},
        {"ticker": "NEWCO", "type": "CS", "primary_exchange": "XNAS"},
    ]}
    _snap(base / "raw" / "reference" / "2024-03" / "page-1.json.gz", mar, "2024-03-01")
    # Trailing comparison snapshot: the session after the frozen end (2024-03-01).
    _snap(base / "raw" / "reference_trailing" / "2024-03-04" / "page-1.json.gz", mar, "2024-03-04")
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
        "sampler_version": pa.SAMPLER_VERSION,
        "overrides_sha256": pa.overrides_fingerprint(VINTAGE),
        "pairs": ["2024-02/LNG_2024-02-22"]}))
    res = pr._audit_results(VINTAGE)
    assert res["per_month"]["2024-02"]["exchange_disagree"] == 1   # held AMEX, unresolved
    assert pr.HALT_DRIFT_EXCHANGE_DISAGREEMENTS == 0


def test_price_and_volume_thresholds_are_strict_under_v3(vintage):
    rows = [{"T": "LNG", "c": 5.0, "v": 2e6}, {"T": "PBR", "c": 15.0, "v": 500_000}]
    _w(vintage / "raw" / "grouped" / f"{FEB[1]}.json.gz", {"results": rows})
    _w(vintage / "raw" / "grouped_raw" / f"{FEB[1]}.json.gz", {"results": rows})
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
    (vintage / "audit_plan.json").write_text(json.dumps({
        "sampler_version": pa.SAMPLER_VERSION, "overrides_sha256": "stale", "pairs": []}))
    res = pr._audit_results(VINTAGE)
    assert res["ran"] is False and res.get("stale") is True


def test_final_month_transition_is_discovered_against_the_trailing_snapshot(vintage):
    trailing = {"results": [
        {"ticker": "LNG", "type": "CS", "primary_exchange": "XNYS"},
        {"ticker": "PBR", "type": "ETF", "primary_exchange": "XNYS"},       # changes in March
        {"ticker": "NEWCO", "type": "CS", "primary_exchange": "XNAS"},
    ]}
    _snap(vintage / "raw" / "reference_trailing" / "2024-03-04" / "page-1.json.gz", trailing, "2024-03-04")
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
        "sampler_version": pa.SAMPLER_VERSION,
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


def test_trailing_snapshot_is_not_available_until_its_session_has_closed(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    class _DT(datetime):
        now_val = datetime(2026, 10, 6, 14, 0, tzinfo=ZoneInfo("America/New_York"))

        @classmethod
        def now(cls, tz=None):
            return cls.now_val
    import datetime as dtmod
    monkeypatch.setattr(dtmod, "datetime", _DT)
    assert pa._session_complete(date(2026, 10, 5)) is True
    assert pa._session_complete(date(2026, 10, 6)) is False          # before the close
    _DT.now_val = datetime(2026, 10, 6, 16, 30, tzinfo=ZoneInfo("America/New_York"))
    assert pa._session_complete(date(2026, 10, 6)) is True


def test_unmarked_or_altered_snapshot_is_refused(vintage):
    (vintage / "raw" / "reference" / "2024-02" / pa.SNAPSHOT_MARKER).unlink()
    with pytest.raises(RuntimeError, match="no completion marker"):
        pa._classification_by_month(VINTAGE)
    _mark(vintage / "raw" / "reference" / "2024-02", "2024-02-01")
    _w(vintage / "raw" / "reference" / "2024-02" / "page-1.json.gz",
       {"results": [], "_request": {"as_of": "2024-02-01", "cursor": None}})
    with pytest.raises(RuntimeError, match="changed since completion"):
        pa._classification_by_month(VINTAGE)


def test_trailing_snapshot_in_the_wrong_date_directory_does_not_count(vintage):
    import shutil
    shutil.move(str(vintage / "raw" / "reference_trailing" / "2024-03-04"),
                str(vintage / "raw" / "reference_trailing" / "2024-03-05"))
    with pytest.raises(RuntimeError, match="trailing comparison snapshot"):
        pa.transition_candidates(VINTAGE)


@pytest.mark.asyncio
async def test_partial_pagination_writes_no_marker_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "ROOT", tmp_path)
    responses = {None: {"results": [{"ticker": "A"}], "next_url": "x?cursor=c2"},
                 "c2": {"_failed": True, "_reason": "http_503"}}

    async def fake_get(client, url, params, allow_404=False):
        return responses[params.get("cursor")]
    monkeypatch.setattr(pa, "_get", fake_get)
    assert await pa._fetch_snapshot(None, "v", ("reference", "2024-02"), date(2024, 2, 1)) is False
    assert not (tmp_path / "v" / "raw" / "reference" / "2024-02" / pa.SNAPSHOT_MARKER).exists()
    responses["c2"] = {"results": [{"ticker": "B"}]}
    assert await pa._fetch_snapshot(None, "v", ("reference", "2024-02"), date(2024, 2, 1)) is True
    m = json.loads((tmp_path / "v" / "raw" / "reference" / "2024-02" / pa.SNAPSHOT_MARKER).read_text())
    assert m["pages"] == 2


@pytest.mark.asyncio
async def test_result_for_other_inputs_is_stale_and_recomputed(vintage, monkeypatch):
    fake_get, calls = _fake_get_factory(_truth)
    monkeypatch.setattr(pa, "_get", fake_get)
    await pa.resolve_transitions(VINTAGE)
    path = vintage / "raw" / "transitions" / "2024-02" / "LNG.result.json.gz"
    r = pa._read_raw(path)
    r["inputs_sha256"] = "other-inputs"
    pa._write_raw_unchecked(path, r)
    st = pa.transition_status(VINTAGE)
    assert not st["complete"] and ("2024-02", "LNG") in st["stale"]
    await pa.resolve_transitions(VINTAGE)
    assert pa.transition_status(VINTAGE)["complete"]


def test_candidate_cache_follows_spine_changes_in_the_same_process(vintage):
    first = pa._expected_transition_keys(VINTAGE)
    trailing = {"results": [
        {"ticker": "LNG", "type": "CS", "primary_exchange": "XNYS"},
        {"ticker": "PBR", "type": "ETF", "primary_exchange": "XNYS"},
        {"ticker": "NEWCO", "type": "CS", "primary_exchange": "XNAS"}]}
    import os
    import time
    time.sleep(0.01)
    _snap(vintage / "raw" / "reference_trailing" / "2024-03-04" / "page-1.json.gz", trailing, "2024-03-04")
    os.utime(vintage / "raw" / "reference_trailing" / "2024-03-04" / "page-1.json.gz")
    assert ("2024-03", "PBR") in pa._expected_transition_keys(VINTAGE) - first


def test_unplanned_audit_file_or_old_sampler_is_not_reported_as_run(vintage):
    _w(vintage / "raw" / "audit" / "2024-02" / "LNG_2024-02-22.json.gz", {
        "results": {"ticker": "LNG", "type": "CS", "primary_exchange": "XNYS"},
        "_audit": {"bucket": "common_stock", "date": "2024-02-22", "ticker": "LNG"}})
    _w(vintage / "raw" / "audit" / "2024-02" / "PBR_2024-02-22.json.gz", {"results": {}, "_audit": {}})
    plan = {"sampler_version": pa.SAMPLER_VERSION, "overrides_sha256": pa.overrides_fingerprint(VINTAGE),
            "pairs": ["2024-02/LNG_2024-02-22"]}
    (vintage / "audit_plan.json").write_text(json.dumps(plan))
    res = pr._audit_results(VINTAGE)
    assert res["ran"] is False and res["extra_count"] == 1
    (vintage / "raw" / "audit" / "2024-02" / "PBR_2024-02-22.json.gz").unlink()
    (vintage / "audit_plan.json").write_text(json.dumps({**plan, "sampler_version": "phase-a/1"}))
    assert pr._audit_results(VINTAGE)["ran"] is False


def test_verify_rejects_v2_local_against_v3_manifest_and_wrong_vintage(vintage):
    (vintage / "contract.json").unlink()
    (vintage / "manifest.json").write_text(json.dumps({"raw_hashes": {"x": "y"}, "contract_version": "v3",
                                                        "vintage": VINTAGE}))
    with pytest.raises(SystemExit, match="local contract v2 != manifest contract v3"):
        pr.verify(VINTAGE)
    (vintage / "manifest.json").write_text(json.dumps({"raw_hashes": {"x": "y"}, "vintage": "1999-01-01"}))
    with pytest.raises(SystemExit, match="manifest is for vintage"):
        pr.verify(VINTAGE)


def test_v2_zero_median_keeps_its_original_relative_halt():
    membership = {}
    for i in range(14):
        d = date(2023 + (i // 12), (i % 12) + 1, 3)
        membership[d] = {"pre_classification": ["X"] * 200,
                         "exclusions": {"exchange_unknown": 1 if i == 13 else 0, "type_unknown": 0}}
    _, halts_v2 = pr.unknown_rate_gates(membership, v3=False)
    _, halts_v3 = pr.unknown_rate_gates(membership, v3=True)
    assert any("relative" in h for h in halts_v2) and halts_v3 == []


def test_marker_over_a_page_that_still_has_next_url_is_refused(vintage):
    d = vintage / "raw" / "reference" / "2024-02"
    _w(d / "page-1.json.gz", {"results": [{"ticker": "LNG", "type": "CS", "primary_exchange": "XASE"}],
                              "next_url": "x?cursor=more",
                              "_request": {"as_of": "2024-02-01", "cursor": None}})
    _mark(d, "2024-02-01")
    with pytest.raises(RuntimeError, match="pagination chain broken"):
        pa._classification_by_month(VINTAGE)


def test_marker_for_another_as_of_date_is_refused(vintage):
    _mark(vintage / "raw" / "reference" / "2024-02", "2024-02-02")
    with pytest.raises(RuntimeError, match="as_of"):
        pa._classification_by_month(VINTAGE)


@pytest.mark.asyncio
async def test_existing_valid_marker_is_validated_on_the_fast_path(vintage):
    _mark(vintage / "raw" / "reference" / "2024-02", "2024-02-02")      # wrong date
    with pytest.raises(RuntimeError, match="as_of"):
        await pa._fetch_snapshot(None, VINTAGE, ("reference", "2024-02"), date(2024, 2, 1))


@pytest.mark.asyncio
async def test_corrupt_existing_page_writes_no_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "ROOT", tmp_path)
    d = tmp_path / "v" / "raw" / "reference" / "2024-02"
    d.mkdir(parents=True)
    (d / "page-1.json.gz").write_bytes(b"not gzip")
    with pytest.raises(Exception):
        await pa._fetch_snapshot(None, "v", ("reference", "2024-02"), date(2024, 2, 1))
    assert not (d / pa.SNAPSHOT_MARKER).exists()


def test_early_close_session_completes_15_minutes_after_its_scheduled_close(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import datetime as dtmod

    class _DT(datetime):
        now_val = datetime(2026, 11, 27, 13, 20, tzinfo=ZoneInfo("America/New_York"))  # day after Thanksgiving

        @classmethod
        def now(cls, tz=None):
            return cls.now_val
    monkeypatch.setattr(dtmod, "datetime", _DT)
    assert pa._session_complete(date(2026, 11, 27)) is True          # 13:00 close + 15 min
    _DT.now_val = datetime(2026, 11, 27, 13, 10, tzinfo=ZoneInfo("America/New_York"))
    assert pa._session_complete(date(2026, 11, 27)) is False


def test_v2_vintage_reads_snapshots_without_markers(vintage):
    (vintage / "contract.json").unlink()
    for marker in (vintage / "raw").rglob(pa.SNAPSHOT_MARKER):
        marker.unlink()
    labels = pa._classification_by_month(VINTAGE)
    assert labels[(2024, 2)]["LNG"]["exchange"] == "AMEX"


def test_v3_manifest_must_name_its_vintage(vintage):
    (vintage / "manifest.json").write_text(json.dumps({
        "raw_hashes": {"x": "y"}, "contract_version": "v3",
        "contract_stamp_sha256": pr._sha256(vintage / "contract.json")}))
    with pytest.raises(SystemExit, match="manifest is for vintage None"):
        pr.verify(VINTAGE)


def test_verify_covers_completion_markers_for_v3(vintage):
    stamp = vintage / "contract.json"
    raw = vintage / "raw"
    files = sorted(list(raw.rglob("*.json.gz")) + list(raw.rglob("_complete.json")))
    manifest = {"vintage": VINTAGE, "contract_version": "v3",
                "contract_stamp_sha256": pr._sha256(stamp),
                "config_sha256": __import__("hashlib").sha256(stamp.read_bytes()).hexdigest(),
                "raw_hashes": {str(p.relative_to(vintage)): pr._sha256(p) for p in files}}
    (vintage / "manifest.json").write_text(json.dumps(manifest))
    assert pr.verify(VINTAGE)["verified"]
    (raw / "reference" / "2024-02" / pa.SNAPSHOT_MARKER).unlink()
    with pytest.raises(SystemExit, match="VERIFY FAILED"):
        pr.verify(VINTAGE)


@pytest.mark.asyncio
async def test_valid_marker_fast_path_returns_true_without_any_request(vintage, monkeypatch):
    async def no_calls(*a, **k):
        raise AssertionError("a valid marker must not trigger a request")
    monkeypatch.setattr(pa, "_get", no_calls)
    assert await pa._fetch_snapshot(None, VINTAGE, ("reference", "2024-02"), date(2024, 2, 1)) is True


@pytest.mark.asyncio
async def test_unproven_unmarked_pages_are_quarantined_and_refetched(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "ROOT", tmp_path)
    d = tmp_path / "v" / "raw" / "reference" / "2024-02"
    _w(d / "page-1.json.gz", {"results": [{"ticker": "OLD"}], "next_url": "x?cursor=c2"})   # no provenance
    _w(d / "page-2.json.gz", {"results": [{"ticker": "OTHER"}]})
    calls = []

    async def fake_get(client, url, params, allow_404=False):
        calls.append(params.get("cursor"))
        return {"results": [{"ticker": "NEW"}]}
    monkeypatch.setattr(pa, "_get", fake_get)
    assert await pa._fetch_snapshot(None, "v", ("reference", "2024-02"), date(2024, 2, 1)) is True
    assert calls == [None]                                         # refetched from page 1
    assert list((d.parent).glob("2024-02.untrusted-*"))           # old pages kept, quarantined
    labels = pa._read_snapshot(d, require_complete=True, expected_as_of=date(2024, 2, 1))
    assert set(labels) == {"NEW"}


def test_page_from_another_chain_breaks_provenance(vintage):
    d = vintage / "raw" / "reference" / "2024-02"
    _w(d / "page-1.json.gz", {"results": [], "next_url": "x?cursor=c2",
                              "_request": {"as_of": "2024-02-01", "cursor": None}})
    _w(d / "page-2.json.gz", {"results": [], "_request": {"as_of": "2024-02-01", "cursor": "zzz"}})
    _mark(d, "2024-02-01")
    with pytest.raises(RuntimeError, match="provenance does not continue the chain"):
        pa._classification_by_month(VINTAGE)


def test_missing_monthly_snapshot_is_refused(vintage):
    import shutil
    shutil.rmtree(vintage / "raw" / "reference" / "2024-02")
    with pytest.raises(RuntimeError, match="monthly snapshot\\(s\\) missing"):
        pa._classification_by_month(VINTAGE)


def test_cursor_is_parsed_from_the_query_not_string_split():
    assert pa._cursor_of("https://api.polygon.io/v3/reference/tickers?cursor=abc&limit=1000") == "abc"
    assert pa._cursor_of("https://x/y?limit=1&cursor=abc") == "abc"
    assert pa._cursor_of(None) is None
    with pytest.raises(RuntimeError, match="exactly one cursor"):
        pa._cursor_of("https://x/y?cursor=a&cursor=b")


@pytest.mark.asyncio
async def test_pagination_passes_only_the_cursor_value(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "ROOT", tmp_path)
    seen = []

    async def fake_get(client, url, params, allow_404=False):
        seen.append(params.get("cursor"))
        if params.get("cursor") is None:
            return {"results": [{"ticker": "A"}], "next_url": "https://x/v3?cursor=c2&limit=1000"}
        return {"results": [{"ticker": "B"}]}
    monkeypatch.setattr(pa, "_get", fake_get)
    assert await pa._fetch_snapshot(None, "v", ("reference", "2024-02"), date(2024, 2, 1))
    assert seen == [None, "c2"]
    labels = pa._read_snapshot(tmp_path / "v" / "raw" / "reference" / "2024-02", True, date(2024, 2, 1))
    assert set(labels) == {"A", "B"}


def test_noncanonical_month_directory_is_refused(vintage):
    import shutil
    shutil.copytree(vintage / "raw" / "reference" / "2024-02", vintage / "raw" / "reference" / "2024-2")
    with pytest.raises(RuntimeError, match="unexpected reference directories"):
        pa._classification_by_month(VINTAGE)
    shutil.rmtree(vintage / "raw" / "reference" / "2024-02")
    with pytest.raises(RuntimeError, match="missing"):
        pa._classification_by_month(VINTAGE)


@pytest.mark.asyncio
async def test_two_quarantines_in_the_same_second_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "ROOT", tmp_path)
    d = tmp_path / "v" / "raw" / "reference" / "2024-02"
    calls = {"n": 0}

    async def fake_get(client, url, params, allow_404=False):
        calls["n"] += 1
        return {"_failed": True, "_reason": "http_503"}
    monkeypatch.setattr(pa, "_get", fake_get)
    for _ in range(2):
        _w(d / "page-1.json.gz", {"results": []})                      # unproven page again
        assert await pa._fetch_snapshot(None, "v", ("reference", "2024-02"), date(2024, 2, 1)) is False
    assert len(list(d.parent.glob("2024-02.untrusted-*"))) == 2


def test_malformed_page_name_is_unproven_not_a_crash(tmp_path):
    d = tmp_path / "snap"
    _w(d / "page-old.json.gz", {"results": []})
    assert pa._pages_have_provenance(d, date(2024, 2, 1)) is False



# ── R9: unadjusted observables ──────────────────────────────────────────────

def test_price_filter_uses_the_unadjusted_close_not_a_split_adjusted_one(vintage):
    """A stock that later reverse-splits shows a high ADJUSTED close; on D it traded at $1."""
    d = FEB[2]
    _w(vintage / "raw" / "grouped" / f"{d}.json.gz", {"results": [
        {"T": "LNG", "c": 150.0, "v": 2e6}, {"T": "PBR", "c": 15.0, "v": 9e6}]})
    _w(vintage / "raw" / "grouped_raw" / f"{d}.json.gz", {"results": [
        {"T": "LNG", "c": 150.0, "v": 2e6}, {"T": "PBR", "c": 1.5, "v": 9e6}]})   # $1.50 on D
    m = pa.build_membership(VINTAGE)[d]
    assert "PBR" not in m["eligible_pre_mcap"] and m["exclusions"]["failed_price"] == 1


def test_missing_unadjusted_bar_excludes_and_counts(vintage):
    d = FEB[3]
    _w(vintage / "raw" / "grouped_raw" / f"{d}.json.gz", {"results": [{"T": "LNG", "c": 150.0, "v": 2e6}]})
    m = pa.build_membership(VINTAGE)[d]
    assert "PBR" not in m["eligible_pre_mcap"] and m["exclusions"]["no_unadjusted_bar"] == 1


def test_grouped_raw_tree_must_match_membership_sessions(vintage):
    (vintage / "raw" / "grouped_raw" / f"{FEB[0]}.json.gz").unlink()
    with pytest.raises(RuntimeError, match="grouped_raw tree"):
        pa.build_membership(VINTAGE)


def test_split_factor_covers_splits_after_the_snapshot_through_d():
    splits = [(date(2024, 6, 10), 10.0), (date(2024, 9, 1), 0.5)]
    assert pa.split_factor(splits, date(2024, 4, 1), date(2024, 6, 7)) == 1.0
    assert pa.split_factor(splits, date(2024, 4, 1), date(2024, 6, 10)) == 10.0
    assert pa.split_factor(splits, date(2024, 4, 1), date(2024, 9, 3)) == 5.0


# ── Phase B ──────────────────────────────────────────────────────────────────

async def _phase_b_ready(vintage, monkeypatch, shares):
    fake_get, _ = _fake_get_factory(_truth)
    monkeypatch.setattr(pa, "_get", fake_get)
    await pa.resolve_transitions(VINTAGE)

    async def details_get(client, url, params, allow_404=False):
        t = url.rsplit("/", 1)[-1]
        if shares.get(t) is None:
            return {"results": None, "_not_found": True}
        return {"results": {"ticker": t, "weighted_shares_outstanding": shares[t],
                            "market_cap": shares[t] * 10}}
    monkeypatch.setattr(pa, "_get", details_get)
    await pa.fetch_mcap_details(VINTAGE)


@pytest.mark.asyncio
async def test_phase_b_lookups_only_for_eligible_quarters_and_mcap_gate(vintage, monkeypatch):
    # PBR at $15 x 30M shares = $450M (in); LNG at $150 x 1M = $150M (out); NEWCO unknown.
    await _phase_b_ready(vintage, monkeypatch, {"PBR": 30e6, "LNG": 1e6, "NEWCO": None})
    st = pa.phase_b_status(VINTAGE)
    assert st["complete"]
    assert {t for t, q in pa.mcap_candidates(VINTAGE)} == {"LNG", "PBR", "NEWCO"}
    m = pa.build_membership_with_mcap(VINTAGE)
    day = m[date(2024, 2, 20)]
    assert day["eligible"] == ["PBR"]
    assert day["exclusions"]["failed_mcap"] == 1 and day["exclusions"]["mcap_unknown"] == 1


@pytest.mark.asyncio
async def test_mcap_estimate_applies_the_split_multiplier_within_the_quarter(vintage, monkeypatch):
    await _phase_b_ready(vintage, monkeypatch, {"PBR": 10e6, "LNG": 1e6, "NEWCO": 1e6})
    # 1:2 forward split on 2024-02-14: shares double, unadjusted price halves from then on.
    _snap(vintage / "raw" / "splits" / "page-1.json.gz", {"results": [
        {"ticker": "PBR", "execution_date": "2024-02-14", "split_from": 1, "split_to": 2}]}, "2024-03-01")
    for d in FEB + [MAR1]:
        if d >= date(2024, 2, 14):
            rows = pa._read_raw(vintage / "raw" / "grouped_raw" / f"{d}.json.gz")["results"]
            for r in rows:
                if r["T"] == "PBR":
                    r["c"] = 7.5
            _w(vintage / "raw" / "grouped_raw" / f"{d}.json.gz", {"results": rows})
    est = pa.mcap_estimates(VINTAGE)
    assert est[(date(2024, 2, 13), "PBR")] == pytest.approx(150e6)
    assert est[(date(2024, 2, 14), "PBR")] == pytest.approx(150e6)          # continuous across the split


@pytest.mark.asyncio
async def test_mcap_audit_sample_is_deterministic_and_allocates_12_13(vintage, monkeypatch):
    await _phase_b_ready(vintage, monkeypatch, {"PBR": 21e6, "LNG": 1e6, "NEWCO": 1e8})
    a = pa.mcap_audit_sample(VINTAGE)
    assert a == pa.mcap_audit_sample(VINTAGE)
    parts = {p for rows in a.values() for p, *_ in rows}
    assert "band" in parts                                    # PBR: 21M x $15 = $315M, inside +-20%
    for rows in a.values():
        assert sum(1 for p, *_ in rows if p.startswith("sentinel")) <= pa.SENTINEL_PER_MONTH


def test_phase_b_ledger_has_its_own_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "ROOT", tmp_path)
    led = pa._open_ledger("v", "B")
    assert led.ceiling == pa.PHASE_B_CALL_CEILING and led.path.name == "request_ledger_phase_b.jsonl"
    led.close()
