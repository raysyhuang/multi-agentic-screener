"""The candidate export must carry the ordering selection actually used.

PR #76 added `strategy_rank`, `rejection_stage` and the contemporaneous slot
state to `Candidate` expressly to make selection auditable. The exporter was
never wired to emit them: it re-sorted by score, renumbered from 1, and shipped
that derived ordering as `rank`. So the columns existed in the database and did
not exist for anyone consuming the published artifact.

The cost was not hypothetical. An audit built on the exported rank concluded
that rank-1 candidates were being skipped and that unpicked candidates
outperformed picked ones. Both were artifacts of a correlation-dropped name
holding derived rank 1 with picked=false.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class _Candidate:
    def __init__(self, **kw):
        defaults = dict(
            run_date=date(2026, 8, 1), ticker="AAA", signal_model="sniper",
            composite_score=90.0, strategy_rank=None, selection_stage_reached=None,
            rejection_stage=None, rejection_reason=None, slots_total=None,
            slots_occupied=None, slots_available=None, correlated_with=None,
            correlation=None,
        )
        defaults.update(kw)
        for k, v in defaults.items():
            setattr(self, k, v)


def _export(candidates, signals=(), outcome_by_sig=None):
    from export_dashboard_data import _candidates_payload  # noqa: PLC0415

    return _candidates_payload(candidates, signals, outcome_by_sig or {})


def test_the_order_selection_used_is_exported_not_a_score_resort():
    """A correlation-dropped name can top the score ordering and not be picked.

    This is the shape that broke the audit: score_rank 1, strategy_rank 2,
    picked=false — while the name selection actually ranked first was reported
    at rank 2.
    """
    rows = _export([
        _Candidate(ticker="DROPPED", composite_score=99.0, strategy_rank=2,
                   rejection_stage="correlation", rejection_reason="correlation_filtered"),
        _Candidate(ticker="TAKEN", composite_score=95.0, strategy_rank=1),
    ])
    by_ticker = {r["ticker"]: r for r in rows}

    assert by_ticker["DROPPED"]["score_rank"] == 1, "score ordering should still be reported"
    assert by_ticker["DROPPED"]["strategy_rank"] == 2
    assert by_ticker["TAKEN"]["strategy_rank"] == 1, (
        "the name selection ranked first must be exported as rank 1"
    )
    assert "rank" not in by_ticker["TAKEN"], (
        "a bare `rank` field is ambiguous — the two orderings must be named apart"
    )


def test_rejection_cause_is_exported_so_picked_false_is_not_four_causes_in_one():
    from export_dashboard_data import _candidates_payload  # noqa: F401

    rows = _export([
        _Candidate(ticker="QUOTA", strategy_rank=3,
                   rejection_stage="quota", rejection_reason="below_quota"),
        _Candidate(ticker="CAP", strategy_rank=4, composite_score=80.0,
                   rejection_stage="capacity", rejection_reason="capacity_censored",
                   slots_total=3, slots_occupied=3, slots_available=0),
        _Candidate(ticker="CORR", strategy_rank=5, composite_score=70.0,
                   rejection_stage="correlation", rejection_reason="correlation_filtered",
                   correlated_with="QUOTA", correlation=0.87),
    ])
    by_ticker = {r["ticker"]: r for r in rows}

    assert by_ticker["QUOTA"]["rejection_stage"] == "quota"
    assert by_ticker["CAP"]["rejection_stage"] == "capacity"
    assert by_ticker["CAP"]["slots_available"] == 0
    assert by_ticker["CORR"]["correlated_with"] == "QUOTA"
    assert by_ticker["CORR"]["correlation"] == pytest.approx(0.87)
    assert len({r["rejection_stage"] for r in rows}) == 3, (
        "distinct rejection causes must remain distinguishable in the export"
    )


def test_an_unrecorded_rank_exports_as_null_not_as_a_guess():
    """NULL means 'not recorded', never 'presumed first'.

    Rows written before the ledger began recording carry no strategy_rank.
    Filling that with the score ordering would reintroduce the original defect
    while looking like data.
    """
    rows = _export([_Candidate(ticker="OLD", strategy_rank=None)])

    assert rows[0]["strategy_rank"] is None
    assert rows[0]["score_rank"] == 1


# ── export -> audit integration ──────────────────────────────────────────────

def test_the_audit_consumes_the_export_without_a_keyerror(tmp_path, capsys):
    """The export and its only consumer must agree on field names.

    Renaming `rank` to `strategy_rank`/`score_rank` fixed the ambiguity and
    simultaneously broke `rank_quality_audit.py`, which still read `c["rank"]`.
    A valid new export raised KeyError — the export's stated purpose is to feed
    this audit, so a schema change that the consumer cannot read is not a fix.

    Exercises the three shapes that make `picked: false` ambiguous: a
    correlation-dropped name, a capacity/slot-censored name, and a below-quota
    name.
    """
    import json
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from rank_quality_audit import audit_live  # noqa: PLC0415

    rows = _export([
        _Candidate(ticker="TAKEN", composite_score=95.0, strategy_rank=1,
                   selection_stage_reached=True),
        _Candidate(ticker="DROPPED", composite_score=99.0, strategy_rank=2,
                   rejection_stage="correlation", rejection_reason="correlation_filtered",
                   correlated_with="TAKEN", correlation=0.91),
        _Candidate(ticker="CAPPED", composite_score=88.0, strategy_rank=3,
                   rejection_stage="capacity", rejection_reason="capacity_censored",
                   slots_total=3, slots_occupied=3, slots_available=0),
        _Candidate(ticker="QUOTA", composite_score=80.0, strategy_rank=4,
                   rejection_stage="quota", rejection_reason="below_quota"),
        # Pre-ledger row: no strategy_rank. Must be reported as unknown, never
        # coerced into the score ordering.
        _Candidate(ticker="OLD", composite_score=70.0, strategy_rank=None),
    ])
    export = {
        "candidates": rows,
        "trades": {"sniper|mas_official": [
            {"ticker": t, "pnl_pct": 1.0} for t in ("TAKEN", "DROPPED", "CAPPED")
        ]},
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps(export))

    audit_live(str(path))            # must not raise
    out = capsys.readouterr().out

    assert "KeyError" not in out
    assert "5 candidate rows" in out
    assert "1 row(s) carry no strategy_rank" in out, (
        "an unrecorded rank must be reported as unknown, not silently ranked"
    )


def test_the_audit_never_falls_back_to_the_score_ordering():
    """Substituting score_rank for a missing strategy_rank rebuilds the defect.

    It would look like data and read like a rank, while being the ordering the
    pipeline did not use — the exact confusion that produced the false
    "rank-1 candidates are skipped" conclusion.
    """
    source = (Path(__file__).resolve().parents[1] / "scripts" / "rank_quality_audit.py").read_text()
    live = source[source.index("def audit_live"):]

    assert 'c["rank"]' not in live, "a bare `rank` read has returned"
    assert 'c.get("score_rank")' not in live and 'c["score_rank"]' not in live, (
        "the live arm must not read the score ordering as a selection rank"
    )
    assert 'c.get("strategy_rank") is not None' in live
