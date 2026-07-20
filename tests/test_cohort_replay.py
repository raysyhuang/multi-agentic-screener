"""Regression tests for scripts/cohort_replay.py — the share-class symbol alias.

Neo's finding: two PBR-A manual-sleeve rows were silently dropped because Polygon
uses PBR.A while the cohort uses the dash form. Lock the dash->dot fallback.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from scripts.cohort_replay import polygon_symbol_candidates, _poly_fetch


def test_symbol_candidates_class_share():
    # Dash class-share -> try dash then dot.
    assert polygon_symbol_candidates("PBR-A") == ["PBR-A", "PBR.A"]
    assert polygon_symbol_candidates("BRK-B") == ["BRK-B", "BRK.B"]


def test_symbol_candidates_plain_ticker():
    # No dash -> single candidate, no spurious alias.
    assert polygon_symbol_candidates("AAPL") == ["AAPL"]


class _FakePoly:
    """Returns data only for the dot form, empty for the dash form."""
    def __init__(self):
        self.calls = []

    async def get_ohlcv(self, sym, start, end):
        self.calls.append(sym)
        if sym == "PBR.A":
            return pd.DataFrame({"date": [date(2026, 5, 28)], "open": [10.0],
                                 "high": [10.5], "low": [9.5], "close": [10.0], "volume": [1000]})
        return pd.DataFrame()  # dash form has no data


@pytest.mark.asyncio
async def test_poly_fetch_falls_back_to_dot_form():
    poly = _FakePoly()
    df = await _poly_fetch(poly, "PBR-A", date(2026, 5, 1), date(2026, 6, 1))
    assert df is not None and not df.empty          # resolved via the alias
    assert poly.calls == ["PBR-A", "PBR.A"]          # tried dash first, then dot


@pytest.mark.asyncio
async def test_poly_fetch_no_alias_for_plain_ticker():
    poly = _FakePoly()
    df = await _poly_fetch(poly, "AAPL", date(2026, 5, 1), date(2026, 6, 1))
    assert df is None and poly.calls == ["AAPL"]     # only one attempt, no alias


def test_cli_defaults_mirror_live_settings():
    """The replay CLI must default every execution parameter from live settings —
    hardcoded defaults (trail 0/0, cost 5bp) each manufactured a fake
    live-vs-engine gap in the MR reconciliation."""
    from scripts.cohort_replay import build_arg_parser
    from src.config import get_settings

    s = get_settings()
    args = build_arg_parser().parse_args(["--cohort", "x.csv"])
    assert args.cost_bps == pytest.approx(s.slippage_pct * 10000)   # 10bp/side today
    assert args.trail_activate_pct == pytest.approx(s.trail_activate_pct)
    assert args.trail_distance_pct == pytest.approx(s.trail_distance_pct)


def test_exit_config_snapshot_derives_from_settings():
    """Every persisted signal must carry the exit config it was created under
    (provenance for exact historical replays — only FUTURE signals get this;
    the 90d MR cohort predates it)."""
    from types import SimpleNamespace
    from src.main import build_exit_config_snapshot

    s = SimpleNamespace(
        slippage_pct=0.001, trail_activate_pct=0.5, trail_distance_pct=0.3,
        score_tiered_stops_enabled=True, partial_tp_enabled=False,
        sniper_time_stop_days=1, entry_gap_max_atr=0.2,
    )
    snap = build_exit_config_snapshot(s)
    assert snap == {
        "slippage_pct": 0.001,
        "trail_activate_pct": 0.5,
        "trail_distance_pct": 0.3,
        "score_tiered_stops_enabled": True,
        "partial_tp_enabled": False,
        "sniper_time_stop_days": 1,
        "entry_gap_max_atr": 0.2,
    }
