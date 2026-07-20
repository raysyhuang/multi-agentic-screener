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
