"""H4 helpers — synthetic, no network. Pins that the predecessor lookup only
looks BACKWARD within the quarter-gap window, and that the difference CI
collapses to the point estimate when every observation shares one entry date."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.h4_consecutive_beats import diff_cluster_ci, tag_previous


def _ev(ticker, report_date, surprise):
    return {"ticker": ticker, "report_date": report_date, "surprise": surprise}


def test_previous_report_is_the_immediately_preceding_quarter_only():
    evs = tag_previous([
        _ev("AAA", "2025-01-30", 15.0),
        _ev("AAA", "2025-04-29", 12.0),     # 89 days later -> predecessor is the 15.0
        _ev("AAA", "2026-01-28", 20.0),     # 274 days after the last -> no predecessor in window
        _ev("BBB", "2025-04-29", 30.0),     # another ticker never leaks across
    ])
    by = {(e["ticker"], e["report_date"]): e["prev_surprise"] for e in evs}
    assert by[("AAA", "2025-01-30")] is None
    assert by[("AAA", "2025-04-29")] == 15.0
    assert by[("AAA", "2026-01-28")] is None
    assert by[("BBB", "2025-04-29")] is None


def test_a_later_report_never_becomes_a_predecessor():
    evs = tag_previous([_ev("AAA", "2025-04-29", 12.0), _ev("AAA", "2025-01-30", 99.0)])
    first = next(e for e in evs if e["report_date"] == "2025-01-30")
    assert first["prev_surprise"] is None       # input order must not matter


def test_difference_ci_collapses_on_a_single_shared_date():
    a = pd.DataFrame({"entry_date": ["2026-01-05"] * 3, "x": [2.0, 4.0, 6.0]})
    b = pd.DataFrame({"entry_date": ["2026-01-05"] * 2, "x": [1.0, 1.0]})
    point, lo, hi = diff_cluster_ci(a, b, "x", n_boot=500)
    assert point == pytest.approx(3.0)
    assert lo == pytest.approx(3.0) and hi == pytest.approx(3.0)
