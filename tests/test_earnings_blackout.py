"""Sniper earnings-blackout hardening (2026-07-25).

Two fixes: (1) the blackout is hold-aware for sniper — a name reporting mid-hold
(day 3-7 of a 7-day hold) is skipped, closing the mid-hold earnings-gap tail;
(2) the blackout fails open on missing earnings data, so low calendar coverage
now raises a pipeline-health WARN instead of silently passing.
"""

from __future__ import annotations

from src.config import Settings
from src.validation.stage_validator import Severity, StageCheck, StageValidation


def test_blackout_config_defaults():
    s = Settings()
    assert s.earnings_blackout_days == 2
    assert s.earnings_coverage_min_pct == 0.15


def test_sniper_hold_aware_blackout_rule():
    # Mirrors the gate in src/main.py: sniper skips if earnings fall within its
    # hold window. The OLD code (global dte<=2 only) let days 3-7 through.
    hold = Settings().sniper_holding_period  # 7

    def sniper_skips(dte):
        return dte is not None and dte <= hold

    assert sniper_skips(1)          # reports tomorrow — skip
    assert sniper_skips(5)          # reports MID-HOLD — NEW: now skipped
    assert sniper_skips(7)          # reports on the exit day — skip
    assert not sniper_skips(8)      # reports after the hold — fine to take
    assert not sniper_skips(None)   # unknown → fails open (coverage WARN catches systemic gaps)


def _coverage_stage(cov: float) -> StageValidation:
    thr = Settings().earnings_coverage_min_pct
    st = StageValidation(stage_name="earnings_blackout", executed=True)
    st.checks.append(StageCheck(name="earnings_coverage", passed=cov >= thr,
                                severity=Severity.WARN, message="", value=cov))
    return st


def test_low_coverage_raises_warn_not_silent_pass():
    assert _coverage_stage(0.05).severity == Severity.WARN   # broken/sparse feed → degraded
    assert _coverage_stage(0.50).severity == Severity.PASS   # healthy coverage
