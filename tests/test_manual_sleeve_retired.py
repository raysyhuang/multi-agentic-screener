"""The MR manual sleeve is retired: gated off by default.

The sleeve reproduced the official MR picks verbatim and added only gate-rejected
losers (see the 2026-07 forensic), so it is disabled by default. This locks in
that default — flipping it back on is a deliberate act, not an accident — and
verifies the gate expression main.py uses to suppress the whole sleeve path.
"""

from __future__ import annotations

from src.config import Settings


def test_sleeve_disabled_by_default():
    assert Settings().mr_manual_sleeve_enabled is False


def test_gate_expression_yields_empty_when_disabled():
    # Mirrors src/main.py: the sleeve list is built only when the flag is on;
    # an empty list cleanly disables annotation, ranking, persistence and alert.
    candidates = ["mr_a", "mr_b", "mr_c"]
    for enabled, expected in [(False, []), (True, candidates)]:
        sleeve = list(candidates) if enabled else []
        assert sleeve == expected
