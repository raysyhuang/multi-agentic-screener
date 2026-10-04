# RECLAIM collector — must-fix before any revival

Parked draft (`36146f2`). Codex review, 2026-10-04, found these defects. They do not affect official picks while the flag is off, but each one would corrupt Reclaim's own record:

1. **Partial runs counted as covered.** `MIN_FRACTION_WITH_C_BAR = 0.50` lets a fetch missing up to half the universe persist `status="OK"`. §4A says a partial run leaves C uncovered. Require full coverage, or persist `PARTIAL` and never `OK`.
2. **Forward reporter drops the age dimension.** `scripts/reclaim_shadow_report.py` starts its session grid 10 days before the earliest trigger, so `_age()` returns None for episodes up to 60 sessions older, and `nn_controls` drops age for VALID triggers. Persist trigger age, or start the grid at the earliest episode start.
3. **Turning the flag off strands clones.** The tracker only runs while `reclaim_shadow_enabled` is on. Gate only new triggers on the flag; run the tracker whenever unresolved reclaim rows exist.
4. **Earnings capture uses the historical endpoint** (`get_earnings_surprise`). It rarely brackets e+5, so timely snapshots end up `BLOCK_EARNINGS_UNKNOWN`. Capture historical + upcoming dates and persist the coverage.
5. **Still unbuilt:** persistence/idempotency tests, `alembic upgrade head` + `alembic check` on 0003, and the flag-on byte-identical regression (spec gate 23).
6. **Stale engine.** This branch carries the engine as of `de07ff2`. Rebase onto `feat/reclaim-shadow` for the frozen-KILL and terminal-bar fixes.
