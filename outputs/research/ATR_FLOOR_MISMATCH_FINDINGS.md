# Sniper ATR floor: backtests and production sample different universes

**Status:** confirmed independently twice — Victor (VPS Boston) and Claude Code (Ray's Mac).
**Date:** 2026-08-16. **Repo state:** `origin/main` = `fd14b31e`.
**Disposition:** park the fix past Monday 2026-08-17. Record the finding now so it is not
rediscovered a third time.

## The defect

`score_sniper()` takes `atr_pct_floor` as a parameter and rejects any candidate below it
(`src/signals/sniper.py:82`). The value it receives depends entirely on the caller:

| Site | Value | Effect |
|---|---|---|
| `src/config.py:226` | `sniper_atr_pct_floor = 5.0` | the configured setting |
| `src/main.py:1275` | `atr_pct_floor=settings.sniper_atr_pct_floor` | **production reads 5.0 — correct** |
| `src/signals/sniper.py:57` | `atr_pct_floor: float = 3.5` | signature default |
| `src/research/signal_backtest.py:635` | `atr_pct_floor: float = 3.5` | **backtest default** |
| `src/research/signal_backtest.py:752` | `params.get("atr_pct_floor", 3.5)` | **backtest fallback** |

Every backtest run through `signal_backtest.py` without explicitly passing the setting admits
candidates with ATR% between **3.5 and 5.0** that production rejects outright.

## Why it matters

This is **not** a severity or optimism issue. It is a population issue: a sniper backtest is
computed over a **different universe** than the one production trades. Trade counts, win rates,
expectancy and drawdown from those runs are not estimates of production behaviour with error
bars — they describe a strategy that was never deployed.

It is also **distinct from the known large-cap cache problem**, and the two compound:

- The cache issue says the backtest universe **under-samples** high-ATR names (median ATR% 2.28
  against a live cohort's higher dispersion).
- This issue says the backtest **admits a band production refuses**.

Different mechanisms, same direction. Both inflate the apparent tradability of a model whose
headline result has already been retracted once as a fill artifact.

## Consequence for the acceptance criteria

It is an independent argument for the rule already in
`docs/paper_sleeve_acceptance_criteria.md` that backtest bands are **not** admissible comparators.
That rule was justified on the cache problem alone. It survives on this ground too, and would
survive even if the cache were fixed tomorrow.

## This is the pattern CLAUDE.md already warns about

> "Replay/backtest harnesses MUST mirror the live tracker's FULL execution config from settings…
> Default these from `get_settings()`, never hardcode."

Three parameter-mismatch artifacts were caught in the 2026-07 MR reconciliation, each of which
manufactured a fake live-vs-engine gap. This is a fourth instance of the same class, in a file the
reconciliation did not cover.

## Fix, when it is picked up

Delete the hardcoded `3.5` at all three sites and default from `get_settings()`, matching
`main.py:1275`. Then **re-run any sniper backtest whose numbers are still quoted anywhere** — the
old ones describe a different universe and cannot be adjusted after the fact, for the same reason
slippage cannot be adjusted post-hoc (changing the admitted population changes which trades exist,
not just their returns).

Until that rerun exists, no sniper backtest figure from `signal_backtest.py` should be quoted
without stating the ATR floor it actually ran at.
