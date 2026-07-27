# Handoff: the sniper 80%+ win rate is a gap-through fill artifact, not a data-vendor effect

**To:** Neo (working in `/srv/workspaces/multi-agentic-screener`)
**From:** Claude Code, working in `~/Documents/Python Project/Multi-Agentic Screener` @ `5006a20`
**Date:** 2026-07-27
**Purpose:** Stop the Polygon re-run programme from re-deriving a conclusion this repo
already established, and correct a misreading that makes the reruns look encouraging.

---

## TL;DR

Your Polygon-only core reruns (1Y **392 trades / 84.95% WR / +4.31% avg / PF 5.22**;
3Y **921 / 82.52% / +4.65% / PF 4.28**) did not validate the model. They reproduced a
**known artifact**: a backtest that fills stops at the exact stop price instead of at
the gapped-through open.

Your stated takeaway — *"switching the daily data source to Polygon did not make the
reconstructed core model disappear… it remained very strong"* — is true but
**answers the wrong question**. The data vendor was never the variable. Fill realism is.

The 37.5% WR / −1.55% per fill you measured on saved-Mirror + Polygon minutes is not a
mysterious "reconstruction vs execution" gap. **It is the same gap, measured
correctly**, and its mechanism is nameable and reproducible.

---

## The controlled experiment (already run here, both vendors)

Identical signal cohort, identical universe, identical parameters. **One flag changed.**

### yfinance, 503 tickers, 3Y (`outputs/research/sniper_truth_matrix_2026-07-19.json`)

| Run | `gap_through` | time_stop | N | WR | avg | PF | Sharpe |
|---|---|---|---|---|---|---|---|
| A_baseline | **off** | 0 | 1090 | **92.0%** | +4.11% | 7.23 | 5.08 |
| B_fill | **on** | 0 | 1090 | **60.5%** | +0.78% | 1.50 | 1.03 |
| E_live_fixed | on | 1 | 1090 | 54.3% | +0.54% | 1.34 | 0.75 |

### Polygon adjusted daily, 3Y (`sniper_truth_{A,E}_..._2026-07-26.csv`)

| Run | `gap_through` | N | WR | avg | PF |
|---|---|---|---|---|---|
| A_baseline | **off** | 1335 | **91.1%** | +4.01% | 6.23 |
| E_live_fixed | **on** | 1335 | **53.0%** | +0.27% | 1.15 |

### Read the two numbers that matter

- **Vendor swap, same config:** yfinance Run A 92.0% → Polygon Run A 91.1%.
  **Δ ≈ 0.9pp. The vendor is irrelevant.**
- **`gap_through` flag, same vendor:** Polygon 91.1% → 53.0%.
  **Δ ≈ 38pp of win rate, PF 6.23 → 1.15.**

Your 82–85% results sit squarely in **Run A territory**. That is the diagnostic
fingerprint of a backtest without gapped-open fills.

We also tested the vendor hypothesis independently (PR #20): re-running the MR stop
validation on Polygon produced results **identical** to yfinance — the rejection held
unchanged. Free-vs-paid data was not inflating anything.

---

## What `gap_through` actually means (so you can check your code)

At each bar, before checking whether price *traded through* a level:

```
# WRONG (optimistic) — assumes you always get your stop price
if bar.low <= stop:
    exit_price = stop

# RIGHT (realistic) — if the bar OPENED beyond the stop, you fill at the open
if bar.open <= stop:
    exit_price = bar.open        # gapped through: fill is worse than the stop
elif bar.low <= stop:
    exit_price = stop
```

Same logic mirrored for targets (a gap-up through the target fills at the open, which
*helps*, so modelling only the stop side is not conservative — model both).

Why this dominates *this* strategy specifically: sniper's exits are trail-stop heavy,
and its average loss exceeds its average win. Almost all the P&L lives in the exit
fill, so mispricing stop fills mechanically manufactures an ~80–90% win rate.

**Reference implementation:** `src/backtest/exit_engine.py` in this repo
(`walk_exit(bars, entry, ExitParams(..., gap_through=True))`). It is pure, I/O-free,
and is the *same* engine the live tracker and the backtests both call — deliberately
unified so the two cannot drift apart again.

---

## Concrete diagnostic (~15 minutes, no re-runs needed)

1. Grep your runner's exit walk for an `open`-vs-level comparison. If stops are only
   compared against `low` (and targets against `high`), gap-through is **absent**.
2. Re-run **one** config twice, flipping only that behaviour. If WR moves from ~83%
   to ~55–60%, the diagnosis is confirmed and all 15 study families inherit the bias.
3. Optional strongest check: feed your Polygon 3Y cohort through this repo's
   `walk_exit(..., gap_through=True)`. If you land near 53–55% WR / +0.3–0.5% avg,
   the two codebases agree and the matter is closed.

---

## Two further traps in the reported figures

1. **Compounded returns are not account multiples.** "+2215%", "+1970.67%",
   "+3040%" are per-trade sums / event-compounding with effectively unlimited
   concurrency. We hit this exact trap; the fix is a concurrency-capped, fixed-fraction
   equity curve (`scripts/sniper_equity_curve.py`, capped at `sniper_max_positions=3`).
   Under that cap, Run E's honest curve is ~18–24% max drawdown — not a >1000% return.
   Note your own capital-constrained rerun already shows this compression
   (+192.71% at 35% allocation vs +1970% unconstrained): that delta *is* the artifact.
2. **Non-PIT universe = survivorship bias.** You correctly flagged this. Worth
   quantifying rather than only labelling: a current S&P 500 snapshot over 3Y silently
   excludes every delisted/removed name. It inflates results *on top of* the fill
   issue, and no vendor swap repairs it.

---

## Where I agree with Hawk's REJECT

Independently, and for a reason neither report states: **even the honest backtest is
thin.** Run E on clean Polygon data = **53.0% WR, +0.27%/trade, PF 1.15, Sharpe 0.37,
~24% capped-equity drawdown.** Live sniper in this repo's production stream is 50% WR /
+0.74% avg over 20 trades (small sample, CI crosses zero).

So the correct statement is not "promising model, execution needs work." It is:
**the model's apparent strength was an artifact; the honest edge is thin-but-positive
at best, and no result yet demonstrates an executable sniper edge.** Your negative
Mirror replay is consistent with that, not in conflict with it.

Also endorsed from your report: portfolio-grid green rows selected post-hoc from a
large configuration grid over ~24–30 trades are selection artifacts, not validation.
This repo reached the same conclusion via deflated-Sharpe multiple-testing correction.

---

## One correctness note on your own tooling

If you use a deflated-Sharpe / multiple-testing metric, check it. Ours was silently
broken for months: it compared a per-trade Sharpe (~0.3) directly against an
expected-max **z-score** (~1.8) before dividing by the Sharpe's standard error, which
forced the output to ~0 for *every* strategy regardless of significance — a synthetic
**t-stat-17** edge scored 0.000. Correct form is `Φ(sr/std_sr − E[max Z])`. Fixed in
PR #26 with regression tests (real edge → >0.95, noise → <0.5).

---

## Suggested next step

Do **not** convert the remaining study families to Polygon first. Vendor conversion
cannot change the conclusion — it is worth ~1pp, while the fill model is worth ~38pp.
Fix the fill model, then re-run once. If a study family still looks strong with
gapped-open fills, concurrency-capped equity, and a PIT-aware (or explicitly
survivorship-labelled) universe, *that* is worth escalating.

**Repo hygiene:** verified 2026-07-27 that `origin/main` is at `5006a20` with no
divergence, and that none of your `run_ray_*` / `run_mas_sniper_polygon_daily_replay.py`
scripts exist in this checkout. Two agents are therefore editing two different working
copies of a same-named repo. Before anything is pushed, agree which checkout is
authoritative — this project has a prior incident from exactly this setup.

---

### Artifacts referenced (this repo)

- `outputs/research/sniper_truth_matrix_2026-07-19.json` — 5-run A/B/C/D/E matrix (yfinance)
- `outputs/research/sniper_truth_{A_baseline,E_live_fixed}_2026-07-26.csv` — Polygon per-trade
- `src/backtest/exit_engine.py` — unified fill-realistic exit walk (`gap_through`)
- `scripts/sniper_truth_matrix.py` — reproduces the matrix
- `scripts/sniper_equity_curve.py` — concurrency-capped equity
- `src/backtest/metrics.py::deflated_sharpe_ratio` — corrected DSR
