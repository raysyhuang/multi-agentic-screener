# RECLAIM native lanes — historical replay (2026-10-04)

**Label: RECONSTRUCTED_DESCRIPTIVE. Not evidence, not a gate result.** Spec: RECLAIM_MAS_SPEC_v1.1. Registry row: `R-2026-10-reclaim-native`.

## Question

Neo + Hawk ruled WAIT on building the live collector because Reclaim had no usable evidence: zero clean canonical triggers and about 53 descriptive rows on 16 dates. Before anything gets built: does the reclaim → retest → confirm timing rule beat comparable episodes from the same pool that did not trigger, once it has a few hundred observations?

## Method

`scripts/reclaim_backtest.py` replays both native lanes through **the exact engine a collector would use** (`src/signals/reclaim.py` and `src/research/reclaim_replay.py` / `reclaim_report.py`):

- the native pools;
- the XNYS state machine;
- the raw-open decision;
- structural-stop fixed-horizon exits through `walk_exit`, at 10 bp per side with gap-through fills;
- frozen nearest-neighbour (NN) controls drawn from the same-day at-risk set.

The engine reproduces all four §9 worked examples exactly (LFUS, ACMR, DUOL, KSS) and matches Range's QNT ATR14/ADX14 values; both are pinned in `tests/test_reclaim_engine.py`.

- Window: 2022-01-03 → 2026-10-02 on Polygon adjusted daily bars. Drift triggers start in 2023 (252-session warm-up).
- Universe: 1,000 names taken from **today's** FMP screener (mcap ≥ $1B, ranked by dollar volume). Names flagged `isActivelyTrading=False` were dropped.
- "Clean-equivalent" cohort = replay-equivalent prehistory (the 20 prior sessions lie in-window and the symbol is absent from all of them) ∧ mechanical PASS ∧ no FMP earnings date in [e−5, e+5].

## Biases (all favour the strategy except the last)

- **Survivor-biased universe:** current members and current market cap replayed backwards. Names that died are missing, which inflates returns.
- **Earnings dates are not point-in-time.** The earnings cut is a sensitivity filter, not the live fail-closed gate.
- **Ticker identity only.** Range here is `RANGE_TECH_NATIVE`, not an exact Range V0.1 rebuild.
- 2022 Range triggers carry no SPY regime label (252-session vol warm-up).

## Result

Trigger ledger: 2,192 triggers. RANGE: 383 PASS, 256 SKIP_RISK, 85 SKIP_EXTENDED, 2 unfillable. DRIFT: 1,008 PASS, 345 SKIP_RISK, 109 SKIP_EXTENDED, 4 unfillable. Of these, 1,093 are clean-equivalent.

Clean-equivalent cohort; NN excess = trigger net minus mean of 3 NN controls (§7 primary):

| Lane | h | n | dates | mean net | median net | win | vs SPY | vs NN | stop-touch | date-cluster CI (net) |
|---|---|---|---|---|---|---|---|---|---|---|
| RANGE | 5 | 333 | 241 | −0.02% | −0.49% | 48% | −0.18 | −0.12 | 31% | [−0.50, +0.73] |
| RANGE | 10 | 330 | 239 | +0.30% | −2.15% | 42% | +0.36 | **+0.02** | 48% | [−0.68, +1.12] |
| RANGE | 20 | 323 | 235 | +1.17% | −3.54% | 37% | +0.35 | −0.03 | 60% | [−0.42, +3.14] |
| DRIFT | 5 | 758 | 427 | −0.22% | −0.54% | 44% | −0.43 | **−0.14** | 26% | [−0.58, +0.11] |
| DRIFT | 10 | 755 | 425 | −0.23% | −1.60% | 41% | −0.76 | **−0.20** | 43% | [−0.81, +0.16] |
| DRIFT | 20 | 750 | 422 | +0.06% | −2.65% | 37% | −1.42 | −0.32 | 57% | [−0.78, +0.64] |

- **DRIFT_G3_NATIVE has no edge, and it is wrong-signed.** NN excess is ≤ 0 at h5 and at h10, the h10 median is −1.6%, and it trails SPY in every year (2023–2026) and every regime. That meets Neo/Hawk's pre-registered **KILL_NO_EDGE** condition (at ≥30 clean h10 / ≥12 dates: both h5/h10 excess ≤ 0 with h10 median net ≤ 0) by a wide margin. It fails while survivor bias is pushing it up.
- **RANGE_TECH_NATIVE adds nothing over its controls.**
  - h10 NN excess is +0.02pp, and its sign flips by year: 2022 −1.5, 2023 +1.6, 2024 −1.7, 2025 0.0, 2026 +0.6.
  - The +0.30% h10 mean comes from five trades. Without the top five it is −0.13%; with five trimmed from each tail it is +0.04%.
  - The median is −2.1%.
  - Every CI spans 0.
- **The §7 spec flags:** PROMOTION is false in both lanes. KILL is also false in both, but only because h10 stop-touch sits just under the 50% bar (RANGE 48%, DRIFT 43%). Neo/Hawk's later gate drops that stop-touch condition from KILL_NO_EDGE.
- **Mechanism read:** both lanes behave like a positively skewed lottery. The typical trade loses (win rates 37–48%) and a few large winners carry the mean. The confirmation step does not separate winners from the episodes that never triggered.

## Implication

This supports Neo + Hawk's WAIT, and points further toward not building:

- **Do not build the native collector.** One lane is wrong-signed and the other is indistinguishable from its controls, on several hundred observations, under biases that favour it.
- **What stays untested:**
  - *Drift canonical, with its G1 growth / G2 valuation legs.* This is the "preserve G1/G2 through transport" point. The replay could not test it because MAS has no point-in-time fundamentals history. If anyone pursues Reclaim, this is the only version left open, and it belongs in the canonical lane.
  - *Range's pre-registered box-breakout sibling* (BOX V0.1, PARKED). It is a different mechanism. Testing it would need its own registry row and variant count, not a tweak to this one.
- **No parameter sweep was run, and none should be.** Tuning windows or bands on this replay is exactly the mined-filter outcome Neo warned about.

## Artifacts

- `outputs/research/reclaim/reclaim_backtest_2022-01-03_2026-10-02.json`: full tables, per-year and per-regime splits, provenance.
- The per-trigger rows CSV (~0.9 MB) is left local and regenerable: `python -m scripts.reclaim_backtest --start 2022-01-03 --end 2026-10-02`.
