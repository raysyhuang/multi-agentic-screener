# Quality Veto Layer — Pre-Ranking Fundamental Gates (2026-08-15)

## Question

Ranker IC ≈ 0 (see `rank_quality_FINDINGS.md`). The gates are the alpha. Can we add
pre-ranking vetoes that catch story-quality issues the universe filter misses?

Three hypothesis cases from the 2026-08-10 weekend tape:
- **GTLB**: extended tape, too close to recent highs vs Street PT
- **ONDS**: dilution shock, share count ~2.6–3.3× YoY
- **BMNR**: not a fundamental buy at ~1× NAV (ATM-funded treasury, structural)

The existing universe filter (`src/signals/filter.py`) only does price, volume,
exchange, ETF/fund, ticker format, and split artifacts. It never asks "is this a
story name."

## What Was Built

A **quality veto layer** (`src/signals/veto.py`) that runs **before ranking** and
applies three look-ahead-safe vetoes:

### 1. Extended Tape (`skip_reason="veto_extended"`)

**Logic**: Veto if close >= (20d high − 0.1×ATR cushion).

**Look-ahead safe**: Uses only the ticker's own daily OHLCV history.

**Fail-open rules**:
- DataFrame is None, empty, or has fewer than 34 bars (20 + 14 for ATR)
- Close/high data is missing or invalid
- ATR is invalid or zero

**Example**: A stock pinned at its 20-day high is extended and at risk of mean
reversion or consolidation. The signal may still work, but the entry setup is
less favorable.

### 2. Dilution / Share-Count Shock (`skip_reason="veto_dilution"`)

**Logic**: Veto if (shares_now / shares_1y_ago) > 2.0 (default threshold).

**Look-ahead safe**: Uses historical fundamental data that was already reported.

**Fail-open rules**:
- fundamental_data is None or missing share count fields
- Only current shares available (no historical comparison)
- Share count is zero or invalid

**Example**: ONDS-class names where shares outstanding roughly doubled or tripled
YoY. This is often ATM offerings, PIPE deals, or dilutive M&A. The fundamental
thesis is compromised because per-share metrics (EPS, book value) are no longer
comparable across periods.

**Data sources**: Tries `profile.sharesOutstanding`, `profile.shares`, and
`ratios[].weightedAverageShsOut` / `weightedAverageShsOutDil`. Compares Q0 (current)
against Q-4 (1 year ago) if ratios is a time-series list.

### 3. Data Sanity (`skip_reason="veto_data_sanity"`)

**Logic**: Veto if two snapshots have the same metric and disagree by > 10%
(default tolerance). Checks revenue, shares outstanding, and EPS.

**Look-ahead safe**: Pure comparison of two already-provided snapshots (e.g., FMP
profile vs FMP ratios).

**Fail-open rules**:
- Either snapshot is None
- Metrics are missing in one or both snapshots
- Metrics are zero or invalid (can't compute relative difference)

**Example**: FMP fields have been wrong (HOOD Q2 rev, COHR FY26 annual = Q4,
share counts). If profile says revenue = $100M and ratios say revenue = $200M,
something is stale or miskeyed. The thesis grounding is unreliable.

**Comparison**: Currently compares FMP `profile` vs FMP `ratios` (both from the
same provider). Future work could compare FMP vs Massive/10-Q fields if those
become available in the pipeline.

## Integration

**Where**: `src/main.py`, Step 5.5 (after signal scoring, before ranking).

**Shadow mode (default)**: Vetoed signals are kept in the output with
`veto_reason` attached. They persist to the database with `skip_reason` set to
the veto label (`veto_extended`, `veto_dilution`, `veto_data_sanity`). All stats
queries already filter `skip_reason.is_(None)`, so shadow rows are automatically
excluded from official performance statistics (same pattern as validation gate
shadows).

**Hard veto mode** (`quality_veto_shadow_only=False`): Vetoed signals are removed
from the pick stream entirely. Default is `True` (shadow-only) to collect evidence
before any live flip.

**Settings flags**:
- `quality_veto_enabled` (default `True`): Enable the veto layer
- `quality_veto_shadow_only` (default `True`): Keep vetoed signals as shadows

## Fail-Open Design

All three vetoes fail open when data is missing or insufficient. This is by design:

- **The pipeline must not break** if FMP changes a field name or returns no ratios
- **A data outage must not veto everything** and silently turn off the strategy
- **Short history must not block** (e.g., a recent IPO with <34 bars)

The fail-open rate will be visible in the veto funnel logs. If extended veto is
failing open on 80% of names, that's a data coverage issue, not a veto logic bug.

## Testing

**Unit tests** (`tests/test_quality_veto.py`, green in CI, no network, no parquet):

1. **Extended tape**: Synthetic OHLCV with close pinned at 20-day high → veto fires
2. **Extended tape**: Close in mid-range → no veto
3. **Extended tape**: Short history (<34 bars) → fail open
4. **Dilution**: Shares 3× YoY → veto fires
5. **Dilution**: Shares 1.1× YoY (normal growth) → no veto
6. **Dilution**: No historical share data → fail open
7. **Data sanity**: 2× revenue disagreement → veto fires
8. **Data sanity**: 2% revenue disagreement (within tolerance) → no veto
9. **Data sanity**: One side missing a metric → fail open
10. **Shadow mode**: Vetoed signals are kept with `veto_reason` attached
11. **Hard veto mode**: Vetoed signals are removed from output
12. **Invariant**: Non-vetoed signals pass through unchanged in shadow mode

All tests use synthetic data (no external dependencies). This ensures the veto logic
itself is testable and debuggable without live data or network access.

## No Backtest / Return Claims

**This is a research-first PR.** No backtest win rate or return numbers are claimed.

The veto layer is **shadow-only by default**. A live flip requires
selected-vs-selected evidence: ≥30 shadow rows where the veto label is present,
and a per-veto comparison of shadowed picks vs official picks that passed the
veto (same bar as choppy sniper, STRATEGY_REVIEW §0.4).

**Why no backtest?** The three vetoes depend on:
1. Real-time fundamental data (share counts, revenue) that backtests typically
   lack or smooth over
2. Intraday tape context (extended = close near high) that point-in-time
   snapshots may not capture reliably
3. Data sanity checks that compare provider snapshots — a backtest would need to
   replay historical FMP/Massive disagreements, which are not archived

Running a backtest with synthetic "always fail open" data would produce a false
"veto has no edge" conclusion. The evidence will come from live shadow tracking.

## No Live Flip Until Evidence

**Default**: `quality_veto_shadow_only=True`. Vetoed signals are persisted with
`skip_reason=veto_*` and excluded from all performance stats.

**Live flip criteria** (same as choppy sniper, §0.4):
- ≥30 shadow rows per veto reason
- Per-veto selected-vs-selected comparison (vetoed shadows vs official picks that
  passed the veto on the same days)
- Win rate or return edge demonstrated in shadow data before any veto is allowed
  to block a live pick

Until that bar is met, the vetoes are **shadow-only tracking** and have zero
impact on official picks or performance statistics.

## What This Does NOT Do

- **Does NOT change ranking weights** (ranker IC ≈ 0, so tuning weights is futile)
- **Does NOT widen picks** (that is a separate capital allocation question)
- **Does NOT flip choppy ×0.6** (regime down-weighting is orthogonal)
- **Does NOT change live PEAD/sniper/MR caps** (position sizing is separate)
- **Does NOT invent a new signal** (this is a gate, not a scorer)

The veto layer is a **quality gate** that catches names the universe filter
misses. It is not a scoring model, not a ranking input, and not a regime rule.

## Next Steps

1. **Accumulate shadow data**: Run live with `quality_veto_enabled=True` and
   `quality_veto_shadow_only=True` for ≥30 vetoed picks per reason.

2. **Selected-vs-selected analysis**: Compare vetoed shadows (what the gate would
   have blocked) against official picks that passed the veto on the same days.

3. **Per-veto decision**: Each veto gets its own evidence threshold. Extended tape
   may have edge while dilution does not (or vice versa). They are independent
   gates, not a bundle.

4. **Live flip (if justified)**: Set `quality_veto_shadow_only=False` for vetoes
   that pass the evidence bar. Others stay shadow-only or get disabled.

## Validation-Blocked Picks (§0.4 Addendum)

**Not implemented in this PR.** STRATEGY_REVIEW §0.4 asked for validation-blocked
picks to get `skip_reason="validation_blocked"` instead of being dropped on the
floor, reusing the same shadow persist path.

That would require refactoring the validation gate's `blocked_picks` list to
persist with a label, and the veto layer already demonstrates the pattern. If the
validation gate's persist is the same as the veto layer's (which it is), then
adding `skip_reason="validation_blocked"` is a small, obvious change and can land
in a follow-up. Bundling it here would widen the scope beyond the quality veto
layer.

If that work is urgent, it is a 10-line change to `src/main.py` around line 2113
where `blocked_picks` are shadow-booked: change `skip_reason=SHADOW_SKIP_REASON`
to `skip_reason="validation_blocked"` and lock it with a test. This PR establishes
the pattern; applying it to the validation gate is mechanical.

## Summary

Three look-ahead-safe vetoes (extended tape, dilution, data sanity) run before
ranking and mark candidates with `skip_reason` in shadow mode. All vetoes fail
open on missing data. Unit tests green. No backtest claims. No live flip until
selected-vs-selected evidence (≥30 rows per veto).

The veto layer is a quality gate, not a signal. It catches story-quality issues
the universe filter misses (extended names, dilution shocks, data disagreements).
Default is shadow-only; official picks are unchanged until evidence justifies a
live flip.
