# Local vs HTTP Engine Mode A/B Protocol

## Goal

Determine whether `engine_run_mode=local` is materially better/worse than legacy `engine_run_mode=http` for:

- reliability
- consistency / date alignment
- output equivalence (engine payloads and synthesis inputs)
- operational latency

This protocol isolates **execution mode** only. It is not a strategy-tuning test.

## Important Constraint (Current Code)

The multi-engine backtest orchestrator (`src/backtest/multi_engine/orchestrator.py`) uses local backtest adapters directly and does **not** switch on `engine_run_mode`.

Therefore, a fair `local vs http` A/B test must be run at the **collector/pipeline level** (Step 10 cross-engine collection + downstream synthesis), not via the current multi-engine backtest harness.

## Test Design

### Primary comparison

Compare two runs for the same `target_date`:

1. `engine_run_mode=local`
2. `engine_run_mode=http`

Keep everything else fixed.

### Controls (must match)

- Same commit SHA (MAS + collector + synthesis code)
- Same env vars except engine mode / engine URLs
- Same `target_date`
- Same FMP/Polygon keys and quotas available
- Same database snapshot (or run on a frozen clone if possible)
- Same KooCore-D and Gemini STST code versions (remote endpoints must match local folders)
- Same timeout settings (`engine_fetch_timeout_s`)

### Repetitions

Run each mode at least `N=3` times on the same `target_date` (to capture transient network variance).

## Metrics to Compare

### Reliability

- Engine response success rate (`2/2`, `1/2`, etc.)
- Quality rejections count
- Fallback usage count (KooCore timeout/fallback, degraded flags)
- Error types (`timeout`, `no_response`, `schema`, `quality_rejected`)

### Consistency

- Engine `target_date` alignment (per engine)
- Count of picks per engine
- Exact ticker overlap / Jaccard similarity by engine
- Payload hash equality (normalized JSON hash ignoring timestamps)
- Cross-engine synthesis input equivalence (pre-LLM)

### Downstream Effect

- Cross-engine synthesized pick set overlap
- Weights / confidence differences (if any)
- Validation gate pass/fail parity
- Telegram summary parity (same picks / different picks)

### Performance / Ops

- Step 10 duration
- Total run duration
- p50/p95 engine fetch latency (HTTP mode)
- CPU/memory pressure on app dyno/host (local mode)

## Acceptance Criteria (example)

Prefer `local` mode if all are true:

- Equal or better success rate / fewer degraded runs
- Better target-date alignment consistency
- Equivalent outputs (or explainable differences) on >= 95% of repeated runs
- Operational latency is acceptable for your SLA

Prefer `http` mode only if:

- Local mode causes frequent timeouts/resource saturation
- HTTP mode provides equal consistency with materially lower runtime impact

## Practical Execution Plan

### Phase 1: One-date deterministic A/B

Pick one recent date (e.g. `2026-02-20`) and run:

- `local` mode x3
- `http` mode x3

Collect and compare artifacts.

### Phase 2: Multi-date robustness A/B

Repeat on a small set of dates covering different regimes:

- bull day
- bear/choppy day
- high-volatility day

### Phase 3: Operational soak (optional)

Run daily for 1 week in shadow mode:

- production mode active
- alternate mode runs in parallel but does not publish alerts

## Required Instrumentation (Minimal)

To make this rigorous, add a collector comparison harness that records:

- mode (`local` / `http`)
- target date
- per-engine normalized payload hash
- per-engine pick count
- degraded/fallback flags
- collection duration
- failures list

Output format:

- JSON artifact per run in `artifacts/engine_ab/`
- optional DB table later if you want trend tracking

## Implemented Harness

Script:

- `scripts/compare_engine_collection_modes.py`

Behavior:

1. Accept `--target-date`
2. Run collector in `local` mode
3. Run collector in `http` mode
4. Normalize payloads
5. Compute hashes and diffs
6. Emit comparison report (JSON + console summary)

Example commands:

```bash
# Single A/B run on a fixed target date
./.venv/bin/python scripts/compare_engine_collection_modes.py --target-date 2026-02-20

# Repeat 3x to measure transient variance
./.venv/bin/python scripts/compare_engine_collection_modes.py --target-date 2026-02-20 --repeat 3

# Keep artifacts small (default), or include normalized payloads for deep diffing
./.venv/bin/python scripts/compare_engine_collection_modes.py --target-date 2026-02-20 --include-normalized-payloads
```

## Caveats

- HTTP mode only compares fairly if remote endpoints are pinned to the same engine code/data assumptions as local folders.
- Differences caused by remote app stale outputs or different deployment SHAs do **not** prove local mode is better/worse; they prove environments differ.
- LLM-based downstream synthesis can add nondeterminism; compare pre-LLM artifacts first.
- Current harness timeout is collector-level; local thread-backed engine work may outlive the timeout briefly. For strict wall-clock enforcement across repeated runs, use a subprocess wrapper (next improvement).

## Current Conclusion (before A/B execution)

No evidence currently shows the one-folder local integration is worse.

Recent failed tuning candidates were strategy-parameter issues, not an architecture-mode comparison.
