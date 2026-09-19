# H4 — drift after a second consecutive big beat: REJECTED at G1 (2026-09-19, v2 after Codex review)

**Verdict: REJECTED.** Every variant fails condition (d), the requirement that consecutive beats drift *more* than first-time beats. The last row below even points the other way.

| Run | Consecutive − first | Date-cluster CI |
|---|---|---|
| Registered run (v1): predecessors taken from inside the price window only | +0.005pp | [−0.53, +0.53] |
| v2: predecessors from the full report history, registered timing | +0.03pp | [−0.47, +0.52] |
| v2: full history, look-ahead-free timing | **−0.36pp** | [−0.84, +0.11] |

On its own, the consecutive cohort earns +0.26% to +0.35% per 20 sessions. That fails the +0.50% threshold and the CI-above-zero condition.

## Pre-registration

The criteria are in the docstring of `scripts/h4_consecutive_beats.py`. It was committed in `6e1faaa` and pushed at 2026-09-19T05:02:36Z. The wide price parquet did not exist until 05:06:08Z.

## Changes after the Codex review

- **Predecessor boundary fix.** The registered run looked for each event's predecessor only among events inside the price window. The first in-window beat of every name therefore lost a predecessor it actually had; for example, ACCD's 2023-10-04 event lost its 2023-06-29 report. `--predecessors full` (now the default) searches the whole earnings history. A preceding report with an unknown surprise is treated as a boundary and is not skipped.
- **Refreshed earnings and timing.** The same two changes as H1 apply here; see `h1_pead_wide_FINDINGS.md` § 2.

## Results, v2 (`h4_consecutive_beats_v2_{registered,volume}.json`)

| Timing | Consecutive | First | Difference |
|---|---|---|---|
| registered | n=7,240, +0.32% [−0.05, +0.70] | n=5,852, +0.29% [−0.09, +0.65] | +0.03pp [−0.47, +0.52] |
| look-ahead-free | n=7,398, +0.26% [−0.07, +0.57] | n=5,952, **+0.62%** [+0.26, +0.99] | −0.36pp [−0.84, +0.11] |

## Reading

Anchoring does not show up. If a second surprise is priced differently from a first at all, it is priced *more* completely, which fits the "serial beaters are known" prior.

The first-beat cohort's +0.62% under look-ahead-free timing is a post-hoc observation, not a result. Descriptively, it lines up with H1's finding that drift lives in neglected names.

## Caveats

- These are the same universe and timing caveats as H1.
- This is Stage 0: no stops, no costs.

## Reproduce

```bash
python scripts/h4_consecutive_beats.py --timing registered --json-out outputs/research/h4_consecutive_beats_v2_registered.json
python scripts/h4_consecutive_beats.py --timing volume     --json-out outputs/research/h4_consecutive_beats_v2_volume.json
```
