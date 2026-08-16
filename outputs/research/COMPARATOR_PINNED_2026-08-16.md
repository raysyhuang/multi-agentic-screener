# Comparator — PINNED (corrected). 2026-08-16

**Produced by:** Victor (Claude Code, VPS Boston). **Reviewed by:** Hawk, who found four defects in
the first version. **Arbiter is not the producer** — Hawk or Ray rules this.
**Method:** `comparator-pinning-METHOD.md`, falsification conditions fixed before the numbers were seen.

## Source — a frozen copy, not a URL

| Field | Value |
|---|---|
| Frozen artifact | `outputs/research/frozen/data-2026-08-14T205945Z.json` |
| **sha256** | `f4de7a2e7bf566b0935ca08ebea5e84cc501c4f72773bab1b7a5eb6b84d512a0` |
| Bytes | 254,904 |
| `generated_at` | 2026-08-14T20:59:45.235253+00:00 |
| Origin | `scripts/export_dashboard_data.py` output, deployed to GitHub Pages by the production pipeline |
| Access | public, unauthenticated. **No production database was queried.** |

> **Why frozen.** The Pages export is overwritten on every run with `window_days: 90`. Monday's run
> drops trades before ~2026-05-18, and a comparator pinned to the URL would silently move with no
> amendment. **A pin must reference this file and hash.** (Hawk.)

## The figures

| Stream | n | avg `pnl_pct` | win rate | mean alpha vs SPY | alpha CI | `significant` |
|---|---|---|---|---|---|---|
| `mean_reversion\|mr_manual_sleeve` | 63 | +0.1847% | 50.8% | +0.1673 | [−0.4158, +0.7623] | false |
| **`sniper\|mas_official`** | **60** | **+0.7490%** | **53.3%** | **+0.6408** | **[−0.5856, +1.8895]** | **false** |
| `mean_reversion\|mas_official` | 35 | +0.4039% | 54.3% | +0.5847 | [−0.1881, +1.3334] | false |
| `pead\|pead_neglected` | 10 | −0.1942% | 60.0% | −1.3135 | [−3.3192, +0.5539] | false |
| `pead\|pead_paper` | 10 | −0.6139% | 30.0% | −1.2616 | [−2.7474, +0.4297] | false |

**Aggregates:** 178 trade rows, all with `pnl_pct` · 65 `run_history` entries, 2026-05-18 → 2026-08-14 ·
13 open positions.

## Rulings — against conditions fixed before the numbers were seen

**`+0.74%` / 50% WR over 20 trades** → **SUPERSEDED, not wrong.** n tripled (20 → 60), estimate barely
moved (+0.7490%, 53.3% WR). Still **not admissible on its own terms**: the CI crossed zero at n=20 and
**still crosses zero at n=60**.

**`−0.97%` / ~50% WR, MAS-GH sniper** → **RETRACTED.** Not reproduced — a sign flip, not a magnitude
disagreement. No supporting artifact exists in the repo, on the VPS, or in the bundle. Independently
recomputed from the raw JSON by Hawk, matching exactly.

**`−0.14%` / ~42% WR, IBKR** → **UNLOCATABLE, not retracted.** *(Corrected — the first version said
retracted; Hawk is right that this was over-strong.)* IBKR is a separate broker. Its absence from the
**MAS** production export means I looked in the wrong place, not that the figure is disconfirmed. It
needs a source named before it can be ruled at all.

## Four defects Hawk found in the first version, and what they mean

**1. The pinned metric is not the decisional metric.** I pinned raw `pnl_pct` (+0.7490%). Tier 2
condition 2 reads **alpha vs SPY** (+0.6408% for the same stream). Condition 5 says "clears the pinned
comparator on expectancy" without saying *which* expectancy. **Both quantities are now stated above;
the doc must name one.** I recommend alpha vs SPY, to match condition 2 and avoid two metrics in one bar.

**2. The comparator establishes nothing itself.** Every live stream has `significant: false` — sniper's
alpha CI spans **2.48 percentage points** and includes zero. So condition 2 demands the *paper sleeve's*
CI exclude zero, while condition 5 asks it to beat a live book whose own CI has established nothing.
**Two evidentiary standards side by side, and the weaker one is the gate.** Either make condition 5
CI-aware, or state plainly that it compares point estimates and is the softer test.

**3. Bundle ambiguity — the sharpest one.** The mirror bundle and this production bundle use **identical
stream keys** (`sniper|mas_official`, `pead|pead_neglected`, …). `alpha_summary[<stream>]["spy"]["ci_lo"]`
resolves in **both** and returns different values — the mirror had `sniper|mas_official` at n=3 while
this has n=60. A reader taking condition 2 from the production bundle would **evaluate the live book
against itself** and satisfy conditions 2 and 5 with identical rows. This is the #88 field-path defect
one level up: the path resolves, but not uniquely. **The doc must name the bundle, not just the path.**

**4. Rolling window.** Addressed by freezing above.

## Corrections to my own first version

- **Counts were wrong: 178 closed trades and 65 `run_history` rows, not 168 and 73.** Every per-stream
  figure was correct because I computed those; the aggregates came from a model-generated summary of the
  JSON that I did not recompute. **In a record whose whole purpose is stated row counts, I mixed a
  computed figure with a summarized one.** The per-stream figures reproduce exactly under Hawk's
  independent recomputation.
- **The 13 open positions are not over cap.** They are 6 `pead_paper` + 4 `pead_neglected` — exactly
  `pead_max_concurrent = 10` — plus 3 sniper. The "cap of 3–5" I cited is `pead_max_positions`,
  per-run admission. Different mechanism. This is the same conflation Grok already corrected once.

## Limitations

1. **90-day window, not full history.** Method §2 forbids date filters precisely because window choice
   selects flattering numbers. **These are pinned as 90-day figures and must be labelled so.**
2. **Method §1 wording.** My method ranked "dashboard renderings" inadmissible. This is not a rendering
   — it is the machine-readable bundle the acceptance doc already names a source of truth. Recommended
   clarification: admissible = the exported bundle **with `generated_at` and hash recorded**;
   inadmissible = screenshots, chat messages, memory, and any figure quoted without its artifact.
3. **Not verified against the database.** Nothing here contradicts the DB; nothing here checked it either.

## Standing observation

Every correction in this document came from another agent reading the same source — Hawk on the counts,
the metric, the circularity and the cap; Grok on the PEAD mechanism earlier. None came from anyone
reasoning more carefully alone.
