# Multi-Agentic Screener — Project Rules

## Common Errors to Avoid
- Replay/backtest harnesses MUST mirror the live tracker's FULL execution config from settings — `trail_activate_pct=0.5`, `trail_distance_pct=0.3`, `slippage_pct=0.001` (10bp/side), gap-through fills. Three parameter-mismatch artifacts were caught in the 2026-07 MR reconciliation (trail defaulted 0/0, cost 5 vs 10bp, ticker alias); each manufactured a fake live-vs-engine gap. Default these from `get_settings()`, never hardcode.
- Slippage/cost is NOT a post-hoc constant in the exit engine: the slippage-adjusted entry fill is the base for trail activation, so changing cost changes exit PATHS (5→10bp shifted a cohort's engine avg by −0.081pp, not −0.10pp/row). Never "adjust for costs" after the fact — rerun the engine at the correct cost.
- Polygon uses dot-form share-class tickers (`PBR.A`, `BRK.B`); the cohort/universe normalizer uses dash form (`PBR-A`). Alias dash→dot on Polygon fetches (see `polygon_symbol_candidates` in `scripts/cohort_replay.py`) or rows silently drop.
- `outputs/` is gitignored — research `*_FINDINGS.md` docs and summary JSONs must be force-added (`git add -f outputs/research/...`) or they won't persist to the repo. Since the staging guard landed this also needs `ALLOW_LOCAL_FILES=1` on the commit, which is the point: force-adding into an ignored tree should be a deliberate, recorded act.

## Staging guard (install once per clone)
```bash
git config core.hooksPath scripts/hooks
```
`scripts/hooks/pre-commit` refuses newly staged **or renamed-into** local-only files — `backups/`, `deploy/`, `skills-lock.json`, `*.env`, dumps, keys, and `"foo 2.py"` sync duplicates. It reads the **staged index**, so a messy working tree is fine; only what you are about to commit is checked.

Two `git add -A` sweeps reached PRs under review in one session; the second put trade-level exit prices and P&L into public history, where the commit stays retrievable by SHA even after the branch is deleted. Stage explicit paths. When a blocked file genuinely belongs, say so on the record:
```bash
ALLOW_LOCAL_FILES=1 git commit -m "... (ALLOW_LOCAL_FILES: why this path belongs)"
```
Both halves are required: the environment variable permits the commit, and `scripts/hooks/commit-msg` demands the written reason — the variable alone leaves nothing in git history, so a reviewer would see the file and not the justification.

## Authoritative checkout (decided 2026-07-27 by Ray)
- **The Claude Code checkout (`~/Documents/Python Project/Multi-Agentic Screener`) plus `origin/main` is the single source of truth for this repo.**
- Other checkouts of this repo on other hosts (e.g. the VPS `/srv/workspaces/multi-agentic-screener` used by the Neo agent) are **research sandboxes only**. They must NOT push to `origin/main`. Their local script edits are not canonical and can be overwritten.
- To land work from a sandbox: open a PR against `origin/main` (or hand over the artifact/finding and re-implement here) so it gets reviewed + CI-gated like anything else. Never merge by pushing from a sandbox.
- Why: two agents writing to two working copies of a same-named repo is exactly the setup behind the 2026-04-11 unattended-push incident. Single authority removes the ambiguity.
- Sanity check when cross-agent work is in flight: `git fetch origin && git log --oneline -3 origin/main` and confirm local `HEAD == origin/main`.

## Project Conventions
- Keep `mas_official` and `mr_manual_sleeve` performance stats separate — blending them produced the false "MR is a coin flip" conclusion.
- New strategy claims go through the unified exit engine (`src/backtest/exit_engine.py`) with realistic gap-through fills + concurrency-capped equity before any label/promotion. Persistent memory (MEMORY.md) holds the full research log.
- Backtests run WITHOUT gap-through fills reproduce an ~80-90% sniper win rate that is a pure artifact (Run A). The data vendor is not the variable: swapping yfinance→Polygon at fixed config moves WR ~1pp, while the `gap_through` flag moves it ~38pp (91.1%→53.0%). See `outputs/research/HANDOFF_gap_through_diagnosis.md`.
- **Provenance:** any artifact claiming a data source must stamp `get_last_ohlcv_provenance()` (provider, requested/returned, missing, per-ticker failures). `fetch_ohlcv` falls back Polygon→yfinance; that fallback and every dropped ticker are now RECORDED, never silent, and a `.provenance.json` manifest is written beside each Polygon cache. Use `fetch_ohlcv(..., strict=True)` when a run must be Polygon-only — it raises instead of silently producing mixed-provenance data.
