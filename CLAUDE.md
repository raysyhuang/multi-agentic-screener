# Multi-Agentic Screener — Project Rules

## Common Errors to Avoid
- Replay/backtest harnesses MUST mirror the live tracker's FULL execution config from settings — `trail_activate_pct=0.5`, `trail_distance_pct=0.3`, `slippage_pct=0.001` (10bp/side), gap-through fills. Three parameter-mismatch artifacts were caught in the 2026-07 MR reconciliation (trail defaulted 0/0, cost 5 vs 10bp, ticker alias); each manufactured a fake live-vs-engine gap. Default these from `get_settings()`, never hardcode.
- Polygon uses dot-form share-class tickers (`PBR.A`, `BRK.B`); the cohort/universe normalizer uses dash form (`PBR-A`). Alias dash→dot on Polygon fetches (see `polygon_symbol_candidates` in `scripts/cohort_replay.py`) or rows silently drop.
- `outputs/` is gitignored — research `*_FINDINGS.md` docs and summary JSONs must be force-added (`git add -f outputs/research/...`) or they won't persist to the repo.

## Project Conventions
- Keep `mas_official` and `mr_manual_sleeve` performance stats separate — blending them produced the false "MR is a coin flip" conclusion.
- New strategy claims go through the unified exit engine (`src/backtest/exit_engine.py`) with realistic gap-through fills + concurrency-capped equity before any label/promotion. Persistent memory (MEMORY.md) holds the full research log.
