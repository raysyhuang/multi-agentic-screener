# Review — Boston VPS MAS paper mirror (Hermes / "Neo")

**Date:** 2026-08-10
**Reviewer:** Claude Code, from the authoritative checkout (`~/Documents/Python Project/Multi-Agentic Screener`, per CLAUDE.md §"Authoritative checkout")
**Subject:** sanitized operational brief for `openclaw-1` / `/srv/workspaces/multi-agentic-screener-mirror-clean`
**Method:** every claim checked against `origin/main` source at `6b0c205f`. No secrets read, no live system touched, no changes made.

## Verdict: **MODIFY**

The architecture is correct and correctly scoped. Three stated guarantees are decorative, and the delivery schedule inverts every US winter. Two items (F4, F2) can cause real harm and are both fixable launcher-side with no upstream change.

---

## 1. Claims verified TRUE

| Claim | Evidence |
|---|---|
| VPS SHA `6b0c205f` current | `git log origin/main` → `6b0c205 Fix the candidate-audit identity, harden the SW cache, test behaviour not source (#61)` is the tip. Not drifted. |
| `825 passed, 4 deselected` | Reproduced: `pytest --collect-only` on this content gives `825/829 tests collected (4 deselected)`. (A local run collecting 832 was traced to an untracked junk file, `tests/test_earnings_blackout 2.py` — local only.) |
| "No broker/order submission path is enabled" | Structurally true: `find src/broker -name "*.py"` returns **nothing** on `main`. IBKR code lives only on the `ibkr-automation` branch. This is a stronger guarantee than any env var. |
| Workflow name `Scheduled Pipelines (GitHub-hosted)` | `.github/workflows/scheduled-pipelines.yml:1`. |
| `execution_mode=quant_only`, `trading_mode=PAPER`, `pead_enabled` are real settings | `src/config.py:134`, `:137`, `:208`. |
| Mirror "runs current upstream MAS worker" | `python -m src.worker --run-now` is the supported entrypoint and rejects unknown flags loudly (`src/worker.py:53-60`). Correct choice — no fork needed. |

---

## 2. Findings

### F1 — `LOCAL_MIRROR_NO_TRADE` / `LOCAL_MIRROR_NO_TELEGRAM` enforce nothing · **HIGH**

Zero hits repo-wide (`grep -rn` across `*.py`, `*.yml`, `*.md`). Worse, `src/config.py:286` sets `"extra": "ignore"`, so pydantic accepts and silently discards them — they will never even error.

They document intent; no code reads them. The actual guarantees come from elsewhere (no broker module on `main`; empty Telegram creds). The risk is the false interlock: anyone reading the launcher believes there is a kill switch backing those names.

**Fix:** either delete them, or convert them into launcher-side assertions that *do* something (see F2 fix).

---

### F2 — "`TELEGRAM_BOT_TOKEN` unset" does NOT guarantee no Telegram · **HIGH**

`src/config.py:283-284`:

```python
model_config = {
    "env_file": str(ENV_PATH),      # ENV_PATH = PROJECT_ROOT / ".env"  (config.py:22-23)
    "env_file_encoding": "utf-8",
    "extra": "ignore",
}
```

`PROJECT_ROOT` resolves from the source file, i.e. `/srv/workspaces/multi-agentic-screener-mirror-clean/.env`.

pydantic-settings v2 precedence is **init args > environment variables > dotenv file > defaults**. *Unsetting* an env var therefore hands the decision to the dotenv file, not to the empty default. If that checkout's `.env` ever carries `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`, the mirror will post into the real `[MAS-GH]` stream.

`DATABASE_URL` is safe only incidentally — the launcher sets it explicitly, and an explicit env var does beat the dotenv.

**Fix (minimal, launcher-side):**

1. Export **explicit empty strings**, not unset: `TELEGRAM_BOT_TOKEN=""`, `TELEGRAM_CHAT_ID=""`. An empty env var beats the dotenv value.
2. Assert on the **resolved settings object** after import, and fail closed:

```python
s = get_settings()
assert s.telegram_bot_token == "" and s.telegram_chat_id == ""
assert "mas_mirror_" in s.database_url and s.database_url.startswith("postgresql")
assert s.trading_mode == "PAPER"
assert importlib.util.find_spec("src.broker.ibkr") is None   # survives an IBKR merge to main
```

Asserting on resolved behaviour rather than on the environment is the only check that survives both dotenv precedence and future upstream refactors. It is also what turns F1's decorative names into real guards.

`send_alert` soft-no-ops and returns `False` on empty creds (`src/output/telegram.py:107-109`), so the empty-string path is safe and non-fatal.

---

### F3 — Schema bootstrap diverges from prod · **MEDIUM** (answers review Q4)

Production order (`.github/workflows/scheduled-pipelines.yml:137-141`):

```yaml
- name: Apply database migrations
  run: alembic upgrade head
- name: Run <pipeline> pipeline
  run: python -m src.worker --run-now
```

Mirror order: worker only. `src/worker.py:31` → `init_db()` → `src/db/session.py:106` → `Base.metadata.create_all`.

On a fresh `mas_mirror_main`, `create_all` builds tables straight from ORM metadata with **no `alembic_version` row**. Two consequences:

- **(a) Silent behavioural drift.** Anything a migration does that the ORM does not express — data backfills, indexes, constraints, server defaults — never lands. The mirror can produce different results from prod *on identical code*, which defeats the purpose of a mirror.
- **(b) A one-way door.** The first `alembic upgrade head` against that DB will fail ("already exists"). The tempting fix, `alembic stamp head`, makes (a) permanent and invisible.

**Fix:** run `alembic upgrade head` against `mas_mirror_main` before invoking the worker — mirror prod's bootstrap exactly. If the DB is already `create_all`-built, drop and rebuild through alembic; it is disposable paper state. Do **not** stamp.

---

### F4 — The 18:55 Asia/Shanghai slot inverts every US winter · **HIGH** (answers review Q2)

China observes no DST, so `18:55 Asia/Shanghai` is pinned at **10:55 UTC** year-round. Upstream is deliberately DST-aware — two cron lines per job plus a zone-matching guard (`scheduled-pipelines.yml:12-33`, and the `LINE_ZONE`/`CUR_ZONE` resolve step):

```yaml
# Morning pipeline — 06:00 ET Mon-Fri   (EDT 10:00 UTC / EST 11:00 UTC)
- cron: "0 10 * * 1-5"
- cron: "0 11 * * 1-5"
```

| Period | MAS-GH morning start | VPS brief | Headroom |
|---|---|---|---|
| EDT (Mar–Nov) | 10:00 UTC | 10:55 UTC | 55 min |
| **EST (Nov–Mar)** | **11:00 UTC** | **10:55 UTC** | **−5 min** |

For roughly four months a year the brief fires *before* the run it is meant to report.

And it does not fail closed. A health check keyed on "latest run of the workflow, `conclusion == success`" will find **yesterday's** successful run, see the SHA aligned, declare the pipeline healthy, and then execute a mirror pipeline pre-market on stale data. That is a silent wrong answer, not a visible outage — the worst failure mode for a diagnostic.

The EDT 55-minute headroom is also anchored wrong. That workflow's own header records why: *"GitHub routinely fires crons 1-2h late, and the previous exact-hour guard silently skipped every late run (all four 2026-07-21 slots were skipped this way)."* A fixed offset from a **nominal** cron time repeats the mistake the zone-matching rewrite fixed.

**Fix:**

1. Schedule the launcher in `America/New_York` (e.g. 07:15 ET weekdays) so it tracks the same DST the upstream cron tracks.
2. Require the successful run's date to equal **today in ET**; otherwise emit `pipeline pending` and exit 0.
3. Better than a clock offset: poll for a *completed* run created today, with a bounded retry window, and anchor on completion rather than on elapsed time.

---

### F5 — `run=success` does not mean the pipeline ran · **MEDIUM**

The `run-pipeline` job resolves `run=true/false` and, when false, **succeeds having executed nothing** — `if: steps.resolve.outputs.run == 'true'` guards both the alembic step and the worker step. By design, every weekday one of the two cron lines for each slot is skipped-but-green (that is the DST zone-matching mechanism).

So `conclusion == "success"` green-lights a no-op run roughly as often as a real one.

**Fix:** key the health check on the job output `ran` (`scheduled-pipelines.yml:56-59`, `ran: ${{ steps.resolve.outputs.run }}`), or on the `publish-dashboard` job having executed — it is already gated on `needs.run-pipeline.outputs.ran == 'true'`. A freshly deployed `dashboard/data.json` is an equally good positive signal.

---

### F6 — "main advanced → pipeline pending → skip" is too broad · **MEDIUM**

PR #61 itself touched `README.md`, `.gitignore`, `dashboard/index.html`, `dashboard/sw.js`. Under the stated contract, a docs- or frontend-only merge suppresses the entire daily brief for a day — an availability loss with no corresponding risk.

**Fix:** treat main as "ahead" only when the diff touches pipeline-relevant paths (`src/`, `alembic/`, `pyproject.toml`, `.github/workflows/scheduled-pipelines.yml`). Otherwise fast-forward, run, and note the SHA difference in the message body.

---

### F7 — Data-hierarchy section does not match the code · **LOW/MEDIUM** (answers review Q6)

1. **`massive`, `tushare`, `adanos`, `finnhub` appear nowhere in `src/` or `scripts/`.** Those keys exist in the VPS environment, but MAS reads none of them. Listing them under "data-source hierarchy in current MAS code" implies coverage that does not exist. MAS reads `polygon_api_key`, `fmp_api_key`, `financial_datasets_api_key`, `fred_api_key` (`src/config.py:28-31`).
2. **The live OHLCV chain is Polygon → FMP → yfinance**, not "Polygon primary, yfinance fallback" (`src/data/aggregator.py:72`, `:89-127`). FMP is a *price* fallback as well as a fundamentals source. An FMP-served bar is different-provenance data and should be labelled as such.
3. **The provenance machinery covers the wrong path.** `get_last_ohlcv_provenance()` and `fetch_ohlcv(..., strict=True)` live at `src/research/signal_backtest.py:69,79` — the **research** path. The **live** aggregator has no equivalent accessor; it only tags cache rows with `source=` (`aggregator.py:96,113,124`). So the CLAUDE.md provenance rule does not currently cover what the mirror actually executes. See O1 below.

---

## 3. Answers to the review questions

**Q1 — Is GitHub → VPS-only plus `no_push` sufficient and correctly scoped?**
Correct in shape; one layer thin. `no_push` on the remote is bypassed by `git push --repo <url>` and by `gh`. Add:
- a `pre-push` hook in the mirror clone that hard-fails unconditionally;
- confirmation that no write-scoped GitHub token is present in that shell's environment;
- check the mirror out **detached at `origin/main`** rather than on a tracking branch, so "fast-forward only" is structural and local commits cannot accumulate.

FYI, the application does shell out to git — `src/governance/artifacts.py:120-131` runs `git rev-parse --short HEAD` for run reproducibility. Read-only and harmless, but worth knowing the app touches git at all.

**Q2 — Is the one-message scheduling contract sound?**
No — see **F4** (winter inversion, silent stale run), **F5** (green no-op runs read as success), **F6** (docs commits suppress the brief).

**Q3 — Are the environment restrictions adequate to prevent message/broker/live side effects?**
Broker axis: **yes**, and for a better reason than stated — there is no broker code on `main` at all, which is a structural guarantee rather than a configuration one. Messaging axis: **no** — see **F2**. The mitigation is the resolved-settings assertion block, which also gives F1's names real teeth and fails closed if the IBKR branch is ever merged.

**Q4 — Is the DB bootstrap/migration approach appropriate?**
No — see **F3**. `create_all` instead of `alembic upgrade head` is the single most likely source of quiet mirror-vs-prod divergence.

**Q5 — Low-risk observability improvements.**
Ordered by value:

- **O1 · Provider tally per run.** The aggregator already knows the answer — it stamps `source="polygon"|"fmp"|"yfinance"` on every cache put (`aggregator.py:96,113,124`). Add a counter and emit `{polygon: N, fmp: N, yfinance: N, failed: [tickers]}` into `run-meta.json`. This is the live-path analogue of `get_last_ohlcv_provenance()` and closes F7.3. Small additive upstream PR; production benefits identically.
- **O2 · Circuit-breaker state at run end** (`src/data/circuit_breaker.py`). "yfinance served 400 bars because Polygon's breaker was open" is currently only reconstructable from scattered warning lines.
- **O3 · No-pick attribution.** The exporter already emits ranked candidates with a `picked` flag (`scripts/export_dashboard_data.py:192-220`, added by PR #61) alongside `candidates_scored`. Have `run-meta.json` record universe size → candidates scored → per-gate rejection counts → the 8 validation-card check results. Then `picks=0` names a gate instead of being a mystery. Precedent: the PR #39 `regime split` log line is exactly what cracked the MR block, on data rather than inference.
- **O4 · Stamp `alembic current` plus an ORM-metadata hash** into `run-meta.json` — makes F3 drift visible instead of silent.
- **O5 · Diff mirror picks against the official dashboard `data.json`** for the same date and record the delta. Converts "names can differ" from a standing caveat into a measured number, which is the only way the mirror's diagnostic value can be assessed over time.

**Q6 — Contract vs actual upstream interface.**
See **F1**, **F5**, **F7**. The worker entrypoint claim is correct and well chosen. Agreed on not restoring deleted legacy mirror scripts — nothing in this review requires it. Every recommendation above is either VPS-launcher-side or a small additive upstream PR (O1, O3), landed through the normal PR → CI → merge path.

---

## 4. Recommended order of work

1. **F4** — reschedule in `America/New_York`, require run date == today ET. Launcher-side, no upstream change. Prevents ~4 months/yr of silent stale-data briefs.
2. **F2** — explicit empty Telegram vars plus the resolved-settings assertion block. Launcher-side.
3. **F5** — health check on `outputs.ran` / `publish-dashboard`, not `conclusion`.
4. **F3** — `alembic upgrade head` on `mas_mirror_main`; rebuild if already `create_all`-built.
5. **F6** — path-scoped "ahead" detection.
6. **O1/O3** — upstream PRs for live-path provider tally and no-pick attribution.
7. **F1/F7** — correct the brief's wording so it describes what the code does.

---

## 5. Note on test reproduction

The `825 passed / 4 deselected` figure is confirmed at the collection level on this content. A local full run under anaconda Python 3.11 showed 15 order-dependent failures that all pass in isolation; the VPS Python 3.12 venv reports clean and `main` CI is green, so this is recorded as a local-environment artifact, **not** a finding against the mirror or upstream.
