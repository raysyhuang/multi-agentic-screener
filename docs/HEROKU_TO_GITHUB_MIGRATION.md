# Heroku → GitHub Actions migration

Goal: stop paying for Heroku by moving the scheduled pipeline off the always-on
Heroku `worker` dyno and onto GitHub Actions cron, while keeping the database on
a **free external managed Postgres** instead of the Heroku Postgres add-on.

This is a real migration, not a config flip — today the pipeline runs entirely
on Heroku (the `worker` dyno's APScheduler in `src/worker.py`, plus Heroku
Postgres). GitHub currently runs only CI. **Do not cancel Heroku until the
parallel-run verification below passes.**

## What moves where

| Piece | Before (Heroku) | After (GitHub-only) |
|---|---|---|
| Scheduler | `worker` dyno, APScheduler | `.github/workflows/scheduled-pipelines.yml` cron |
| Morning / afternoon / evening / weekly jobs | APScheduler → `run_*` functions | `python -m src.worker --run-now / --check-now / --collect-now / --meta-now` |
| Telegram alerts | sent inside pipeline | unchanged — sent inside pipeline |
| Database | Heroku Postgres add-on | external managed Postgres (Neon / Supabase / Railway free tier) |
| Dashboard (`web` dyno) | Heroku `web` | optional — drop it, or host on a free tier pointed at the same DB |

Schedules preserved (US/Eastern): morning **06:00**, afternoon **16:30**,
evening **21:30** (Mon–Fri), weekly meta review **Sun 19:00**. The workflow uses
dual UTC cron lines + an Eastern-time guard so daylight-saving shifts don't move
the run or double-fire it.

## Step 0 — workflow file (already installed)

The workflow ships in this PR already in place at
**`.github/workflows/scheduled-pipelines.yml`** — no manual move needed. Merging
this PR to `main` is what activates the cron schedules (scheduled workflows only
fire from the default branch). Until it's merged you can still test any job from
the branch via **Actions → Run workflow** (`workflow_dispatch`).

## Step 1 — Stand up a free external Postgres

Pick one (all have a free tier that easily covers this workload):

- **Neon** (https://neon.tech) — serverless Postgres, generous free tier.
- **Supabase** (https://supabase.com) — Postgres + dashboard.
- **Railway** (https://railway.app) — Postgres service.

Create a database and copy its connection string. It will look like:

```
postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require
```

The code normalizes `postgres://` and `postgresql://` to the async
`postgresql+asyncpg://` driver automatically (see `src/db/session.py`), so paste
the URL as the provider gives it.

## Step 2 — Migrate the existing data (optional but recommended)

If you want to keep history/state from the current Heroku DB:

```bash
# Dump from Heroku
heroku pg:backups:capture --app <your-heroku-app>
heroku pg:backups:download --app <your-heroku-app>   # -> latest.dump

# Restore into the new external DB
pg_restore --no-owner --no-acl --clean --if-exists \
  -d "postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require" latest.dump
```

If you're fine starting fresh, skip this — the workflow runs `alembic upgrade
head` on every run, so the schema is created automatically on first execution.

## Step 3 — Add GitHub repository secrets

**Settings → Secrets and variables → Actions → New repository secret.**

Required (quant-only mode):

| Secret | Notes |
|---|---|
| `DATABASE_URL` | the external Postgres URL from Step 1 |
| `TELEGRAM_BOT_TOKEN` | same bot token used today |
| `TELEGRAM_CHAT_ID` | same chat id used today |
| `FMP_API_KEY` **or** `POLYGON_API_KEY` | at least one market-data provider |

Optional (add if you use them):

`FINANCIAL_DATASETS_API_KEY`, `FRED_API_KEY`, `API_SECRET_KEY`,
`KOOCORE_API_URL`, `GEMINI_API_URL`, `TOP3_7D_API_URL`, `ENGINE_API_KEY`,
and — only if you switch off quant-only — `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`.

To run a mode other than the default `quant_only`, set a repository **variable**
(not secret) `EXECUTION_MODE` to `hybrid` or `agentic_full`.

## Step 4 — Test each pipeline manually (no waiting for cron)

Merge this branch to `main` (scheduled workflows only fire from the default
branch), then in **Actions → Scheduled Pipelines (GitHub-hosted) → Run
workflow**, run each pipeline once and confirm:

1. The job goes green.
2. `alembic upgrade head` succeeds against the external DB.
3. The expected Telegram alert arrives.

Run order to try: `morning`, then `afternoon`, `evening`, `weekly`.

You can also run manually from a checkout:

```bash
export DATABASE_URL=...
export TELEGRAM_BOT_TOKEN=...  TELEGRAM_CHAT_ID=...  FMP_API_KEY=...
alembic upgrade head
python -m src.worker --run-now      # morning pipeline, runs once and exits
```

## Step 5 — Parallel run (Heroku + GitHub) for ~1–2 weeks

Leave Heroku running. Let the GitHub cron fire on its normal schedule alongside
Heroku and compare, each trading day:

- Both send the same/consistent Telegram alerts at the right Eastern times.
- The scheduled runs in the Actions tab are green (watch for the DST-guard
  "Skipping — wrong DST offset" notices; exactly one of each pair should run).
- The external DB is being written to as expected.

If Heroku and GitHub would double-send during this window and that's noisy, you
can point the two at **separate** Telegram chats (a temporary test chat id in
the GitHub secret) so you can compare without spamming your main channel.

## Step 6 — Cut over and cancel Heroku

Once a couple of weeks look clean:

1. Stop the Heroku scheduler so it can't double-run:
   `heroku ps:scale worker=0 --app <your-heroku-app>` (and `web=0` if you don't
   need the dashboard).
2. Confirm GitHub keeps alerting normally for a day or two on its own.
3. Cancel the Heroku app and the **Heroku Postgres add-on** (that add-on is a
   large part of the bill — make sure Step 2's external DB is the live one first).
4. Clean up the repo: remove `Procfile` and the `worker`/`web` process
   definitions once nothing depends on them.

## Cost after migration

- Scheduler/compute → GitHub Actions minutes (these short jobs are well within
  the free allowance for a private repo; public repos are unlimited).
- Database → external free Postgres tier.
- Dashboard → dropped, or a free web tier if you want to keep it.

## Rollback

Nothing here modifies Heroku until Step 6, and the pipelines are idempotent, so
rollback is simply: keep the Heroku `worker` running (don't scale it to 0) and
disable the GitHub workflow (rename to `.disabled` or turn it off in the Actions
tab). No data changes are required to revert.
