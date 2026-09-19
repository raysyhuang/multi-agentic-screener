# Paper-sleeve measurement window — start-condition evidence (gathered 2026-09-19)

Companion to `docs/paper_sleeve_acceptance_criteria.md` § "The start condition". That section names three conditions and says the start is *determined by the condition and then recorded, never chosen and then justified*. This file is the record of what was checked, where, and what it showed, so the recorded start date can be audited by someone who was not there.

## How the evidence was gathered

Read-only, from Ray's authoritative checkout, over key-authenticated SSH to the Boston mirror host (`openclaw-1`). No file on the host was written. The three queries, verbatim:

1. Every afternoon manifest: `python3` over `glob("/home/agent/.hermes/cron/output/mas_vps_paper_mirror/*/afternoon/run-meta.json")`, printing `run_date`, `dashboard_sha256`, `source_sha`, `launcher_sha256`, `launcher_git_head`, `launcher_git_tracked`, `launcher_git_clean`, `launcher_path`, `completed_at_utc`.
2. Scheduler executions: `select started_at, finished_at, status, error from executions where job_id='94022cc9cad0' and started_at>='2026-08-28'` against `/home/agent/.hermes/cron/executions.db` (the job is `MAS VPS PAPER mark-to-market — post-close artifact-only`, `35 21 * * 1-5`, enabled).
3. Locally: `git merge-base --is-ancestor <source_sha> origin/main` for every distinct `source_sha`, and `git show <sha>:scripts/mas_vps_paper_mirror.py | shasum -a 256` for the launcher content hash at those revisions.

## Condition 1 — launcher on `origin/main`, deployed copy's hash matches

`scripts/mas_vps_paper_mirror.py` entered `origin/main` in `f27f2e6` (2026-08-20) and was last changed by `0f61b92` (#111, 2026-08-31). Its content hash at `0f61b92`, `435bbd5` and `3819baf` is identical: **`d802de597cce…`**.

Every afternoon manifest from **2026-08-31** onward reports `launcher_sha256 = d802de597cce…`, `launcher_git_tracked = true`, `launcher_git_clean = true`, `launcher_path = /srv/workspaces/multi-agentic-screener-mirror-clean/scripts/mas_vps_paper_mirror.py`, remote `github.com/raysyhuang/multi-agentic-screener.git`. The deployed copy is therefore byte-identical to the file on `origin/main`, running from a clean tracked checkout.

Manifests from **2026-08-20 through 2026-08-28** carry **no launcher provenance fields at all** (`launcher_sha256`, `launcher_git_head` etc. absent). The provenance stamp was added to the launcher in `cce04bf` (2026-08-21), so the copy deployed on those days predates it and its identity cannot be attested. Those eight days are **not** counted toward condition 2, although each did produce a bundle.

**Condition 1: met from 2026-08-31.**

## Condition 2 — five consecutive valid measurement days by that launcher

"Valid measurement day" = a scheduled afternoon run producing a bundle with non-null `dashboard_sha256` and exit code zero. Exit code: the launcher raises on any non-zero subprocess (`mas_vps_paper_mirror.py` ~line 219) and writes `run-meta.json` only at the end (~line 396), so a manifest's existence implies every step exited 0; the scheduler independently recorded every execution below as `completed` with an empty `error`.

| run date (ET) | `dashboard_sha256` | `source_sha` (on `origin/main`?) | launcher sha | scheduler |
|---|---|---|---|---|
| 2026-08-31 Mon | `6819811855ba…` | `0f61b92` yes | `d802de5…` | completed |
| 2026-09-01 Tue | `8098f9c088b8…` | `463cbe5` yes | `d802de5…` | completed |
| 2026-09-02 Wed | `48f3a682f3ab…` | `435bbd5` yes | `d802de5…` | completed |
| 2026-09-03 Thu | `48327697941b…` | `435bbd5` yes | `d802de5…` | completed |
| 2026-09-04 Fri | `be9d2e58a8f1…` | `435bbd5` yes | `d802de5…` | completed |

Five scheduled runs, Mon–Fri, no invalid day between them. The run continued valid every trading day through 2026-09-18 (`3819baf`, the day's `origin/main`), 16 of 16 scheduler executions `completed`.

**Condition 2: met on 2026-09-04.**

## Condition 3 — this document records the start date, in a commit dated on or before it

Not met on 2026-09-04: nobody recorded anything. The criteria doc's rule is that the window opens on the first `entry_date` **on or after the calendar date on which all three conditions are simultaneously true**. Condition 3 becomes true only when the recording commit lands. That commit is the one carrying this file, dated **2026-09-19**.

**Start date: the first `entry_date` on or after 2026-09-19.** 2026-09-19 is a Saturday; the first qualifying entries are those filled from the Monday 2026-09-21 morning run onward.

Consequence, stated so it cannot later be read as an oversight: every paper trade entered **2026-08-31 through 2026-09-18** — including `RBRK` (`pead_neglected`, entered 2026-08-31, +16% unrealised on 2026-09-17) and `IOT` (2026-09-08) — is **OUT**: visible in `trades[]`, describable, never counted toward `n`, any Tier test, or S1. That is the doc's own "deliberately unfavourable" clause doing its job. Backdating the start to 2026-09-04 would have added those trades to the sample after their results were known, which is the exact move the condition exists to prevent.

## What this does not establish

- That the scheduler was *pointed at* the versioned launcher on every day between 08-31 and today is inferred from `launcher_path` + `launcher_git_tracked` in each manifest, which the launcher stamps about itself. The launcher's own manifest comment says the content hash "establishes code-content equivalence only; repointing the scheduler is a separate step". The scheduler record (`jobs.json`, job `94022cc9cad0`) was read and points at that path; a host-side change after 2026-09-19 would need re-checking.
- Nothing here is a statement about any sleeve's performance. `n` for every stream is 0 as of the start date.
