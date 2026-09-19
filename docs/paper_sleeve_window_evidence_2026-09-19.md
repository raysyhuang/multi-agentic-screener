# Paper-sleeve measurement window — start-condition evidence (gathered 2026-09-19)

Companion to `docs/paper_sleeve_acceptance_criteria.md` § "The start condition". That section names three conditions and says the start is *determined by the condition and then recorded, never chosen and then justified*. This file is the record of what was checked, where, and what it showed, so the recorded start date can be audited by someone who was not there.

## How the evidence was gathered

Read-only, from Ray's authoritative checkout, over key-authenticated SSH to the Boston mirror host (`openclaw-1`). No file on the host was written. Query descriptions (the Python was typed inline over SSH and is paraphrased here; the SQL and git commands are as run):

1. Every afternoon manifest: `python3` over `glob("/home/agent/.hermes/cron/output/mas_vps_paper_mirror/*/afternoon/run-meta.json")`, printing `run_date`, `dashboard_sha256`, `source_sha`, `launcher_sha256`, `launcher_git_head`, `launcher_git_tracked`, `launcher_git_clean`, `launcher_path`, `completed_at_utc`.
2. Scheduler executions: `select started_at, finished_at, status, error from executions where job_id='94022cc9cad0' and started_at>='2026-08-28'` against `/home/agent/.hermes/cron/executions.db` (the job is `MAS VPS PAPER mark-to-market — post-close artifact-only`, `35 21 * * 1-5`, enabled).
3. Scheduler definition: `/home/agent/.hermes/cron/jobs.json`, entry `94022cc9cad0` (`enabled: true`, `35 21 * * 1-5`); an older disabled job `62fee433f11c` was also present and ignored.
4. Locally: `git merge-base --is-ancestor <source_sha> origin/main` for each of `368481554 cb8ec41 cce04bf 0f61b92 463cbe5 435bbd5 3819baf` (all exit 0), and `git show <sha>:scripts/mas_vps_paper_mirror.py | shasum -a 256` at `1b0d4fa`, `0f61b92`, `435bbd5`, `3819baf` (all `d802de597cce…`).

## Condition 1 — launcher on `origin/main`, deployed copy's hash matches

`scripts/mas_vps_paper_mirror.py` entered `origin/main` in `f27f2e6` (2026-08-20; an earlier copy existed on the unmerged PR #89 branch, `d7e4b8c`) and was last changed on `origin/main` by `1b0d4fa` (#110, 2026-08-30). Its content hash at `1b0d4fa`, `0f61b92`, `435bbd5` and `3819baf` is identical: **`d802de597cce…`**.

Every afternoon manifest from **2026-08-31** onward reports `launcher_sha256 = d802de597cce…`, `launcher_git_tracked = true`, `launcher_git_clean = true`, `launcher_path = /srv/workspaces/multi-agentic-screener-mirror-clean/scripts/mas_vps_paper_mirror.py`, remote `github.com/raysyhuang/multi-agentic-screener.git`. The deployed launcher file is therefore byte-identical to the file on `origin/main` and is a tracked file matching its HEAD blob (`launcher_git_clean` checks the launcher file only, not the whole checkout).

Manifests for the seven afternoon runs from **2026-08-20 through 2026-08-28** (08-20, 08-21, 08-24, 08-25, 08-26, 08-27, 08-28) carry **no launcher provenance fields at all** — neither `launcher_sha256` (written by the launcher since `f27f2e6`) nor the checkout fields `launcher_path/git_head/tracked/clean/remote` (added in `cce04bf`, 2026-08-21). A missing field establishes missing attestation, not which code ran; the versioned launcher is first *attested* on 2026-08-31. Those seven days are **not** counted toward condition 2, although each did produce a bundle.

**Condition 1: met from 2026-08-31.**

## Condition 2 — five consecutive valid measurement days by that launcher

"Valid measurement day" = a scheduled afternoon run producing a bundle with non-null `dashboard_sha256` and exit code zero. Exit code is taken from the **scheduler's execution record** (`status = completed`, empty `error`), not inferred from the manifest: the launcher does raise on a non-zero pipeline step (`mas_vps_paper_mirror.py` ~line 219), but it writes `run-meta.json` before the final dashboard parse/summary (~lines 396–407, a later exception returns 1), its provenance subprocesses return `None` on failure rather than raising (~lines 284–295), and a repeat run reuses the day's directory without clearing an earlier manifest (~lines 350–352). Manifest presence alone is therefore not proof of exit 0.

| run date (ET) | `dashboard_sha256` | `source_sha` (on `origin/main`?) | launcher sha | scheduler |
|---|---|---|---|---|
| 2026-08-31 Mon | `6819811855ba…` | `0f61b92` yes | `d802de5…` | completed |
| 2026-09-01 Tue | `8098f9c088b8…` | `463cbe5` yes | `d802de5…` | completed |
| 2026-09-02 Wed | `48f3a682f3ab…` | `435bbd5` yes | `d802de5…` | completed |
| 2026-09-03 Thu | `48327697941b…` | `435bbd5` yes | `d802de5…` | completed |
| 2026-09-04 Fri | `be9d2e58a8f1…` | `435bbd5` yes | `d802de5…` | completed |

Five scheduled runs, Mon–Fri, no invalid day between them. The lane continued valid every trading day through 2026-09-18 (`3819baf`, the day's `origin/main`): the scheduler query with `started_at >= 2026-08-28` returned 16 executions (08-28 plus the 15 weekdays 08-31 → 09-18), all `completed`, none with an error.

**Condition 2: met on 2026-09-04.**

## Condition 3 — this document records the start date, in a commit dated on or before it

Not met on 2026-09-04: nobody recorded anything. The criteria doc's rule is that the window opens on the first `entry_date` **on or after the calendar date on which all three conditions are simultaneously true**. Condition 3 becomes true only when the recording commit lands. That commit is the one carrying this file, dated **2026-09-19**.

**Start date: the first `entry_date` on or after 2026-09-19.** 2026-09-19 is a Saturday; the next scheduled morning run is Monday 2026-09-21, so that is the first *opportunity* for a qualifying entry — whether one occurs depends on the signals.

Consequence, stated so it cannot later be read as an oversight: every paper trade entered **before 2026-09-19** is **OUT** — visible in `trades[]`, describable, never counted toward `n`, any Tier test, or S1. `RBRK` (`pead_neglected`, entered 2026-08-31) is out under *either* candidate date, so it is not what distinguishes them. `IOT` (`pead_neglected`, entered 2026-09-08) is: a backdated 2026-09-04 start would have made it eligible for counting after its path was already partly known. That is why 09-04 was not recorded — not because a later start is "safer" in general (excluding trades can also delay an S1 stop, and rules 5–6 reject a uniformly-conservative reading), but because the rule as written yields 09-19 and 09-04 would be a discretionary choice made with results in hand.

## What this does not establish

- **How the scheduler reaches the versioned launcher** (traced 2026-09-19, read-only): job `94022cc9cad0` runs `script = mas_vps_paper_mirror_afternoon.sh` (`/home/agent/.hermes/scripts/`, `no_agent: true`, `last_status: ok`). That wrapper runs `python3 $HOME/.hermes/scripts/mas_vps_paper_mirror.py --phase afternoon` and **exits non-zero on failure** so the scheduler records an error rather than hiding it. That file is a **629-byte host-only shim** (sha256 `5bc5e3177505…`, mtime 2026-08-30 13:52), not the repo launcher: it sets `MAS_MIRROR_REPO=/srv/workspaces/multi-agentic-screener-mirror-clean`, `MAS_MIRROR_OUT_ROOT`, `MAS_MIRROR_ENV_FILES`, then `runpy.run_path("<repo>/scripts/mas_vps_paper_mirror.py", run_name="__main__")`. So the artifacts are produced by the versioned launcher (`sha256sum` of the repo file on the host = `d802de597cce…`, matching `origin/main`), and `launcher_path` in the manifests names the repo file because `launcher_provenance()` resolves `__file__` inside it. The shim itself is host-only and unversioned; its full content is three `MAS_MIRROR_*` environment assignments and one `runpy.run_path` delegation — configuration and delegation only; no pipeline or trading logic. Condition 1 is about the launcher that produces the artifacts, which is on `origin/main`; the shim is recorded here so its existence is not a surprise later, and versioning a copy of it is a reasonable follow-up. A host-side change to the shim, wrapper or job after 2026-09-19 would need re-checking.
- No performance conclusion is drawn here. `n` for every stream is 0 as of the start date. The author attests that no S1/S2 determination was pending on any stream at the time of this amendment (S1 needs `n ≥ 30`; S2's 20% drawdown threshold is still listed as *proposed* in the criteria doc); that is an attestation, not something these manifests prove.
