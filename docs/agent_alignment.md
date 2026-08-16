# Agent Alignment — read before working this repo

Several AI agents work on MAS from different surfaces. This document is what they align to. It is deliberately **repo-side only**: host, scheduler, and deployment internals are not here — ask Ray for those.

## 0. Roster — who exists, and what each can actually do

§6 ("no agent executes unless it is named") requires a roster, and this document previously had none — it referred to "several agents" without naming them, which left the rule depending on knowledge that lived only in a chat channel. Names and surfaces are not host secrets; they are the half §6 needs.

| Name | Surface | GitHub write | Host shell |
|---|---|---|---|
| **Claude Code (Ray's Mac)** | Claude Code, canonical checkout | **yes** — branches, PRs, merges | no |
| **Victor** | Claude Code, VPS Boston | **yes** — repo + workflow scopes (granted 2026-08-16; was severed before) | **yes** |
| **Hawk** | OpenRouter agent | `gh` available; reviews — exact scopes unconfirmed | no |
| **Neo** | Codex agent | via PR | yes |
| **Grok bot** | Cursor agent | opens PRs via cloud agent | no |

**Assignment consequences**, learned by getting them wrong:

- **Host-side tasks are not automatically Ray's.** Several items were routed to "Ray or Neo only" on the assumption nobody else could reach the host. Victor could, and the task sat blocked on a false premise. If a task is host-side, Victor is assignable.
- **Capability cells go stale and were wrong within hours of this table landing.** Victor's said "no GitHub write"; scopes were granted the same morning. Hawk's overstated the constraint too. **Confirm capability before routing on it** — an agent blocked by a stale table is the same waste as one assigned work it cannot do.
- **§1 still binds regardless of scope.** Push access is not authority: no agent pushes to `origin/main`, ever. Work lands by branch + PR + green CI, and the guard and the grant are not in tension.
- **The correct split for host-side work** is: name the executor, name a separate reviewer, and route the push to whoever holds credentials — which may now be the executor. Name all three roles even when two collapse onto one agent.

Keep this table in your own persistent memory as well. An agent that re-derives who-did-what from conversation will misattribute across a session boundary — that has already happened twice.

## Sync point

Do not trust a SHA written in a document; documents go stale on every merge. Get it from git:

```bash
git fetch origin && git log --oneline -3 origin/main
```

`origin/main` is the sync point. If your checkout is behind it, rebase before you start.

## 1. Single authority

**`origin/main` is the source of truth — not any checkout, including Ray's.** Every working copy on every machine is a peer that fetches from it and lands work by PR.

> **Why authority moved off the Mac (2026-08-16).** This previously read "the Claude Code checkout on Ray's Mac, plus `origin/main`." That definition requires a specific laptop to be reachable in order to mean anything, and the Mac travels and gets switched off while the VPS runs 24/7. A rule whose referent is in a bag at the airport is not a rule.
>
> It also fails the monitoring case: from the Claude iPhone app, `origin/main` is visible and neither working tree is. A definition of truth that cannot be observed from where it gets checked is the wrong definition.
>
> **Nothing is loosened.** Every guard below already pointed at `origin/main`. The Mac clause was doing no work that `origin/main` was not already doing — it was only creating a second authority for whenever the two disagreed, which is the situation it was least able to adjudicate.

- Never push to `origin/main` from any checkout. **This now includes Ray's.** No machine has a private path to `main`.
- To land work: open a PR against `origin/main`, get CI green, merge there.
- Local edits are non-canonical **on every machine** and can be overwritten without notice. If you care about it, it belongs in a PR or it does not exist.
- **A checkout's state is a claim about itself, never about the repo.** Verify with `git fetch origin && git log --oneline -3 origin/main` — never by asking another machine what it has.

Why: two agents writing to two working copies of a same-named repo is the setup behind the 2026-04-11 unattended-push incident.

## 2. `main` has no branch protection

The branch-protection API returns 404 for `main`. **Nothing mechanically stops a bad push**, so the discipline is currently the only guard. Never commit to `main`, even locally. Branch from `origin/main` → PR → CI green → merge.

This is a known gap, not an endorsement. See §6.

## 3. Install the staging guard, once per clone

```bash
git config core.hooksPath scripts/hooks
```

`scripts/hooks/pre-commit` reads the **staged index** and refuses newly staged or renamed-into local-only files: `backups/`, `deploy/`, `*.env`, dumps, keys, and `"foo 2.py"` sync duplicates. A messy working tree is fine — only what you are about to commit is checked.

It exists because two `git add -A` sweeps reached PRs under review in one session, and the second put trade-level exit prices and P&L into public history, where a commit stays retrievable by SHA even after the branch is deleted.

**Stage explicit paths. Never `git add -A`.**

## 4. `outputs/` is gitignored

Research findings and summary JSONs need `git add -f` **and** an explicit override:

```bash
ALLOW_LOCAL_FILES=1 git commit -m "... (ALLOW_LOCAL_FILES: why this path belongs)"
```

Both halves are required — the variable permits the commit, `scripts/hooks/commit-msg` demands the written reason. The variable alone leaves a reviewer seeing the file and not the justification.

## 5. CI gates

`.github/workflows/ci.yml` runs **`lint` + `test` + `migrations`**. All three green before merge, no exceptions.

- **ruff is pinned to `0.7.4`** — do not bump it inside a feature PR.
- `python -m pytest` = unit only (integration excluded), ~800 tests.
- Tests seed numpy per-test. If you see nondeterminism, that is a bug, not flake to retry past.

## 6. No agent executes unless it is named

> **No agent executes anything unless it is named as executor in the instruction it received. Silence is not authorization.** An instruction that says "go" without naming who should produce a question, not an action.

On 2026-08-15 a single unaddressed "confirmed — go" produced **two identical scheduled jobs seventeen seconds apart**, created by two different agents reading the same message. Both would have fired, double-writing the same artifact directory and falsifying its manifest hash — the exact failure the directory layout had just been designed to prevent. Caught and de-duplicated; nothing was lost.

Stated the other way round — "the drafter must name an executor" — the rule fails **open**: when the drafter forgets, everyone who can act does. Inverted, a drafting slip produces zero actions and a question.

Drafters should still name the executor explicitly, but the system must not depend on their remembering.

**Corollary: write the roster down.** An agent that re-derives who-did-what from conversation will misattribute across a session boundary. Roster, ownership, and open decisions belong in each agent's persistent memory, not only in a chat channel.

## 7. Diagnostic harnesses in `scripts/`

Seven hand-run harnesses live under `scripts/` — nothing in `src/` imports them. `sniper_component_ic.py`, `sniper_equity_curve.py`, `sniper_gap_risk.py`, `pead_e1_test.py`, `sniper_diagnostic{,2}.py`, `backfill_dryrun_full.py`. **If you are about to rebuild any of this, read them first.**

**One trap, documented in-file:** `sniper_component_ic.py` hardcodes the frozen V3 exit params (trail 1.0/0.5, zero slippage), **not** the live config (0.5/0.3, 10bp/side, gap-through). Its component ICs vs raw forward returns are the finding and are unaffected; its `trade_pnl_pct` column **must not be quoted as sniper expectancy**. Route any expectancy claim through `src/backtest/exit_engine.py` at settings-derived params first.

The two DB diagnostics read the live database and cannot run from the canonical checkout.

**Lesson from how that trap was handled:** a sibling defect in `sniper_equity_curve.py` was originally documented in a PR body instead of being fixed. That was wrong, and it was fixed properly in a follow-up. **Agents read code, not merge history.** If you find a defect, fix it or put the warning in the file.

## 8. Conventions that have already burned us

- **Never `--delete-branch` a PR that has a stacked child.** GitHub auto-closed a PR when its base branch was deleted out from under it.
- **Any artifact claiming a data source must stamp `get_last_ohlcv_provenance()`.** `fetch_ohlcv` falls back Polygon→yfinance; use `strict=True` when a run must be Polygon-only.
- **Verify defaults and callers, not just the primitive.** Every real bug in a recent month was a correct function behind unsafe wiring.
- **Confirm probe findings at full scale before concluding.** Small-sample probes routinely produce false signals that full-universe reruns reverse.
- **Check `baseRefName` before judging an old PR stale.** Two PRs closed in August read as merely old; the real disqualifier was that they targeted the archived `main-legacy` line, which is not an ancestor of `main` and has no rebase path.

## 9. Measurement discipline

Paper-sleeve results are governed by [`docs/paper_sleeve_acceptance_criteria.md`](paper_sleeve_acceptance_criteria.md), pre-registered before the measurement lane produced its first scheduled results.

Read it before quoting any number. In particular: **a stream below n = 30 is not quotable** outside a descriptive sentence that states `n` and that the CI crosses zero. This project has already retired an 82% sniper win rate and a 69.5% MR win rate that were both artifacts. The cost of quoting early numbers is not embarrassment — it is that they get built on.

## 10. Commit and PR style

- Imperative summary line. Body explains **why**, not what.
- Don't amend commits — create new ones.
- Squash-merge is the default. The one exception so far: a pre-registration document whose authority is its commit timestamp was rebase-merged, because squashing would have collapsed the timestamp into a fresh commit.
