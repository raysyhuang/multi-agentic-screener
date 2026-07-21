# MAS Truth-Paper Track v1

## Purpose

`mas-truth-paper-v1` records the reconciliation-hardened `truth-lean` candidate
alongside MAS-main without changing MAS-main selection or authorizing trades.
It is a **PAPER-only** measurement sleeve.

## Frozen policy

The control-plane policy is `config/paper_tracks/truth_lean_v1.json`.

- Candidate commit: `b54c7591f3a0228c535608d5b3387ce7f66a150f`
- Effective date: `2026-07-21`
- Policy SHA-256: computed from canonical policy JSON by
  `scripts/truth_paper_manifest.py`; do not hand-edit a reported hash.
- Alert label: `MAS-TRUTH-PAPER`

Changing a rule, candidate commit, or effective date requires a new policy file
and track ID (for example, `truth_lean_v2.json`). Never move the v1 ref or
rewrite its history.

## Isolation contract

The GitHub workflow runs from `main` because GitHub Actions schedules execute
only from the default branch. It checks out the frozen candidate in `candidate/`
and runs it with `DATABASE_URL_TRUTH_LEAN` exposed to the candidate as
`DATABASE_URL`.

`DATABASE_URL_TRUTH_LEAN` **must not** point to the MAS-main database. The two
tracks use overlapping `DailyRun(run_date)` identities, so sharing a database
would overwrite or commingle records.

The new database must be a dedicated managed PostgreSQL database (recommended),
not an ad-hoc local database. It must be reachable from GitHub-hosted runners
and have a role that can run the candidate's Alembic migrations.

## One-time provisioning

1. Provision a dedicated managed PostgreSQL database named for the paper track.
2. Construct its SSL-required connection URL. Do not commit it.
3. Set the GitHub repository secret:

   ```bash
   gh secret set DATABASE_URL_TRUTH_LEAN --repo raysyhuang/multi-agentic-screener
   ```

4. Confirm `gh secret list --repo raysyhuang/multi-agentic-screener` shows
   `DATABASE_URL_TRUTH_LEAN` (the value is intentionally never readable back).
5. Manually dispatch **MAS Truth Paper Track** with `pipeline=morning`.
6. Verify all of the following before relying on weekday cron:
   - the run uses candidate SHA `b54c759…`;
   - migrations and the morning runner complete against the new database;
   - the Telegram message is labelled `MAS-TRUTH-PAPER` and remains paper-only;
   - the GitHub run artifact contains `manifest-morning.json` only; verify the
     protected job log separately if runner diagnostics are needed;
   - MAS-main records have not changed.

The workflow fails closed when `DATABASE_URL_TRUTH_LEAN` or required provider/
Telegram secrets are absent. It retains a non-secret provenance manifest after
candidate checkout whenever possible.

## Daily evidence and promotion boundary

Each run writes the candidate commit, policy hash, track ID, pipeline type,
runner outcome, and GitHub run identity into a retained non-secret manifest.
The candidate database remains the durable record for all signals, fills,
no-fills, cancellations, censored/open positions, no-trade days, and failures.
GitHub artifacts retain only the non-secret manifest for 90 days. Candidate stdout/stderr
remains in the protected GitHub Actions job log and is deliberately **not**
exported as a downloadable artifact because provider/dependency output cannot
be treated as a trustworthy secret-free format.

This record is evidence for forward-paper evaluation only. It is not live
promotion evidence and must not produce a trade instruction outside the
explicitly labelled paper channel.
