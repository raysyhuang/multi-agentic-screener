# Comparator pinning — METHOD, pinned before execution

**Author:** Victor (Claude Code, VPS Boston / `openclaw-1`)
**Written:** 2026-08-15, **before** any production query has been run.
**Purpose:** fix the rule while the answer is still invisible, per Hawk's condition. A comparator
chosen after seeing the numbers is a chosen comparator wearing a pinned comparator's clothes.

**Status:** **ATTEMPTED 2026-08-16, COULD NOT EXECUTE.** Ray authorised the production read.
The connection was refused by the server. See §8. No query in §2 was ever run against production.

---

## 1. Which artifact is authoritative

**The production database is authoritative. Documents are not.**

Ranked, and the ranking is fixed now:

1. **Production DB `outcomes` joined to `signals`** — the system of record. Wins over everything below.
2. `outputs/research/*.md` — narrative, may quote stale or superseded numbers. **Evidence of what
   someone believed, not of what is true.**
3. Dashboard renderings, Discord messages, memory. **Never admissible.**

If the DB and `HANDOFF_gap_through_diagnosis.md` disagree, the DB wins and the document is
recorded as superseded — with its line number, so the supersession is auditable.

## 2. Population definition — the query is fixed before it runs

```sql
-- Per live book, over the full available history.
SELECT s.signal_model,
       s.signal_source,
       count(*)                                   AS n,
       round(avg(o.pnl_pct)::numeric, 4)          AS avg_pnl_pct,
       round((count(*) FILTER (WHERE o.pnl_pct > 0)::numeric
              / nullif(count(*), 0) * 100), 2)    AS win_rate_pct,
       min(o.entry_date)                          AS first_entry,
       max(o.exit_date)                           AS last_exit,
       count(DISTINCT o.entry_date)               AS distinct_entry_days
FROM outcomes o
JOIN signals s ON s.id = o.signal_id
WHERE o.still_open = false
  AND o.pnl_pct IS NOT NULL
  AND o.skip_reason IS NULL          -- skipped picks are not trades
GROUP BY 1, 2
ORDER BY n DESC;
```

**Inclusion rules, fixed in advance:**
- `still_open = false AND pnl_pct IS NOT NULL` — the doc's own definition of a closed trade. A
  `pnl_pct: null` row is not a low-confidence trade; it is not a trade. This is the exact defect
  behind the retracted 85.7% WR.
- `skip_reason IS NULL` — a skipped candidate consumed a slot but is not a trade.
- **No date filter.** Restricting the window is the most available way to select a flattering
  number, so the full history is reported and any sub-window is reported *alongside* it, never
  instead of it.
- **Per book, never pooled.** `mas_official` and the IBKR sleeve are separate populations.

**Reported for each book:** n, win rate, average `pnl_pct`, first entry, last exit, distinct entry
days, and the query's own timestamp. A figure without all seven is not pinned.

## 3. Expected row counts, stated before the query runs

So that a surprising result is visibly surprising rather than quietly accepted:

| Book | Expectation | If violated |
|---|---|---|
| MAS-GH sniper (`sniper` / `mas_official`) | n between **20 and 60** | n < 20 → the `+0.74% over 20 trades` figure is the whole population and no newer measurement exists. n > 100 → the documents are badly stale; investigate before pinning. |
| IBKR sleeve | n **unknown** — no artifact establishes it | n = 0 → the `~42% / −0.14%` figure has no basis in the DB and must be retracted outright. |

## 4. What would make each existing figure WRONG — decided now

**`+0.74%` avg / 50% WR** (`HANDOFF_gap_through_diagnosis.md:126`, n=20, CI crosses zero):
- **Wrong if** the query returns n > 20 for `sniper|mas_official` and avg `pnl_pct` differs in
  sign. Then it was a valid reading of a smaller population, since superseded.
- **Still not usable even if reproduced exactly**, because n=20 is below the n≥30 floor and its
  own source states the CI crosses zero. Reproduction makes it *honest*, not *admissible*.

**`−0.97%` avg / ~50% WR (MAS-GH) and `−0.14%` / ~42% (IBKR)**:
- **Wrong if** the query returns a materially different value for the same population, and no
  artifact is produced showing the population that generated them.
- **Retracted outright if** no artifact can be produced at all. A number in a pre-registration
  with no derivable source is worse than no number: it is unfalsifiable and it looks like evidence.

**Neither figure may be selected on the basis of which one the query resembles.** The query output
is the comparator. Both existing figures are then marked *confirmed*, *superseded*, or *retracted*
against it — a status, not a choice.

## 5. Execution controls

- **Read-only is enforced by the server, not promised.** All connections use
  `PGOPTIONS="-c default_transaction_read_only=on"`. Verified on the local mirror: a no-op
  `UPDATE ... WHERE false` returns `ERROR: cannot execute UPDATE in a read-only transaction`
  while `SELECT` succeeds. No read-only *role* exists on this host — `ray_mirror` holds full DML
  including `TRUNCATE` — so the session-level control is the available mechanism. Creating a role
  would itself be a write to the cluster and is not mine to do.
- **Rationale, per Hawk:** the mirror DB is the provenance root of the whole measurement. An
  accidental write voids the chain the pre-registration rests on. The control is about provenance
  integrity, not data safety.
- **No production connection without Ray naming it explicitly** — database, read-only intent.
  "Pin the comparator" is not authorization to read a live trading system.

## 6. Producer is not arbiter

Victor produces the numbers. **Victor does not decide whether they clear a bar.** Hawk or Ray
reviews the output against this method and rules confirmed / superseded / retracted. The pinned
result lands in `origin/main` via PR — §1 of the alignment doc requires it, and a comparator that
lives only on the VPS is not pinned in any sense that survives.

## 7. Deviations

Any departure from this method is recorded here with its reason **before** the result is reported,
never after. If the query cannot run as written, that is a finding to report — not a prompt to
improvise a different query.

## 8. Execution attempt — 2026-08-16

Ray authorised the production read. It failed, and the failure is structural rather than incidental.

**Target** (shape only, credentials never printed):
`postgres://<redacted>@c55vaqijj0vpoi.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com/d1e3fn5hnios62`
— read from `DATABASE_URL` in `/root/projects/multi-agentic-screener/.env`.

**Server response** — two distinct refusals:

```
FATAL:  password authentication failed for user "u5egpbjb9hhiom"
FATAL:  no pg_hba.conf entry for host "177.7.46.190", user "...", database "...", no encryption
```

**What this establishes:**
- The network path works. Those are PostgreSQL `FATAL` responses, not timeouts — TCP connected and
  the server answered. This is not a firewall problem.
- **The credential in the mirrored `.env` is rejected.** It is stale, rotated, or never valid for
  this user.
- **VPS Boston is not in the server's `pg_hba` allowlist** for unencrypted connections, and by IP.

**Therefore: this host structurally cannot read production, and that is very likely correct.** A
research sandbox that force-mirrors from a laptop should not hold working credentials to the live
trading database. The right conclusion is not "get Victor a password."

**Consequence for pinning:** the comparator cannot be produced from VPS Boston at all — not from the
repo, not from the local mirrors (4 and 0 closed trades), and not from production. It must be run
from a host that already holds legitimate access — Ray's Mac, or `heroku pg:psql` — by whoever holds
it, following §2 of this method verbatim.

**Not attempted, deliberately:** locating any other credential on this host, or any workaround to the
refusal. A follow-up diagnostic distinguishing stale-password from IP-allowlist was blocked by this
session's permission classifier; it was not retried by other means. The refusal is the finding.

**Side finding relevant to rotation:** the `DATABASE_URL` sitting in the mirrored `.env` does not
authenticate. If it was rotated, this copy is stale and harmless. If it was never rotated, a
non-working production credential is still sitting in a file on a sandbox host. Either way the
answer is the same — it should not be there.
