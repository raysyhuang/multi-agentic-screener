# Captured workflow-run payloads

Real `gh run view <id> --json jobs` output from `Scheduled Pipelines (GitHub-hosted)`,
checked in unmodified.

**Why these exist.** The health gate matches four strings against this workflow: the job name
`Run scheduled pipeline`, the step name `Run morning pipeline`, the workflow name, and the
`mas-run-attestation` artifact name. None appear literally in `scheduled-pipelines.yml` — the
step is `Run ${{ steps.resolve.outputs.pipeline }} pipeline`, resolved at runtime.

**These fixtures cover two of the four, deliberately.** The job name and step name are what
`worker_ran` reads, and if either drifts the gate fails **silently to pending** — the lane
stops and nothing goes red. The workflow name and the artifact name live on different
endpoints, are not present in a `--json jobs` capture, and if either drifts the gate fails
**loudly, to unhealthy and red**. The silent pair is the one worth fixturing.

**What these catch, and what they do not.** Stated precisely, because the obvious reading is
backwards.

They catch **code-side drift**: someone edits `worker_ran` so it no longer matches a payload
this workflow really produces, and a test fails.

They do **not** catch **workflow-side drift**, which is the actual outage shape. Rename the
step in `scheduled-pipelines.yml`, touch no Python, and every test here keeps passing against
the old capture while production breaks — the gate reports pending forever and nothing goes
red. Closing that direction needs an assertion against the YAML itself: the template
`Run ${{ steps.resolve.outputs.pipeline }} pipeline` together with the `morning` case value at
the resolve step. Two hops and ugly, but it is the only thing tying the code's literal to the
file that produces it. Not done here.

Hand-written dicts catch neither direction: they are written to match the code, so they only
assert that the code agrees with itself.

**The failure this guards against is not hypothetical.** If `worker_ran` silently stops
matching, the health gate never reports healthy, the daily brief skips the mirror every
morning, and the measurement lane stops without anything going red. That is precisely the
2026-08-14 → 08-20 outage, arrived at by a different route.

| File | Run | What it is |
|---|---|---|
| `morning_worker_ran.json` | 32358395667 | the real morning worker executing — the health gate cited this run on 2026-08-20 |
| `morning_dst_skip.json` | 32363434075 | the DST duplicate cron line, green overall, `Run morning pipeline` **skipped** |
| `afternoon_worker_ran.json` | 32301227211 | an afternoon run — step name differs, so this must **not** satisfy the morning gate |

Re-capture with:

```bash
gh run view <id> --repo raysyhuang/multi-agentic-screener --json jobs > <file>.json
```
