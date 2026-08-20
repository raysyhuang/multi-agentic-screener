# Captured workflow-run payloads

Real `gh run view <id> --json jobs` output from `Scheduled Pipelines (GitHub-hosted)`,
checked in unmodified.

**Why these exist.** `mas_github_pipeline_health.worker_ran` matches four strings against
this workflow — the job name `Run scheduled pipeline`, the step name `Run morning pipeline`,
and elsewhere the workflow name and the `mas-run-attestation` artifact name. None of those
appear literally in `scheduled-pipelines.yml`: the step is
`Run ${{ steps.resolve.outputs.pipeline }} pipeline`, resolved at runtime.

Hand-written test dicts cannot catch a rename, because they are written to match the code
rather than the workflow. A captured payload can. If someone renames a step and these
fixtures are not re-captured, the tests that read them fail — which is the point.

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
