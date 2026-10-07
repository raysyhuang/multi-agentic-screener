"""Phase A of the PIT universe build — the membership spine.

Contract: outputs/research/PIT_UNIVERSE_CONTRACT.md (v3: frozen #77 + #79, rulings
R1-R9 and the Phase B budget of 2026-10-06 in §12 and its addendum).

Phase A acquires everything except market cap:

    grouped daily bars      1 call per ET session      ~750
    reference list          monthly, paginated         ~216
    classification audit    200 pairs per month        ~7,200
                                                       -------
                                                       ~8,200

Deliverable is data plus evidence — a manifest, a diagnostic report, and the
distinct-ticker count that authorises or aborts Phase B. **No performance
numbers, no strategy consumption.**

Design rules taken from the contract, not invented here:

  * raw response written BEFORE it is parsed, so a build is replayable from
    frozen bytes rather than from a re-query (§5);
  * resumable — an existing raw file is never refetched, so an interrupted run
    resumes instead of re-spending calls (§A.3);
  * every date is an ET market date (§0);
  * classification is forward-held monthly and audited, never treated as
    daily-exact (§3a/§3b);
  * the audit samples the PRE-classification population, stratified, so false
    exclusion is reachable and not just contamination (§3b).

Usage:
    python scripts/pit_universe_phase_a.py spine       [--start 2017-01-03] [--vintage ET-DATE]
    python scripts/pit_universe_phase_a.py transitions [--vintage ET-DATE]
    python scripts/pit_universe_phase_a.py audit       [--vintage ET-DATE]
    python scripts/pit_universe_phase_a.py report  [--vintage ET-DATE]
    python scripts/pit_universe_phase_a.py verify  [--vintage ET-DATE] [--manifest PATH]
    python scripts/pit_universe_phase_a.py package [--vintage ET-DATE]
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from src.config import get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pit")

# httpx logs the full request URL at INFO — including `apiKey=` when the key is
# passed as a query parameter. That puts a live Polygon key into stdout, into
# any redirected log file, and into CI output. Silenced here, and the key is
# sent as a header below so it never appears in a URL at all.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

BASE = "https://api.polygon.io"
ROOT = Path(__file__).resolve().parent.parent / "outputs" / "pit_universe"

# Live screener constraints that Phase A can evaluate. Market cap is Phase B.
MIN_PRICE = 5.0
MIN_SHARE_VOLUME = 500_000
ALLOWED_EXCHANGES = {"NYSE", "NASDAQ"}
# R3 (contract §12): membership mirrors the LIVE eligibility constraints (§2),
# and live admits ADR common stock (FMP isEtf/isFund=false). `CS` alone dropped
# 23 liquid ADRs the book actually trades, PBR picked at rank 1.
ELIGIBLE_TYPES = frozenset({"CS", "ADRC"})
# R7 (§12): the momentum core needs >= 84 monthly observations plus a 12-month
# warm-up, so >= 8 years. Polygon serves survivorship-free grouped bars and
# as-of reference data back to at least 2016 (probed 2026-10-06).
DEFAULT_START = date(2017, 1, 3)
# Contract versions. A vintage is normalized under the rules it was ACQUIRED
# under, recorded in <vintage>/contract.json at spine time. A vintage without the
# stamp predates v3 and keeps v2 semantics (CS only, monthly labels with no
# transition overrides), so a frozen artifact still replays byte-identically.
CONTRACT_VERSION = "v3"
_V2_ELIGIBLE_TYPES = frozenset({"CS"})
RULINGS = "R1-R9 + Phase B budget, 2026-10-06 (contract §12 and its addendum)"
# §11 history rule: >= 200 prior bars as of D. A v3 vintage acquires this many
# warm-up sessions BEFORE its start so early dates can be evaluated; warm-up
# sessions feed the history count only and never emit membership.
MIN_PRIOR_BARS = 200
WARMUP_SESSIONS = MIN_PRIOR_BARS


def contract_version(vintage: str) -> str:
    stamp = ROOT / vintage / "contract.json"
    if stamp.exists():
        return json.loads(stamp.read_text()).get("version", "v2")
    return "v2"


def eligible_types_for(vintage: str) -> frozenset[str]:
    return ELIGIBLE_TYPES if contract_version(vintage) == "v3" else _V2_ELIGIBLE_TYPES


def _stamp_contract(vintage: str, start: date, end: date) -> dict:
    """Create (or validate) the vintage's contract stamp, freezing its range.

    The range is fixed at creation: a resume must never extend a vintage just
    because the calendar moved on. A non-empty vintage without a stamp predates
    v3 and is never re-stamped as v3.
    """
    base = ROOT / vintage
    stamp = base / "contract.json"
    if stamp.exists():
        existing = json.loads(stamp.read_text())
        if existing.get("version") != CONTRACT_VERSION:
            raise RuntimeError(
                f"vintage {vintage} was acquired under contract {existing.get('version')}; "
                f"refusing to extend it under {CONTRACT_VERSION} — start a new vintage")
        if not existing.get("start") or not existing.get("end"):
            raise RuntimeError(f"vintage {vintage} stamp has no frozen range")
        return existing
    if (base / "raw").exists() and any((base / "raw").iterdir()):
        raise RuntimeError(
            f"vintage {vintage} already holds raw data but no contract stamp (v2); "
            "refusing to re-stamp it as v3 — start a new vintage")
    if start != DEFAULT_START:
        raise RuntimeError(
            f"contract v3 fixes the range start at {DEFAULT_START} (R7); got {start}. "
            "A different range is a different contract version.")
    stamp.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "version": CONTRACT_VERSION,
        "eligible_types": sorted(ELIGIBLE_TYPES),
        "transition_resolution": "§3a-v2 (assumes at most one membership-relevant change "
                                 "per ticker per month; residual risk carried by the §3b audit)",
        "price_rule": "close > 5.00 (strict)", "volume_rule": "share volume > 500,000 (strict)",
        "min_prior_bars": MIN_PRIOR_BARS, "warmup_sessions": WARMUP_SESSIONS,
        "start": str(start), "end": str(end),
        "rulings": RULINGS,
    }
    stamp.write_text(json.dumps(rec, indent=2))
    return rec


def frozen_sessions(vintage: str) -> tuple[list[date], list[date]]:
    """(warm-up sessions, membership sessions) for a v3 vintage, from its stamp."""
    import pandas_market_calendars as mcal

    rec = json.loads((ROOT / vintage / "contract.json").read_text())
    start, end = date.fromisoformat(rec["start"]), date.fromisoformat(rec["end"])
    cal = mcal.get_calendar("NYSE")
    main = [d.date() for d in cal.schedule(start_date=start, end_date=end).index]
    pre = [d.date() for d in cal.schedule(start_date=start - timedelta(days=500),
                                          end_date=start - timedelta(days=1)).index]
    n_warm = int(rec.get("warmup_sessions", WARMUP_SESSIONS))
    # pre[-0:] is the WHOLE list, not an empty one: zero must be explicit.
    return (pre[-n_warm:] if n_warm > 0 else []), main


def validated_sessions(vintage: str) -> tuple[list[date], list[date]]:
    """Frozen (warm-up, main) sessions, refusing a grouped tree that differs.

    A missing grouped file is an incomplete spine; an extra one (stale, stray,
    or outside the frozen range) would emit membership or shift the history
    count. Either is refused rather than silently tolerated.
    """
    warmup, main = frozen_sessions(vintage)
    grouped_dir = ROOT / vintage / "raw" / "grouped"
    present = {date.fromisoformat(p.stem.replace(".json", "")) for p in grouped_dir.glob("*.json.gz")}
    expected = set(warmup) | set(main)
    missing, extra = sorted(expected - present), sorted(present - expected)
    if missing or extra:
        raise RuntimeError(
            f"grouped tree does not match the frozen sessions: {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(extra)} extra (e.g. {extra[:3]}) — re-run `spine`")
    raw_dir = ROOT / vintage / "raw" / "grouped_raw"
    raw_present = {date.fromisoformat(p.stem.replace(".json", "")) for p in raw_dir.glob("*.json.gz")}
    if raw_present != set(main):
        raise RuntimeError(
            f"grouped_raw tree does not match the membership sessions: "
            f"{len(set(main) - raw_present)} missing, {len(raw_present - set(main))} extra — re-run `spine`")
    return warmup, main


def trailing_snapshot_date(vintage: str) -> date:
    """The session after the frozen end: the comparison snapshot for the final month."""
    import pandas_market_calendars as mcal

    end = date.fromisoformat(json.loads((ROOT / vintage / "contract.json").read_text())["end"])
    sched = mcal.get_calendar("NYSE").schedule(start_date=end + timedelta(days=1),
                                               end_date=end + timedelta(days=10))
    return sched.index[0].date()


def _trailing_labels(vintage: str) -> dict[str, dict] | None:
    """Labels as of the trailing snapshot date, or None if it was not acquired.

    Used ONLY to detect transitions inside the final membership month (there is
    no later monthly snapshot to compare with). It never labels a session.
    """
    trail = trailing_snapshot_date(vintage)
    snap_dir = ROOT / vintage / "raw" / "reference_trailing" / str(trail)
    if not (snap_dir / SNAPSHOT_MARKER).exists():
        return None                       # absent or incomplete: candidate generation refuses
    return _read_snapshot(snap_dir, require_complete=True, expected_as_of=trail)
_EXCHANGE_MAP = {
    "XNYS": "NYSE", "XNAS": "NASDAQ", "XASE": "AMEX",
    "ARCX": "NYSE", "BATS": "NASDAQ",
    "XNGS": "NASDAQ", "XNCM": "NASDAQ", "XNMS": "NASDAQ",
}

# §3b audit. Sampler version is recorded in the manifest; changing any of these
# constants is a version bump, never a silent edit.
SAMPLER_VERSION = "phase-a/2"  # contract v3: ELIGIBLE_TYPES + transition-resolved labels
AUDIT_SEED = 20260812
AUDIT_PAIRS_PER_MONTH = 200
AUDIT_BUCKETS = ("common_stock", "etf_fund_other", "unknown")

# Conservative pacing. The contract requires one request in flight per endpoint
# family and no burst parallelism (§A.3).
REQUEST_DELAY_S = 0.12

# Hard Phase A ceiling. Contract v3 (R7: ~9.75 years) authorises ~2,450 grouped
# + ~1,500 reference pages + ~23,600 audit pairs + transition resolution, which
# is <= 5 probes per candidate (§3a-v2). The ceiling is enforced in `_get`
# and aborts the run, because a budget that is only ever compared against an
# estimate AFTER the run is not a budget — the 1.9M-call naive build this design
# exists to avoid would have been discovered the same way.
PHASE_A_CALL_CEILING = 45_000
# Phase B (Ray, 2026-10-06): shares lookups only for (ticker, quarter) pairs in
# which the ticker passed every non-market-cap constraint on >= 1 session —
# ~117.5k on this vintage — plus the §3d threshold audit (~75/month) and retries.
PHASE_B_CALL_CEILING = 140_000
def _ceiling_for(phase: str) -> int:
    return {"A": PHASE_A_CALL_CEILING, "B": PHASE_B_CALL_CEILING}[phase]
# §3a-v2 binary search costs ceil(log2(sessions in month)) <= 5 probes per transition.


class BudgetExceeded(RuntimeError):
    """Raised when the run would exceed PHASE_A_CALL_CEILING."""


class RequestLedger:
    """Append-only record of every request the run issues.

    Durable and flushed per line: a run killed mid-flight must leave behind an
    accurate account of what it spent, otherwise resuming double-counts and the
    ceiling protects nothing. Records outcomes too — a request that failed after
    exhausting retries is spend that bought no data, and it must be visible as
    such rather than inferred from a gap in the raw tree.

    Params are stored as a digest, never verbatim: they are low-cardinality here
    and a ledger is a file we may attach to a public artifact.
    """

    def __init__(self, path: Path, ceiling: int | None = None) -> None:
        self.path = path
        # Resolved at construction, not at import, so the module constant stays
        # the single source of truth (and is patchable in tests).
        self.ceiling = ceiling if ceiling is not None else PHASE_A_CALL_CEILING
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.calls = self._replay_count()
        self.failures: list[dict] = []
        self._fh = open(self.path, "a", buffering=1)  # line-buffered

    def _replay_count(self) -> int:
        """Resume the counter from a prior run so the ceiling spans attempts."""
        if not self.path.exists():
            return 0
        n = 0
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    # Parsed, not substring-matched: json.dumps emits
                    # '"event": "request"' with a space, so the obvious
                    # `'"event":"request"' in line` check silently counts zero
                    # and every restart resets the ceiling to full budget.
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    # A skipped line understates spend and so overstates the
                    # remaining budget. Fail closed.
                    raise RuntimeError(f"unparseable ledger line in {self.path}: {line[:80]!r}") from exc
                if not isinstance(rec, dict) or rec.get("event") not in ("request", "failure"):
                    raise RuntimeError(f"unknown ledger event in {self.path}: {line[:80]!r}")
                if rec["event"] == "request":
                    n += 1
        return n

    def would_exceed(self) -> bool:
        return self.calls >= self.ceiling

    @staticmethod
    def _digest(params: dict) -> str:
        return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()

    def record(self, url: str, params: dict, status: int | str, attempt: int,
               allow_404: bool = False) -> None:
        self.calls += 1
        full = self._digest(params)
        self._fh.write(json.dumps({
            "event": "request", "n": self.calls,
            "endpoint": url.replace(BASE, ""),
            "params_sha256": full[:16], "params_sha256_full": full,
            "status": status, "attempt": attempt,
            # Whether a 404 is an observation (allow_404) or an error for this
            # request: only the former may count as an answer.
            "allow_404": allow_404,
        }) + "\n")

    def record_failure(self, url: str, params: dict, reason: str, attempts: int) -> None:
        full = self._digest(params)
        rec = {
            "event": "failure", "endpoint": url.replace(BASE, ""),
            "params_sha256": full[:16], "params_sha256_full": full,
            "reason": reason, "attempts": attempts,
        }
        self.failures.append(rec)
        self._fh.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        self._fh.close()


# Bound to the active run by `_open_ledger`; `_get` refuses to issue without it,
# so an unmetered request path cannot be introduced by accident.
_LEDGER: RequestLedger | None = None


def _open_ledger(vintage: str, phase: str = "A") -> RequestLedger:
    global _LEDGER
    name = "request_ledger.jsonl" if phase == "A" else f"request_ledger_phase_{phase.lower()}.jsonl"
    _LEDGER = RequestLedger(ROOT / vintage / name, _ceiling_for(phase))
    # Recording disallowed client errors as durable failures is a v3 ledger
    # rule (v3 understands recovery); v2 vintages keep their original records.
    _LEDGER.record_client_errors = contract_version(vintage) == "v3"
    logger.info(
        "ledger (phase %s): %d calls already spent on this vintage, ceiling %d",
        phase, _LEDGER.calls, _LEDGER.ceiling,
    )
    return _LEDGER


# ── raw layer ────────────────────────────────────────────────────────────────

def _raw_path(vintage: str, *parts: str) -> Path:
    return ROOT / vintage / "raw" / Path(*parts)


def _write_raw(path: Path, payload: dict) -> str | None:
    """No-op for a failed fetch, so the hole stays visible to a resume.

    Writing a `_failed` sentinel into the raw tree would make it indistinguishable
    from data on the next run: resume logic keys on file existence, so the hole
    would be permanently skipped and silently normalized as "no results". The
    failure is already durable in the ledger; the raw tree stays a record of what
    was actually retrieved.
    """
    if payload.get("_failed"):
        return None
    return _write_raw_unchecked(path, payload)


def _write_raw_unchecked(path: Path, payload: dict) -> str:
    """Persist a raw response and return its content hash.

    Written before anything parses it: the normalized dataset must be a pure
    function of these bytes, or replay determinism is a claim rather than a
    property.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with gzip.open(path, "wb") as fh:
        fh.write(blob)
    return hashlib.sha256(blob).hexdigest()


def _read_raw(path: Path) -> dict:
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read())


async def _get(
    client: httpx.AsyncClient, url: str, params: dict, allow_404: bool = False
) -> dict:
    """One GET with backoff. Never parallel — see §A.3.

    ``allow_404`` returns a sentinel instead of raising. For the audit a 404 is
    an observation — the vendor has no date-specific record for that ticker/date
    — and it must be recorded as UNVERIFIABLE rather than crashing the run or,
    worse, being counted as agreement. Test symbols like ZVZZT produce it.
    """
    if _LEDGER is None:
        raise RuntimeError("no request ledger bound — call _open_ledger() first")

    settings = get_settings()
    # Bearer header, not a query parameter. A URL carrying the key ends up in
    # client logs, proxy logs, and exception messages; a header does not.
    headers = {"Authorization": f"Bearer {settings.polygon_api_key}"}
    attempts = 6
    for attempt in range(1, attempts + 1):
        # Checked before EVERY outbound attempt, not once per logical request.
        # Checking once and then retrying up to 6 times overshoots the ceiling by
        # the retry count: starting at 11,999/12,000, a repeated 503 ends at
        # 12,005. A budget that a retry storm can walk through is not a budget,
        # and the failure mode is worst exactly when the vendor is unhealthy and
        # retries are most frequent.
        if _LEDGER.would_exceed():
            _LEDGER.record_failure(url, params, "budget_ceiling_reached", attempt)
            raise BudgetExceeded(
                f"call ceiling {_LEDGER.ceiling} reached ({_LEDGER.calls} "
                f"spent) on attempt {attempt}. Raising it is a contract change, "
                "not a flag."
            )
        try:
            resp = await client.get(url, params=params, headers=headers, timeout=60)
        except httpx.HTTPError as e:
            _LEDGER.record(url, params, f"network:{type(e).__name__}", attempt, allow_404)
            if attempt == attempts:
                _LEDGER.record_failure(url, params, f"network:{type(e).__name__}", attempt)
                return {"results": None, "_failed": True, "_reason": type(e).__name__}
            logger.warning("network error (%s), retry %d", type(e).__name__, attempt)
            await asyncio.sleep(2 ** attempt)
            continue

        _LEDGER.record(url, params, resp.status_code, attempt, allow_404)

        if resp.status_code == 429:
            wait = min(60, 2 ** attempt)
            logger.warning("429 rate limited, sleeping %ss", wait)
            await asyncio.sleep(wait)
            continue
        if allow_404 and resp.status_code == 404:
            await asyncio.sleep(REQUEST_DELAY_S)
            return {"results": None, "_not_found": True}
        if resp.status_code >= 500:
            # A 5xx is the vendor's problem and is usually transient. Previously
            # this fell through to raise_for_status() and killed the run: one bad
            # shard mid-way through 8,000 calls discarded every call after it.
            # Bounded retry, then a DURABLE failure record and continue — a run
            # with a recorded hole is auditable, a run that died is not.
            if attempt == attempts:
                _LEDGER.record_failure(url, params, f"http_{resp.status_code}", attempt)
                logger.error("5xx exhausted for %s — recorded as a hole", url)
                return {"results": None, "_failed": True, "_reason": f"http_{resp.status_code}"}
            wait = min(60, 2 ** attempt)
            logger.warning("HTTP %d, retry %d in %ss", resp.status_code, attempt, wait)
            await asyncio.sleep(wait)
            continue

        if resp.status_code >= 400 and getattr(_LEDGER, "record_client_errors", False):
            # Any other client error is fatal for this request; record it as a
            # durable failure first so the ledger never shows it as answered.
            _LEDGER.record_failure(url, params, f"http_{resp.status_code}", attempt)
        resp.raise_for_status()
        await asyncio.sleep(REQUEST_DELAY_S)
        return resp.json()

    _LEDGER.record_failure(url, params, "retries_exhausted", attempts)
    return {"results": None, "_failed": True, "_reason": "retries_exhausted"}


def _fetch_live_snapshot(vintage: str) -> Path:
    """Freeze the public dashboard export as raw evidence for the §4 check.

    Stored in the raw tree and hashed into the manifest like any other input:
    the divergence result must be reproducible from frozen bytes, not from
    whatever the live site happens to serve when the report is re-run. The
    dashboard is already public, so this adds no disclosure.
    """
    import urllib.request

    url = "https://raysyhuang.github.io/multi-agentic-screener/data.json"
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        payload = json.loads(resp.read())
    path = _raw_path(vintage, "live", "dashboard.json.gz")
    digest = _write_raw_unchecked(path, payload)
    logger.info(
        "live snapshot: %d candidates, %d run_history rows, sha256=%s",
        len(payload.get("candidates") or []), len(payload.get("run_history") or []),
        digest[:16],
    )
    return path


# ── trading calendar ─────────────────────────────────────────────────────────

def _today_et() -> date:
    """Today's ET market date — never the local date.

    `date.today()` returns the machine's local date. On a UTC+8 host that is
    tomorrow's date for most of the ET trading day, which would put an unstarted
    session into the range and fetch an empty or partial bar file. This is the
    same defect the contract's §0 exists to prevent and that #78 fixed in the
    smoke test; it reappeared in the first function written against the
    contract.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/New_York")).date()


def et_sessions(years: float | None = None, start: date | None = None) -> list[date]:
    """Actual NYSE sessions, so holidays cost no calls.

    Ends at the last COMPLETED session: today's bars do not exist until the
    session closes, and a partial file frozen into a vintage would be worse
    than a missing one. ``start`` (contract v3) takes precedence over ``years``.
    """
    import pandas_market_calendars as mcal

    end = _today_et() - timedelta(days=1)
    if start is None:
        start = end - timedelta(days=int(365.25 * (years if years is not None else 3.0)))
    sched = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    return [d.date() for d in sched.index]


# ── step 1: spine ────────────────────────────────────────────────────────────

async def fetch_spine(vintage: str, years: float | None = None, start: date | None = None) -> None:
    if years is not None:
        raise RuntimeError("contract v3 fixes the range (R7); --years is not accepted")
    sessions_now = et_sessions(None, start or DEFAULT_START)
    _stamp_contract(vintage, start or DEFAULT_START, sessions_now[-1])
    warmup, sessions = frozen_sessions(vintage)
    main_sessions = list(sessions)
    # Reference snapshots are needed for membership months only; warm-up
    # sessions need bars (for the prior-bar count) and nothing else.
    months = sorted({(d.year, d.month) for d in sessions})
    sessions = warmup + sessions
    logger.info(
        "spine: %d ET sessions, %d monthly reference snapshots",
        len(sessions), len(months),
    )

    async with httpx.AsyncClient() as client:
        fetched = skipped = holes = 0
        for d in sessions:
            path = _raw_path(vintage, "grouped", f"{d}.json.gz")
            if path.exists():
                skipped += 1
                continue
            payload = await _get(
                client, f"{BASE}/v2/aggs/grouped/locale/us/market/stocks/{d}",
                {"adjusted": "true"},
            )
            if payload.get("_failed"):
                holes += 1
                logger.error("  grouped %s: unrecoverable (%s)", d, payload.get("_reason"))
                continue
            _write_raw(path, payload)
            fetched += 1
            if fetched % 50 == 0:
                logger.info("  grouped: %d fetched, %d already present", fetched, skipped)
        logger.info(
            "grouped daily done: %d fetched, %d resumed, %d HOLES", fetched, skipped, holes
        )

        # R9: unadjusted bars for the membership sessions (price/volume/mcap as
        # observable on D). Warm-up sessions only feed the history count.
        raw_fetched = 0
        for d in main_sessions:
            path = _raw_path(vintage, "grouped_raw", f"{d}.json.gz")
            if path.exists():
                continue
            payload = await _get(
                client, f"{BASE}/v2/aggs/grouped/locale/us/market/stocks/{d}",
                {"adjusted": "false"},
            )
            if payload.get("_failed"):
                holes += 1
                logger.error("  grouped_raw %s: unrecoverable (%s)", d, payload.get("_reason"))
                continue
            payload["_request"] = {"date": str(d), "adjusted": False}
            _write_raw(path, payload)
            raw_fetched += 1
            if raw_fetched % 100 == 0:
                logger.info("  grouped_raw: %d fetched", raw_fetched)
        # R9: every split executing from the warm-up start to the frozen end.
        sreq = splits_request(vintage)
        if not await _fetch_paged(client, vintage, ("splits",), f"{BASE}{sreq['endpoint']}",
                                  sreq["params"], main_sessions[-1]):
            holes += 1

        # Monthly classification snapshot, taken on the first session of each
        # month and applied FORWARD ONLY (§3a). A snapshot counts only once its
        # pagination has completed (marker written): a truncated snapshot is
        # worse than an absent one, since every ticker on the unreached pages
        # would silently become type_unknown.
        ref_holes: list[str] = []
        warm = set(warmup)
        for year, month in months:
            snap = next(d for d in sessions if (d.year, d.month) == (year, month) and d not in warm)
            sub = ("reference", f"{year:04d}-{month:02d}")
            if not await _fetch_snapshot(client, vintage, sub, snap):
                ref_holes.append("/".join(sub))
        # Trailing comparison snapshot (final month only; never labels a session).
        # Only once that session has COMPLETED: before the close, an as-of query
        # for today is not a post-final-session observation.
        trail = trailing_snapshot_date(vintage)
        if not _session_complete(trail):
            ref_holes.append(f"reference_trailing/{trail} (session not yet complete)")
        elif not await _fetch_snapshot(client, vintage, ("reference_trailing", str(trail)), trail):
            ref_holes.append(f"reference_trailing/{trail}")
        if holes or ref_holes:
            logger.error(
                "spine INCOMPLETE: %d grouped hole(s), %d reference hole(s) %s — "
                "re-run to fill before reporting",
                holes, len(ref_holes), ref_holes[:5],
            )


def _session_complete(d: date) -> bool:
    """True once session d's SCHEDULED close + 15 min has passed (early closes honoured)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pandas_market_calendars as mcal

    now = datetime.now(ZoneInfo("America/New_York"))
    if d < now.date():
        return True
    if d > now.date():
        return False
    sched = mcal.get_calendar("NYSE").schedule(start_date=d, end_date=d)
    if sched.empty:
        return False
    close = sched["market_close"].iloc[0].to_pydatetime() + timedelta(minutes=15)
    return now >= close


SNAPSHOT_MARKER = "_complete.json"


async def _fetch_snapshot(client, vintage: str, sub: tuple[str, ...], as_of: date) -> bool:
    """Reference-tickers snapshot as of a date (see `_fetch_paged`)."""
    return await _fetch_paged(client, vintage, sub, f"{BASE}/v3/reference/tickers",
                              {"market": "stocks", "date": str(as_of), "limit": 1000}, as_of)


async def _fetch_paged(client, vintage: str, sub: tuple[str, ...], url: str,
                       base_params: dict, as_of: date) -> bool:
    """Page through a cursor-paginated endpoint; mark complete only at the last page.

    Resumable: existing pages are re-read (their next_url drives the walk), so a
    run interrupted mid-pagination continues where it stopped and writes the
    completion marker only when a page with no next_url is reached.
    """
    marker = _raw_path(vintage, *sub, SNAPSHOT_MARKER)
    snap_dir = marker.parent
    request_id = _request_identity(url, base_params)
    if marker.exists():
        legacy = "endpoint" not in json.loads(marker.read_text())
        if not legacy:
            _read_paged(snap_dir, True, as_of, request_id)  # raises if invalid: fail closed
            return True
        # Completed under the pre-identity format: it cannot prove which query it
        # answers. Quarantine the whole directory (never delete) and refetch.
        aside = snap_dir.with_name(f"{snap_dir.name}.untrusted-{_utc_stamp()}")
        if aside.exists():
            raise RuntimeError(f"quarantine target {aside} already exists")
        snap_dir.rename(aside)
        logger.warning("  snapshot %s: legacy completed snapshot moved to %s; refetching",
                       "/".join(sub), aside.name)
    # Existing unmarked pages are reused only if each one PROVES it belongs to
    # this snapshot: requested for this as-of date with exactly the cursor the
    # previous page handed out. Pages without that provenance (written by older
    # code, or from another chain) are moved aside — raw data is never deleted —
    # and the snapshot is fetched again from page 1.
    if snap_dir.exists() and not _pages_have_provenance(snap_dir, as_of, request_id):
        aside = snap_dir.with_name(f"{snap_dir.name}.untrusted-{_utc_stamp()}")
        if aside.exists():
            raise RuntimeError(f"quarantine target {aside} already exists")
        snap_dir.rename(aside)
        logger.warning("  snapshot %s: unproven pages moved to %s; refetching", "/".join(sub), aside.name)
    page, cursor, hashes = 1, None, []
    while True:
        path = _raw_path(vintage, *sub, f"page-{page}.json.gz")
        if path.exists():
            payload = _read_raw(path)
        else:
            params = dict(base_params)
            if cursor:
                params["cursor"] = cursor
            payload = await _get(client, url, params)
            if payload.get("_failed"):
                logger.error("  snapshot %s page %d unrecoverable (%s) — left INCOMPLETE",
                             "/".join(sub), page, payload.get("_reason"))
                return False
            payload["_request"] = {"as_of": str(as_of), "cursor": cursor, **request_id}
            _write_raw(path, payload)
        hashes.append(_sha256_file(path))
        nxt = payload.get("next_url")
        if not nxt:
            break
        cursor = _cursor_of(nxt)
        page += 1
    marker.write_text(json.dumps({"as_of": str(as_of), "pages": page, "page_sha256": hashes,
                                  **request_id}))
    logger.info("  snapshot %s: %d page(s), complete", "/".join(sub), page)
    return True


def _cursor_of(next_url: str | None) -> str | None:
    """The `cursor` query parameter of a next_url — parsed, never string-split.

    Splitting on 'cursor=' would swallow any parameters after it ('abc&x=1').
    Exactly one non-empty cursor is required; anything else is a broken chain.
    """
    if not next_url:
        return None
    from urllib.parse import parse_qs, urlsplit
    values = parse_qs(urlsplit(next_url).query).get("cursor", [])
    if len(values) != 1 or not values[0]:
        raise RuntimeError(f"next_url without exactly one cursor: {next_url[:80]!r}")
    return values[0]


def _page_number(path: Path) -> int | None:
    """N for an exact 'page-N.json.gz' name, else None (never raises)."""
    import re
    m = re.fullmatch(r"page-([1-9][0-9]*)\.json\.gz", path.name)
    return int(m.group(1)) if m else None


def _request_identity(url: str, base_params: dict) -> dict:
    """Endpoint and canonical non-cursor parameters: what a page claims to answer."""
    return {"endpoint": url.replace(BASE, ""),
            "params": {k: str(v) for k, v in sorted(base_params.items())}}


REFERENCE_ENDPOINT = "/v3/reference/tickers"
SPLITS_ENDPOINT = "/v3/reference/splits"


def reference_request(as_of: date) -> dict:
    return _request_identity(f"{BASE}{REFERENCE_ENDPOINT}",
                             {"market": "stocks", "date": str(as_of), "limit": 1000})


def splits_request(vintage: str) -> dict:
    warmup, main = frozen_sessions(vintage)
    return _request_identity(f"{BASE}{SPLITS_ENDPOINT}", {
        "execution_date.gte": str(warmup[0] if warmup else main[0]),
        "execution_date.lte": str(main[-1]), "limit": 1000, "order": "asc",
        "sort": "execution_date"})


def _pages_have_provenance(snap_dir: Path, as_of: date, request_id: dict | None = None) -> bool:
    """Every existing page was requested for as_of with its predecessor's cursor."""
    files = list(snap_dir.glob("page-*"))
    numbers = [_page_number(p) for p in files]
    if any(n is None for n in numbers):
        return False                          # a malformed page name is unproven
    pages = [p for _, p in sorted(zip(numbers, files))]
    if [p.name for p in pages] != [f"page-{i}.json.gz" for i in range(1, len(pages) + 1)]:
        return False
    expected_cursor = None
    for p in pages:
        try:
            payload = _read_raw(p)
            req = payload.get("_request")
            if not req or req.get("as_of") != str(as_of) or req.get("cursor") != expected_cursor:
                return False
            if request_id is not None and any(req.get(k) != v for k, v in request_id.items()):
                return False
            expected_cursor = _cursor_of(payload.get("next_url"))
        except Exception:  # noqa: BLE001 — corrupt bytes or a malformed next_url are unproven
            return False
    return True


def _utc_stamp() -> str:
    """Quarantine suffix: UTC time plus a random tag, so two quarantines never collide."""
    import uuid
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_snapshot(snap_dir: Path, require_complete: bool,
                   expected_as_of: date | None = None) -> dict[str, dict]:
    """Labels from one reference snapshot directory (validated by `_read_paged`)."""
    labels: dict[str, dict] = {}
    req = reference_request(expected_as_of) if (require_complete and expected_as_of) else None
    for payload in _read_paged(snap_dir, require_complete, expected_as_of, req):
        for row in payload.get("results", []):
            t = row.get("ticker")
            if t:
                labels[t] = {"type": row.get("type"),
                             "exchange": _EXCHANGE_MAP.get(row.get("primary_exchange", ""), "")}
    return labels


def _read_paged(snap_dir: Path, require_complete: bool,
                expected_as_of: date | None = None, request_id: dict | None = None) -> list[dict]:
    """Payloads of one paged directory; v3 refuses anything not provably complete.

    Complete means: a marker for the expected as-of date, pages 1..N contiguous
    with N >= 1, one recorded hash per page matching the stored bytes, every page
    valid JSON, every non-final page carrying a next_url and the final page none.
    """
    marker = snap_dir / SNAPSHOT_MARKER
    files = list(snap_dir.glob("page-*"))
    numbers = [_page_number(p) for p in files]
    if require_complete and any(n is None for n in numbers):
        raise RuntimeError(f"snapshot {snap_dir.name}: malformed page file name(s)")
    pages = [p for n, p in sorted((n, p) for n, p in zip(numbers, files) if n is not None)]
    payloads: list[dict] = []
    if require_complete:
        name = f"{snap_dir.parent.name}/{snap_dir.name}"
        if not marker.exists():
            raise RuntimeError(f"snapshot {name} has no completion marker — pagination "
                               "incomplete; re-run `spine`")
        m = json.loads(marker.read_text())
        n = int(m.get("pages", 0))
        if n < 1 or len(m.get("page_sha256", [])) != n:
            raise RuntimeError(f"snapshot {name}: malformed marker")
        if expected_as_of is not None and m.get("as_of") != str(expected_as_of):
            raise RuntimeError(f"snapshot {name}: marker as_of {m.get('as_of')} != expected {expected_as_of}")
        if request_id is not None and any(m.get(k) != v for k, v in request_id.items()):
            raise RuntimeError(f"snapshot {name}: marker request {m.get('endpoint')} {m.get('params')} "
                               f"!= expected {request_id}")
        names = [p.name for p in pages]
        if names != [f"page-{i}.json.gz" for i in range(1, n + 1)]:
            raise RuntimeError(f"snapshot {name}: pages {names[:3]}... do not match marker")
        if [_sha256_file(p) for p in pages] != m["page_sha256"]:
            raise RuntimeError(f"snapshot {name}: page bytes changed since completion")
        payloads = [_read_raw(p) for p in pages]
        expected_cursor = None
        for i, payload in enumerate(payloads, 1):
            has_next = bool(payload.get("next_url"))
            if (i < n and not has_next) or (i == n and has_next):
                raise RuntimeError(f"snapshot {name}: pagination chain broken at page {i} of {n}")
            req = payload.get("_request") or {}
            if (req.get("as_of") != m.get("as_of") or req.get("cursor") != expected_cursor
                    or (request_id is not None and any(req.get(k) != v for k, v in request_id.items()))):
                raise RuntimeError(f"snapshot {name}: page {i} provenance does not continue the chain "
                                   f"(as_of {req.get('as_of')}, cursor {req.get('cursor')!r})")
            expected_cursor = _cursor_of(payload.get("next_url"))
    else:
        payloads = [_read_raw(p) for p in pages]
    return payloads


# ── R9: point-in-time observables (unadjusted bars, splits) ──────────────────
#
# Membership on day D must use what was observable on D. Split-adjusted closes
# rewrite history: a $1 stock that later reverse-splits 1:10 shows a $10
# adjusted close years earlier and passes the $5 floor (registry
# R-2026-09-price-screen-lookahead); and quarterly shares x an adjusted close
# misstates market cap by the split ratio (NVDA 2024-06-07: $120.89 adjusted vs
# $1,208.88 actual, so 2.5B shares gave $302B instead of ~$3.02T). Price,
# volume and market cap therefore use UNADJUSTED bars; adjusted bars remain for
# history counting and for returns.

def _raw_bars(vintage: str, d: date) -> dict[str, dict]:
    path = ROOT / vintage / "raw" / "grouped_raw" / f"{d}.json.gz"
    return {r["T"]: r for r in (_read_raw(path).get("results", []) or []) if r.get("T")}


def splits_by_ticker(vintage: str) -> dict[str, list[tuple[date, float]]]:
    """{ticker: [(execution_date, shares multiplier split_to/split_from)]}, sorted."""
    rec = json.loads((ROOT / vintage / "contract.json").read_text())
    out: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for payload in _read_paged(ROOT / vintage / "raw" / "splits", True,
                               date.fromisoformat(rec["end"]), splits_request(vintage)):
        for row in payload.get("results", []):
            t, ed = row.get("ticker"), row.get("execution_date")
            sf, st = row.get("split_from"), row.get("split_to")
            if t and ed and sf and st:
                out[t].append((date.fromisoformat(ed), float(st) / float(sf)))
    return {t: sorted(v) for t, v in out.items()}


def split_factor(splits: list[tuple[date, float]], after: date, through: date) -> float:
    """Product of share multipliers for splits executing in (after, through]."""
    f = 1.0
    for ed, mult in splits:
        if after < ed <= through:
            f *= mult
    return f


# ── normalization ────────────────────────────────────────────────────────────

def _classification_by_month(vintage: str) -> dict[tuple[int, int], dict[str, dict]]:
    """Forward-held monthly labels: {(y, m): {ticker: {type, exchange}}}."""
    out: dict[tuple[int, int], dict[str, dict]] = {}
    ref_root = ROOT / vintage / "raw" / "reference"
    v3 = contract_version(vintage) == "v3"
    first_session: dict[tuple[int, int], date] = {}
    if v3:
        for d in frozen_sessions(vintage)[1]:
            first_session.setdefault((d.year, d.month), d)
        expected_names = {f"{y:04d}-{m:02d}": (y, m) for (y, m) in first_session}
        present_names = {p.name for p in ref_root.glob("*") if p.is_dir() and "." not in p.name}
        missing = sorted(set(expected_names) - present_names)
        extra = sorted(present_names - set(expected_names))
        if missing:
            raise RuntimeError(f"{len(missing)} monthly snapshot(s) missing, e.g. {missing[:3]} — "
                               "a month without its own snapshot would inherit stale labels")
        if extra:
            raise RuntimeError(f"unexpected reference directories {extra[:3]} — only canonical "
                               "YYYY-MM names inside the frozen range are allowed")
        for name, ym in sorted(expected_names.items()):
            out[ym] = _read_snapshot(ref_root / name, require_complete=True,
                                     expected_as_of=first_session[ym])
        return out
    for month_dir in sorted(ref_root.glob("*")):
        if not month_dir.is_dir() or (v3 and "." in month_dir.name):
            continue                  # *.untrusted-* directories are quarantined evidence
        year, month = (int(x) for x in month_dir.name.split("-"))
        if v3:
            if (year, month) not in first_session:
                raise RuntimeError(f"reference snapshot {month_dir.name} is outside the frozen range")
            out[(year, month)] = _read_snapshot(month_dir, require_complete=True,
                                                expected_as_of=first_session[(year, month)])
            continue
        # v2: original semantics (pages in lexical order, no completion marker).
        labels: dict[str, dict] = {}
        for page in sorted(month_dir.glob("page-*.json.gz")):
            for row in _read_raw(page).get("results", []):
                t = row.get("ticker")
                if t:
                    labels[t] = {
                        "type": row.get("type"),
                        "exchange": _EXCHANGE_MAP.get(row.get("primary_exchange", ""), ""),
                    }
        out[(year, month)] = labels
    return out


def _label_for(labels_by_month: dict, d: date) -> dict[str, dict]:
    """The most recent snapshot at or before d — never a later one (§3a)."""
    keys = [k for k in labels_by_month if (k[0], k[1]) <= (d.year, d.month)]
    if not keys:
        return {}
    return labels_by_month[max(keys)]


def build_membership(vintage: str) -> dict[date, dict]:
    """Per-session membership under every constraint Phase A can evaluate."""
    labels_by_month = _classification_by_month(vintage)
    overrides = resolved_overrides(vintage, labels_by_month)
    eligible_types = eligible_types_for(vintage)
    v3 = contract_version(vintage) == "v3"
    strict = v3
    warmup: set[date] = set()
    if v3:
        pre, _main = validated_sessions(vintage)
        warmup = set(pre)
    grouped_dir = ROOT / vintage / "raw" / "grouped"
    per_date: dict[date, dict] = {}
    # Prior bars per ticker as of D (strictly before D), for the §11 history rule.
    bars_seen: Counter = Counter()

    for path in sorted(grouped_dir.glob("*.json.gz")):
        d = date.fromisoformat(path.stem.replace(".json", ""))
        results = _read_raw(path).get("results", []) or []
        if d in warmup:
            for bar in results:
                if bar.get("T") and bar.get("c") is not None:
                    bars_seen[bar["T"]] += 1
            continue
        labels = _label_for(labels_by_month, d)
        if d in overrides:
            labels = {**labels, **overrides[d]}
        raw = _raw_bars(vintage, d) if v3 else {}

        traded, pre_class, eligible = [], [], []
        reasons: Counter = Counter()

        for bar in results:
            ticker = bar.get("T")
            close = bar.get("c")
            volume = bar.get("v")
            if not ticker or close is None or volume is None:
                reasons["no_price_or_volume"] += 1
                continue
            traded.append(ticker)
            if v3:
                # R9: the price and volume a screener could see ON D.
                rb = raw.get(ticker)
                if rb is None or rb.get("c") is None or rb.get("v") is None:
                    reasons["no_unadjusted_bar"] += 1
                    continue
                close, volume = rb["c"], rb["v"]

            # Observable constraints first — this set is the audit population,
            # deliberately drawn BEFORE classification (§3b).
            if strict:
                if close <= MIN_PRICE:
                    reasons["failed_price"] += 1
                    continue
                if volume <= MIN_SHARE_VOLUME:
                    reasons["failed_volume"] += 1
                    continue
            else:                    # v2 replays its original (inclusive) semantics
                if close < MIN_PRICE:
                    reasons["failed_price"] += 1
                    continue
                if volume < MIN_SHARE_VOLUME:
                    reasons["failed_volume"] += 1
                    continue
            pre_class.append(ticker)

            label = labels.get(ticker)
            if label is None or not label.get("type"):
                reasons["type_unknown"] += 1
                continue
            if not label.get("exchange"):
                reasons["exchange_unknown"] += 1
                continue
            if label["type"] not in eligible_types:
                # v2 vintages keep their original reason key so they replay.
                reasons["ineligible_type" if eligible_types is ELIGIBLE_TYPES
                        else "not_common_stock"] += 1
                continue
            if label["exchange"] not in ALLOWED_EXCHANGES:
                reasons["failed_exchange"] += 1
                continue
            if v3 and bars_seen[ticker] < MIN_PRIOR_BARS:
                reasons["insufficient_history"] += 1
                continue
            eligible.append(ticker)

        if v3:
            for bar in results:
                if bar.get("T") and bar.get("c") is not None:
                    bars_seen[bar["T"]] += 1

        per_date[d] = {
            "traded": traded,
            "pre_classification": pre_class,
            "eligible_pre_mcap": eligible,
            "exclusions": dict(reasons),
        }
    return per_date


# ── step 1b: transition resolution (§3a-v2, ruling R1) ────────────────────────
#
# Exchange and security type are EVENT-DRIVEN attributes: a venue transfer or a
# reclassification happens on a specific date, and a forward-held monthly label
# is wrong for up to a month after it (LNG, Feb 2024). For every ticker whose
# label differs between two consecutive monthly snapshots in a way that can
# change membership, the exact first session carrying the new label is found by
# bounded binary search on the per-ticker as-of endpoint, and applied with day
# resolution. Only tickers that pass the observable price/volume constraints on
# some session in the window are resolved: nothing else can enter membership,
# so resolving it would spend calls that cannot change the dataset.

def _eligible(label: dict | None) -> bool:
    """v3 eligibility; transition resolution only exists for v3 vintages."""
    return bool(label) and label.get("type") in ELIGIBLE_TYPES and label.get("exchange") in ALLOWED_EXCHANGES


def _membership_relevant_change(old: dict | None, new: dict | None) -> bool:
    """Type changes always; exchange changes only when they cross the eligible set.

    An absent label (a mid-month listing) differs from any present one.
    """
    old_type = (old or {}).get("type")
    new_type = (new or {}).get("type")
    if old_type != new_type:
        return True
    return _eligible(old) != _eligible(new)


def transition_candidates(vintage: str) -> list[dict]:
    """(ticker, month) pairs needing day resolution, with the sessions to search.

    A label only matters on a session where every NON-classification gate
    already passes: unadjusted close > $5, unadjusted volume > 500K and >= 200
    prior bars (R9 and §11). Those are the ticker's "relevant" sessions. A
    candidate is a membership-relevant label change (§3a-v2) that ALSO has
    relevant sessions in the month and where the old or the new label is
    eligible — when neither is, the ticker is excluded either way. The search
    domain is the relevant sessions only, so the resolved date is exact for
    membership while new listings (no history yet) and changes after a
    delisting's last qualifying session cost nothing.
    """
    labels_by_month = _classification_by_month(vintage)
    months = sorted(labels_by_month)
    grouped_dir = ROOT / vintage / "raw" / "grouped"
    warmup, sessions = validated_sessions(vintage)
    trailing = _trailing_labels(vintage)
    if trailing is None:
        raise RuntimeError("no trailing comparison snapshot — the final month's transitions "
                           "cannot be discovered; re-run `spine`")
    relevant_by_date: dict[date, set[str]] = {}
    bars_seen: Counter = Counter()
    main = set(sessions)
    for d in list(warmup) + list(sessions):
        adjusted = {r["T"] for r in (_read_raw(grouped_dir / f"{d}.json.gz").get("results", []) or [])
                    if r.get("T") and r.get("c") is not None and r.get("v") is not None}
        if d in main:
            relevant_by_date[d] = {
                t for t, r in _raw_bars(vintage, d).items()    # R9: unadjusted, observable on D
                if t in adjusted and r.get("c") is not None and r.get("v") is not None
                and r["c"] > MIN_PRICE and r["v"] > MIN_SHARE_VOLUME
                and bars_seen[t] >= MIN_PRIOR_BARS             # §11, strictly before D
            }
        for t in adjusted:                                      # counted after D is judged
            bars_seen[t] += 1
    out = []
    pairs = [(a, labels_by_month[b]) for a, b in zip(months, months[1:])]
    if months:
        pairs.append((months[-1], trailing))
    for a, new_l in pairs:
        window = [d for d in sessions if (d.year, d.month) == a]
        if not window:
            continue
        old_l = labels_by_month[a]
        tickers = set().union(*(relevant_by_date[d] for d in window))
        for t in sorted(tickers):
            old, new = old_l.get(t), new_l.get(t)
            if not _membership_relevant_change(old, new):
                continue
            if not (_eligible(old) or _eligible(new)):
                continue                # excluded under both labels: membership unchanged
            out.append({"ticker": t, "month": a, "old": old, "new": new,
                        "sessions": [d for d in window if t in relevant_by_date[d]]})
    return out


def _candidate_inputs_sha(c: dict) -> str:
    """Identity of a candidate's inputs: a result is valid only for these exact inputs."""
    return hashlib.sha256(json.dumps({
        "ticker": c["ticker"], "month": list(c["month"]), "old": c["old"], "new": c["new"],
        "sessions": [str(d) for d in c["sessions"]],
    }, sort_keys=True).encode()).hexdigest()


def _label_from_asof(payload: dict) -> dict | None:
    res = payload.get("results") or {}
    if payload.get("_not_found") or not res or not res.get("type"):
        return None
    return {"type": res.get("type"),
            "exchange": _EXCHANGE_MAP.get(res.get("primary_exchange", ""), "")}


def _same(a: dict | None, b: dict | None) -> bool:
    if not a or not b:
        return not a and not b
    return a.get("type") == b.get("type") and _eligible(a) == _eligible(b)


async def resolve_transitions(vintage: str) -> None:
    """Binary-search each candidate's first session carrying the new label.

    Every probe's raw response is persisted before use (replay determinism). A
    probe returning neither the old nor the new label makes the window
    AMBIGUOUS: recorded, and the ticker is treated as unknown (excluded and
    counted) for the unresolved sessions — never guessed.
    """
    if contract_version(vintage) != "v3":
        raise RuntimeError(f"vintage {vintage} is not a contract-v3 vintage; §3a-v2 does not apply")
    cands = transition_candidates(vintage)
    logger.info("transitions: %d candidate(s) to resolve", len(cands))
    async with httpx.AsyncClient() as client:
        for i, c in enumerate(cands, 1):
            t, (y, m) = c["ticker"], c["month"]
            out_path = _raw_path(vintage, "transitions", f"{y:04d}-{m:02d}", f"{t}.result.json.gz")
            inputs_sha = _candidate_inputs_sha(c)
            if out_path.exists() and _read_raw(out_path).get("inputs_sha256") == inputs_sha:
                continue                    # resolved for exactly these inputs
            # Absent, or resolved for different inputs (e.g. a snapshot completed
            # later): recompute. Cached probe responses are reused.
            window = c["sessions"]
            # Invariant: label(window[lo]) == old (the snapshot date itself, by
            # construction) and the next snapshot carries new. Find the first
            # index whose as-of label equals new.
            # `window` holds only the ticker's relevant sessions. Virtual bounds:
            # lo = -1 is the month's snapshot (old label), hi = len(window) the
            # next snapshot (new label). Probe the LAST relevant session first:
            # if it still carries the old label, no relevant session changes.
            lo, hi = -1, len(window)
            probes: list[dict] = []
            ambiguous = probe_failed = False
            first = True
            while hi - lo > 1:             # <= 1 + ceil(log2(len(window))) probes
                mid = len(window) - 1 if first else (lo + hi) // 2
                first = False
                d = window[mid]
                ppath = _raw_path(vintage, "transitions", f"{y:04d}-{m:02d}", f"{t}_{d}.json.gz")
                if ppath.exists():
                    payload = _read_raw(ppath)
                else:
                    payload = await _get(client, f"{BASE}/v3/reference/tickers/{t}",
                                         {"date": str(d)}, allow_404=True)
                    if payload.get("_failed"):
                        logger.error("  transition %s %s probe failed (%s)", t, d, payload.get("_reason"))
                        probe_failed = True
                        break
                    _write_raw_unchecked(ppath, payload)
                label = _label_from_asof(payload)
                probes.append({"date": str(d), "label": label})
                if _same(label, c["new"]):
                    hi = mid
                elif _same(label, c["old"]):
                    lo = mid
                else:
                    ambiguous = True
                    break
            if probe_failed:
                continue            # no result written: a resume retries this window
            result = {
                "ticker": t, "month": f"{y:04d}-{m:02d}", "old": c["old"], "new": c["new"],
                "effective": None if (ambiguous or hi >= len(window)) else str(window[hi]),
                "ambiguous_from": str(window[lo + 1]) if ambiguous else None,
                "probes": probes,
                "inputs_sha256": inputs_sha,
            }
            _write_raw_unchecked(out_path, result)
            if i % 100 == 0:
                logger.info("  transitions: %d/%d", i, len(cands))
    logger.info("transition resolution complete")


def resolved_overrides(vintage: str, labels_by_month: dict) -> dict[date, dict[str, dict | None]]:
    """{session: {ticker: label}} for sessions where §3a-v2 replaces the held label.

    From ``effective`` to the end of the month the ticker carries the NEW label.
    From ``ambiguous_from`` to the end of the month it carries no label (unknown,
    excluded and counted). Before either, the forward-held label stands.
    """
    root = ROOT / vintage / "raw" / "transitions"
    out: dict[date, dict[str, dict | None]] = defaultdict(dict)
    if not root.exists() or contract_version(vintage) != "v3":
        return {}
    _warmup, sessions = validated_sessions(vintage)
    expected = _expected_transition_inputs(vintage)
    for path in sorted(root.glob("*/*.result.json.gz")):
        key = (path.parent.name, path.name.replace(".result.json.gz", ""))
        if key not in expected:
            continue                    # orphan result: never applied
        r = _read_raw(path)
        if r.get("inputs_sha256") != expected[key]:
            continue                    # stale result (other inputs): never applied
        y, m = (int(x) for x in r["month"].split("-"))
        start = r.get("effective") or r.get("ambiguous_from")
        if not start:
            continue
        start_d = date.fromisoformat(start)
        label = r["new"] if r.get("effective") else None
        for d in sessions:
            if (d.year, d.month) == (y, m) and d >= start_d:
                out[d][r["ticker"]] = label
    return dict(out)


def transition_status(vintage: str) -> dict:
    """Expected §3a-v2 candidates vs completed results. Fail-closed prerequisite.

    A failed probe leaves no result, and membership would silently fall back to
    the monthly label for that window — a plausible, wrong universe. Audit and
    report therefore refuse a v3 vintage until every candidate is resolved.
    """
    expected = _expected_transition_inputs(vintage)
    root = ROOT / vintage / "raw" / "transitions"
    done, stale = set(), []
    if root.exists():
        for p in root.glob("*/*.result.json.gz"):
            key = (p.parent.name, p.name.replace(".result.json.gz", ""))
            if key in expected and _read_raw(p).get("inputs_sha256") != expected[key]:
                stale.append(key)           # resolved for other inputs: not done
                continue
            done.add(key)
    missing = sorted(set(expected) - done)
    extra = sorted(done - set(expected))
    return {"expected": len(expected), "resolved": len(set(expected) & done),
            "missing": missing, "extra": extra, "stale": sorted(stale),
            "complete": not missing and not extra}


_EXPECTED_CACHE: dict[tuple, dict] = {}


def _spine_fingerprint(vintage: str) -> str:
    """Cheap identity of the spine inputs (paths, sizes, mtimes)."""
    raw = ROOT / vintage / "raw"
    h = hashlib.sha256()
    for sub_ in ("grouped", "reference", "reference_trailing"):
        for p in sorted((raw / sub_).rglob("*")) if (raw / sub_).exists() else []:
            if p.is_file():
                st = p.stat()
                h.update(f"{p.relative_to(raw)}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _expected_transition_inputs(vintage: str) -> dict[tuple[str, str], str]:
    """{(month, ticker): inputs sha} — cached only while the spine inputs are unchanged."""
    key = (str(ROOT), vintage, _spine_fingerprint(vintage))
    if key not in _EXPECTED_CACHE:
        _EXPECTED_CACHE.clear()
        _EXPECTED_CACHE[key] = {(f"{c['month'][0]:04d}-{c['month'][1]:02d}", c["ticker"]):
                                _candidate_inputs_sha(c) for c in transition_candidates(vintage)}
    return _EXPECTED_CACHE[key]


def _expected_transition_keys(vintage: str) -> set[tuple[str, str]]:
    return set(_expected_transition_inputs(vintage))


def overrides_fingerprint(vintage: str) -> str:
    """Hash of every transition result: the labels the audit sample was drawn under."""
    root = ROOT / vintage / "raw" / "transitions"
    h = hashlib.sha256()
    if root.exists():
        expected = _expected_transition_inputs(vintage) if contract_version(vintage) == "v3" else None
        for p in sorted(root.glob("*/*.result.json.gz")):
            key = (p.parent.name, p.name.replace(".result.json.gz", ""))
            if expected is not None and (key not in expected
                                         or _read_raw(p).get("inputs_sha256") != expected[key]):
                continue
            h.update(str(p.relative_to(root)).encode())
            h.update(json.dumps(_read_raw(p), sort_keys=True).encode())
    return h.hexdigest()


def require_transitions_complete(vintage: str) -> None:
    if contract_version(vintage) != "v3":
        return
    st = transition_status(vintage)
    if not st["complete"]:
        raise RuntimeError(
            f"§3a-v2 incomplete: {len(st['missing'])} of {st['expected']} transitions unresolved "
            f"(e.g. {st['missing'][:3]}; {len(st['stale'])} stale), {len(st['extra'])} orphan result(s) "
            f"(e.g. {st['extra'][:3]}) — run `transitions` / remove orphans before reporting")


def resolved_label(labels_by_month: dict, overrides: dict, ticker: str, d: date) -> dict | None:
    if d in overrides and ticker in overrides[d]:
        return overrides[d][ticker]
    return _label_for(labels_by_month, d).get(ticker)


# ── step 2: classification drift audit (§3b) ─────────────────────────────────

def _bucket(label: dict | None, eligible_types: frozenset[str] = ELIGIBLE_TYPES) -> str:
    if label is None or not label.get("type"):
        return "unknown"
    return "common_stock" if label["type"] in eligible_types else "etf_fund_other"


def audit_sample(vintage: str, membership: dict[date, dict]) -> dict[tuple[int, int], list]:
    """Deterministic stratified sample from the PRE-classification population.

    Canonical (ET date, ticker) ordering before any seeded draw, per the
    sampling discipline: a seed over an unordered population is not
    reproducible.
    """
    labels_by_month = _classification_by_month(vintage)
    overrides = resolved_overrides(vintage, labels_by_month)
    by_month: dict[tuple[int, int], list[tuple[date, str]]] = defaultdict(list)
    for d, rec in membership.items():
        for t in rec["pre_classification"]:
            by_month[(d.year, d.month)].append((d, t))

    sampled: dict[tuple[int, int], list] = {}
    for month, pairs in sorted(by_month.items()):
        strata: dict[str, list] = {b: [] for b in AUDIT_BUCKETS}
        for pair in sorted(pairs):                      # canonical order
            strata[_bucket(resolved_label(labels_by_month, overrides, pair[1], pair[0]),
                           eligible_types_for(vintage))].append(pair)

        per_bucket = AUDIT_PAIRS_PER_MONTH // len(AUDIT_BUCKETS)
        chosen: list = []
        shortfall = 0
        for b in AUDIT_BUCKETS:
            pool = strata[b]
            want = per_bucket
            if len(pool) <= want:
                chosen.extend((b, *p) for p in pool)
                shortfall += want - len(pool)
            else:
                rng = random.Random(f"{AUDIT_SEED}:{SAMPLER_VERSION}:{month}:{b}")
                chosen.extend((b, *p) for p in rng.sample(pool, want))
        # Redistribute deterministically into the largest remaining bucket.
        if shortfall:
            for b in AUDIT_BUCKETS:
                pool = [p for p in strata[b] if (b, *p) not in set(chosen)]
                if not pool:
                    continue
                take = min(shortfall, len(pool))
                rng = random.Random(f"{AUDIT_SEED}:{SAMPLER_VERSION}:{month}:{b}:fill")
                chosen.extend((b, *p) for p in rng.sample(pool, take))
                shortfall -= take
                if not shortfall:
                    break
        sampled[month] = sorted(chosen, key=lambda x: (x[1], x[2]))
    return sampled


async def run_audit(vintage: str) -> None:
    require_transitions_complete(vintage)
    membership = build_membership(vintage)
    sample = audit_sample(vintage, membership)
    if contract_version(vintage) == "v3":
        # Bind the sample to the labels it was drawn under. If transitions are
        # ever re-resolved, the plan no longer matches and the report refuses
        # it, instead of reading audit files bucketed under stale labels.
        plan_path = ROOT / vintage / "audit_plan.json"
        plan = {"sampler_version": SAMPLER_VERSION,
                "overrides_sha256": overrides_fingerprint(vintage),
                "pairs": sorted(f"{m[0]:04d}-{m[1]:02d}/{t}_{d}"
                                for m, pairs in sample.items() for _b, d, t in pairs)}
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise RuntimeError("audit plan changed since the last audit run — start a new "
                               "vintage rather than mixing samples")
        plan_path.write_text(json.dumps(plan))
    total = sum(len(v) for v in sample.values())
    logger.info("audit: %d pairs across %d months", total, len(sample))

    async with httpx.AsyncClient() as client:
        done = 0
        for month, pairs in sorted(sample.items()):
            for bucket, d, ticker in pairs:
                path = _raw_path(vintage, "audit", f"{month[0]:04d}-{month[1]:02d}", f"{ticker}_{d}.json.gz")
                if path.exists():
                    done += 1
                    continue
                payload = await _get(
                    client, f"{BASE}/v3/reference/tickers/{ticker}", {"date": str(d)},
                    allow_404=True,
                )
                if payload.get("_failed"):
                    # Unverifiable-by-failure is NOT the same as the vendor
                    # having no record (`_not_found`). Conflating them would let
                    # an outage masquerade as evidence about the vendor.
                    logger.error("  audit %s %s unrecoverable (%s)",
                                 ticker, d, payload.get("_reason"))
                    continue
                payload["_audit"] = {"bucket": bucket, "date": str(d), "ticker": ticker}
                _write_raw(path, payload)
                done += 1
                if done % 100 == 0:
                    logger.info("  audit: %d/%d", done, total)
    logger.info("audit fetch complete: %d pairs", total)


# ── Phase B: market-cap estimate (§3c) and threshold audit (§3d) ─────────────

MIN_MCAP = 300_000_000          # live screener: marketCapMoreThan=300M (strict)
MCAP_BAND = 0.20
MCAP_SAMPLER_VERSION = "phase-b/1"
MCAP_AUDIT_SEED = 20261006
BAND_PER_MONTH = 50
SENTINEL_PER_MONTH = 25


def _quarter(d: date) -> tuple[int, int]:
    return (d.year, (d.month - 1) // 3 + 1)


def quarter_snapshot_dates(vintage: str) -> dict[tuple[int, int], date]:
    """First membership session of each quarter: the shares-outstanding as-of date."""
    out: dict[tuple[int, int], date] = {}
    for d in frozen_sessions(vintage)[1]:
        out.setdefault(_quarter(d), d)
    return out


def mcap_candidates(vintage: str) -> list[tuple[str, tuple[int, int]]]:
    """(ticker, quarter) pairs where the ticker passed every non-mcap gate on >= 1 session.

    No other quarter's market cap can affect membership, so no other lookup is made.
    """
    require_transitions_complete(vintage)
    pairs = set()
    for d, rec in build_membership(vintage).items():
        q = _quarter(d)
        pairs.update((t, q) for t in rec["eligible_pre_mcap"])
    return sorted(pairs)


def _details_path(vintage: str, t: str, q: tuple[int, int]) -> Path:
    return _raw_path(vintage, "details", f"{q[0]:04d}-Q{q[1]}", f"{t}.json.gz")


async def fetch_mcap_details(vintage: str) -> None:
    snaps = quarter_snapshot_dates(vintage)
    pairs = mcap_candidates(vintage)
    logger.info("phase B: %d (ticker, quarter) shares lookups", len(pairs))
    async with httpx.AsyncClient() as client:
        done = 0
        for i, (t, q) in enumerate(pairs, 1):
            path = _details_path(vintage, t, q)
            if path.exists():
                if _details_valid(vintage, t, q, snaps):
                    continue
                aside = path.with_name(f"{path.name}.untrusted-{_utc_stamp()}")
                path.rename(aside)          # never deleted; refetched below
                logger.warning("  details %s %s: unproven file moved aside", t, q)
            payload = await _get(client, f"{BASE}/v3/reference/tickers/{t}",
                                 {"date": str(snaps[q])}, allow_404=True)
            if payload.get("_failed"):
                logger.error("  details %s %s unrecoverable (%s)", t, q, payload.get("_reason"))
                continue
            payload["_request"] = {"ticker": t, "date": str(snaps[q])}
            _write_raw_unchecked(path, payload)
            done += 1
            if done % 1000 == 0:
                logger.info("  details: %d fetched (%d/%d)", done, i, len(pairs))


def _details_valid(vintage: str, t: str, q: tuple[int, int], snaps: dict) -> bool:
    """The file at the expected path answers exactly (ticker, quarter snapshot date)."""
    try:
        payload = _read_raw(_details_path(vintage, t, q))
    except Exception:  # noqa: BLE001
        return False
    if payload.get("_request") != {"ticker": t, "date": str(snaps[q])}:
        return False
    res = payload.get("results")
    if payload.get("_not_found") is True:
        return res is None                   # a 404 must not also carry a body
    return isinstance(res, dict) and res.get("ticker") == t


def phase_b_status(vintage: str) -> dict:
    expected = {(t, q) for t, q in mcap_candidates(vintage)}
    snaps = quarter_snapshot_dates(vintage)
    root = ROOT / vintage / "raw" / "details"
    present, invalid = set(), []
    if root.exists():
        for p in root.glob("*/*.json.gz"):
            y, qn = p.parent.name.split("-Q")
            key = (p.name.replace(".json.gz", ""), (int(y), int(qn)))
            if key in expected and not _details_valid(vintage, key[0], key[1], snaps):
                invalid.append(key)
                continue
            present.add(key)
    missing, extra = sorted(expected - present), sorted(present - expected)
    return {"expected": len(expected), "present": len(expected & present),
            "missing": missing, "extra": extra, "invalid": sorted(invalid),
            "complete": not missing and not extra}


def mcap_estimates(vintage: str, membership: dict | None = None) -> dict[tuple[date, str], float | None]:
    """{(D, ticker): estimated market cap or None} for every pre-mcap-eligible pair.

    estimate = weighted_shares_outstanding as of the quarter's first session
             x split multiplier for splits executing after that date through D
             x D's UNADJUSTED close (R9).
    """
    st = phase_b_status(vintage)
    if not st["complete"]:
        raise RuntimeError(f"phase B incomplete: {len(st['missing'])} missing, "
                           f"{len(st['extra'])} extra details — run `mcap`")
    membership = membership or build_membership(vintage)
    snaps = quarter_snapshot_dates(vintage)
    splits = splits_by_ticker(vintage)
    shares: dict[tuple[str, tuple[int, int]], float | None] = {}
    out: dict[tuple[date, str], float | None] = {}
    for d in sorted(membership):
        q = _quarter(d)
        raw = _raw_bars(vintage, d)
        for t in membership[d]["eligible_pre_mcap"]:
            if (t, q) not in shares:
                payload = _read_raw(_details_path(vintage, t, q))
                res = payload.get("results") or {}
                wso = res.get("weighted_shares_outstanding")
                shares[(t, q)] = float(wso) if wso and not payload.get("_not_found") else None
            wso = shares[(t, q)]
            close = (raw.get(t) or {}).get("c")
            if wso is None or close is None:
                out[(d, t)] = None
                continue
            out[(d, t)] = wso * split_factor(splits.get(t, []), snaps[q], d) * float(close)
    return out


def build_membership_with_mcap(vintage: str) -> dict[date, dict]:
    """Phase A membership plus the market-cap gate. Only when Phase B is complete."""
    membership = build_membership(vintage)
    est = mcap_estimates(vintage, membership)
    for d, rec in membership.items():
        eligible, reasons = [], Counter()
        for t in rec["eligible_pre_mcap"]:
            v = est[(d, t)]
            if v is None:
                reasons["mcap_unknown"] += 1
            elif v > MIN_MCAP:
                eligible.append(t)
            else:
                reasons["failed_mcap"] += 1
        rec["eligible"] = eligible
        rec["exclusions"] = {**rec["exclusions"], **reasons}
    return membership


def mcap_audit_sample(vintage: str) -> dict[str, list[tuple[str, date, str, float]]]:
    """§3d deterministic sample per month: [(part, D, ticker, estimate)].

    Canonical (D, ticker) order before every seeded draw. Band: up to 50 pairs
    with |est - 300M| <= 20%. Sentinel: 25 outside the band, 12 below / 13
    above, the 13th going ABOVE on even month index (0-based from the range
    start) and BELOW on odd. An underfilled stratum is audited in full and its
    unused allocation is drawn (same seed) from the other stratum.
    """
    est = mcap_estimates(vintage)
    by_month: dict[str, list[tuple[date, str, float]]] = defaultdict(list)
    for (d, t), v in est.items():
        if v is not None:
            by_month[f"{d.year:04d}-{d.month:02d}"].append((d, t, v))
    lo, hi = MIN_MCAP * (1 - MCAP_BAND), MIN_MCAP * (1 + MCAP_BAND)
    out: dict[str, list] = {}
    first = frozen_sessions(vintage)[1][0]
    for month in sorted(by_month):
        y, mo = (int(x) for x in month.split("-"))
        idx = (y - first.year) * 12 + (mo - first.month)   # 0-based from the range start (§3d)
        pairs = sorted(by_month[month], key=lambda x: (x[0], x[1]))
        band = [p for p in pairs if lo <= p[2] <= hi]
        below = [p for p in pairs if p[2] < lo]
        above = [p for p in pairs if p[2] > hi]

        def draw(pool, k, tag):
            if len(pool) <= k:
                return list(pool)
            rng = random.Random(f"{MCAP_AUDIT_SEED}:{MCAP_SAMPLER_VERSION}:{month}:{tag}")
            return sorted(rng.sample(pool, k), key=lambda x: (x[0], x[1]))

        chosen = [("band", *p) for p in draw(band, BAND_PER_MONTH, "band")]
        want_above = 13 if idx % 2 == 0 else 12
        want_below = SENTINEL_PER_MONTH - want_above
        sel_above = draw(above, want_above, "above")
        sel_below = draw(below, want_below, "below")
        # §3d: an underfilled stratum is audited in full and its unused allocation
        # goes to the other stratum as "the next pairs in canonical order" — the
        # first not-yet-chosen pairs of that stratum, not another seeded draw.
        spare = SENTINEL_PER_MONTH - len(sel_above) - len(sel_below)
        for pool, sel in ((above, sel_above), (below, sel_below)):
            if spare <= 0:
                break
            taken = set(sel)
            extra = [p for p in pool if p not in taken][:spare]
            sel.extend(extra)
            sel.sort(key=lambda x: (x[0], x[1]))
            spare -= len(extra)
        chosen += [("sentinel_above", *p) for p in sel_above]
        chosen += [("sentinel_below", *p) for p in sel_below]
        out[month] = chosen
    return out


def _mcap_estimates_fingerprint(sample: dict) -> str:
    h = hashlib.sha256()
    for month in sorted(sample):
        for part, d, t, v in sample[month]:
            h.update(f"{month}|{part}|{d}|{t}|{v:.2f}".encode())
    return h.hexdigest()


def expected_mcap_plan(vintage: str, sample: dict | None = None) -> dict:
    sample = sample if sample is not None else mcap_audit_sample(vintage)
    return {"sampler_version": MCAP_SAMPLER_VERSION, "seed": MCAP_AUDIT_SEED,
            "sample_sha256": _mcap_estimates_fingerprint(sample),
            "pairs": sorted(f"{m}/{part}/{t}_{d}" for m, rows in sample.items() for part, d, t, _ in rows)}


async def run_mcap_audit(vintage: str) -> None:
    sample = mcap_audit_sample(vintage)
    plan = expected_mcap_plan(vintage, sample)
    plan_path = ROOT / vintage / "mcap_audit_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("mcap audit plan changed since the last run — start a new vintage")
    plan_path.write_text(json.dumps(plan))
    total = sum(len(v) for v in sample.values())
    logger.info("mcap audit: %d pairs across %d months", total, len(sample))
    async with httpx.AsyncClient() as client:
        for month, rows in sorted(sample.items()):
            for part, d, t, v in rows:
                path = _raw_path(vintage, "mcap_audit", month, part, f"{t}_{d}.json.gz")
                if path.exists():
                    continue
                payload = await _get(client, f"{BASE}/v3/reference/tickers/{t}",
                                     {"date": str(d)}, allow_404=True)
                if payload.get("_failed"):
                    logger.error("  mcap audit %s %s unrecoverable (%s)", t, d, payload.get("_reason"))
                    continue
                payload["_audit"] = {"part": part, "date": str(d), "ticker": t, "estimate": v}
                _write_raw_unchecked(path, payload)


_GOVERNED_PATHS = ("scripts/pit_universe_phase_a.py", "scripts/pit_universe_report.py")


def require_clean_code() -> str:
    """The commit a v3 step runs must be exactly the code it executes.

    A manifest records HEAD; running with uncommitted edits to the governed
    scripts would attribute their output to a commit that never contained them.
    """
    import subprocess
    repo = Path(__file__).resolve().parent.parent
    for exe in ("git", "/opt/homebrew/bin/git"):
        try:
            st = subprocess.run([exe, "-C", str(repo), "status", "--porcelain", "--", *_GOVERNED_PATHS],
                                capture_output=True, text=True, timeout=30)
            head = subprocess.run([exe, "-C", str(repo), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        if st.returncode == 0 and head.returncode == 0:
            if st.stdout.strip():
                raise SystemExit(f"refusing to run: uncommitted changes to governed scripts:\n{st.stdout}")
            return head.stdout.strip()
    raise SystemExit("refusing to run: git unavailable, cannot establish code identity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["spine", "transitions", "audit", "mcap", "mcap-audit",
                                     "report", "verify", "package", "divergence-fetch"])
    ap.add_argument("--manifest", help="verify against this manifest instead of the vintage's own")
    ap.add_argument("--years", type=float, default=None)
    ap.add_argument("--start", type=date.fromisoformat, default=None,
                    help=f"first ET session (contract v3 default {DEFAULT_START})")
    ap.add_argument("--vintage", default=None, help="ET date tag; defaults to today ET")
    args = ap.parse_args()

    vintage = args.vintage or str(_today_et())
    logger.info("vintage %s  step %s", vintage, args.step)
    if args.step != "verify" and (contract_version(vintage) == "v3" or args.step == "spine"):
        logger.info("code identity: %s", require_clean_code())

    if args.step in ("mcap", "mcap-audit"):
        ledger = _open_ledger(vintage, "B")
        try:
            asyncio.run(fetch_mcap_details(vintage) if args.step == "mcap" else run_mcap_audit(vintage))
        except BudgetExceeded as e:
            logger.error("ABORTED ON BUDGET: %s", e)
            raise SystemExit(2) from e
        finally:
            ledger.close()
            logger.info("phase B ledger closed: %d calls, %d durable failure(s)",
                        ledger.calls, len(ledger.failures))
        return
    if args.step in ("spine", "transitions", "audit"):
        ledger = _open_ledger(vintage)
        try:
            if args.step == "spine":
                start = args.start or (None if args.years is not None else DEFAULT_START)
                asyncio.run(fetch_spine(vintage, args.years, start))
            elif args.step == "transitions":
                asyncio.run(resolve_transitions(vintage))
            else:
                asyncio.run(run_audit(vintage))
        except BudgetExceeded as e:
            logger.error("ABORTED ON BUDGET: %s", e)
            raise SystemExit(2) from e
        finally:
            ledger.close()
            logger.info(
                "ledger closed: %d calls spent, %d durable failure(s)",
                ledger.calls, len(ledger.failures),
            )
            if ledger.failures:
                logger.error(
                    "run has %d unrecoverable request(s) — the vintage is INCOMPLETE "
                    "and must not be reported until a re-run fills them",
                    len(ledger.failures),
                )
    elif args.step == "report":
        from scripts.pit_universe_report import write_report  # noqa: PLC0415

        write_report(vintage)
    elif args.step == "divergence-fetch":
        _fetch_live_snapshot(vintage)
    elif args.step == "verify":
        from scripts.pit_universe_report import verify  # noqa: PLC0415

        verify(vintage, Path(args.manifest) if args.manifest else None)
    else:
        from scripts.pit_universe_report import package  # noqa: PLC0415

        package(vintage)


if __name__ == "__main__":
    main()
