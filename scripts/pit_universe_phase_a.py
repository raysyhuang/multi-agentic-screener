"""Phase A of the PIT universe build — the membership spine.

Contract: outputs/research/PIT_UNIVERSE_CONTRACT.md (v3: frozen #77 + #79, rulings
R1-R8 of 2026-10-06 in §12).

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
        "rulings": "R1-R8, 2026-10-06 (contract §12)",
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
    snap_dir = ROOT / vintage / "raw" / "reference_trailing" / str(trailing_snapshot_date(vintage))
    if not (snap_dir / SNAPSHOT_MARKER).exists():
        return None                       # absent or incomplete: candidate generation refuses
    return _read_snapshot(snap_dir, require_complete=True)
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

    def __init__(self, path: Path) -> None:
        self.path = path
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
                    if json.loads(line).get("event") == "request":
                        n += 1
                except json.JSONDecodeError:
                    logger.warning("unparseable ledger line skipped")
        return n

    def would_exceed(self) -> bool:
        return self.calls >= PHASE_A_CALL_CEILING

    def record(self, url: str, params: dict, status: int | str, attempt: int) -> None:
        self.calls += 1
        self._fh.write(json.dumps({
            "event": "request", "n": self.calls,
            "endpoint": url.replace(BASE, ""),
            "params_sha256": hashlib.sha256(
                json.dumps(params, sort_keys=True).encode()
            ).hexdigest()[:16],
            "status": status, "attempt": attempt,
        }) + "\n")

    def record_failure(self, url: str, params: dict, reason: str, attempts: int) -> None:
        rec = {
            "event": "failure", "endpoint": url.replace(BASE, ""),
            "params_sha256": hashlib.sha256(
                json.dumps(params, sort_keys=True).encode()
            ).hexdigest()[:16],
            "reason": reason, "attempts": attempts,
        }
        self.failures.append(rec)
        self._fh.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        self._fh.close()


# Bound to the active run by `_open_ledger`; `_get` refuses to issue without it,
# so an unmetered request path cannot be introduced by accident.
_LEDGER: RequestLedger | None = None


def _open_ledger(vintage: str) -> RequestLedger:
    global _LEDGER
    _LEDGER = RequestLedger(ROOT / vintage / "request_ledger.jsonl")
    logger.info(
        "ledger: %d calls already spent on this vintage, ceiling %d",
        _LEDGER.calls, PHASE_A_CALL_CEILING,
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
                f"Phase A ceiling {PHASE_A_CALL_CEILING} reached ({_LEDGER.calls} "
                f"spent) on attempt {attempt}. Raising it is a contract change, "
                "not a flag."
            )
        try:
            resp = await client.get(url, params=params, headers=headers, timeout=60)
        except httpx.HTTPError as e:
            _LEDGER.record(url, params, f"network:{type(e).__name__}", attempt)
            if attempt == attempts:
                _LEDGER.record_failure(url, params, f"network:{type(e).__name__}", attempt)
                return {"results": None, "_failed": True, "_reason": type(e).__name__}
            logger.warning("network error (%s), retry %d", type(e).__name__, attempt)
            await asyncio.sleep(2 ** attempt)
            continue

        _LEDGER.record(url, params, resp.status_code, attempt)

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
    """True once session d has closed (16:15 ET margin), judged in ET, never local time."""
    from datetime import datetime, time
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("America/New_York"))
    return d < now.date() or (d == now.date() and now.time() >= time(16, 15))


SNAPSHOT_MARKER = "_complete.json"


async def _fetch_snapshot(client, vintage: str, sub: tuple[str, ...], as_of: date) -> bool:
    """Page through /v3/reference/tickers as of a date; mark complete only at the last page.

    Resumable: existing pages are re-read (their next_url drives the walk), so a
    run interrupted mid-pagination continues where it stopped and writes the
    completion marker only when a page with no next_url is reached.
    """
    marker = _raw_path(vintage, *sub, SNAPSHOT_MARKER)
    if marker.exists():
        return True
    page, cursor, hashes = 1, None, []
    while True:
        path = _raw_path(vintage, *sub, f"page-{page}.json.gz")
        if path.exists():
            payload = _read_raw(path)
        else:
            params = {"market": "stocks", "date": str(as_of), "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            payload = await _get(client, f"{BASE}/v3/reference/tickers", params)
            if payload.get("_failed"):
                logger.error("  snapshot %s page %d unrecoverable (%s) — left INCOMPLETE",
                             "/".join(sub), page, payload.get("_reason"))
                return False
            _write_raw(path, payload)
        hashes.append(_sha256_file(path))
        nxt = payload.get("next_url")
        if not nxt:
            break
        cursor = nxt.split("cursor=")[-1]
        page += 1
    marker.write_text(json.dumps({"as_of": str(as_of), "pages": page, "page_sha256": hashes}))
    logger.info("  snapshot %s: %d page(s), complete", "/".join(sub), page)
    return True


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_snapshot(snap_dir: Path, require_complete: bool) -> dict[str, dict]:
    """Labels from one snapshot directory; v3 refuses an unmarked or altered one."""
    marker = snap_dir / SNAPSHOT_MARKER
    pages = sorted(snap_dir.glob("page-*.json.gz"), key=lambda p: int(p.name.split("-")[1].split(".")[0]))
    if require_complete:
        if not marker.exists():
            raise RuntimeError(f"snapshot {snap_dir.name} has no completion marker — pagination "
                               "incomplete; re-run `spine`")
        m = json.loads(marker.read_text())
        names = [p.name for p in pages]
        if names != [f"page-{i}.json.gz" for i in range(1, m["pages"] + 1)]:
            raise RuntimeError(f"snapshot {snap_dir.name}: pages {names[:3]}... do not match marker")
        if [_sha256_file(p) for p in pages] != m["page_sha256"]:
            raise RuntimeError(f"snapshot {snap_dir.name}: page bytes changed since completion")
    labels: dict[str, dict] = {}
    for page in pages:
        for row in _read_raw(page).get("results", []):
            t = row.get("ticker")
            if t:
                labels[t] = {"type": row.get("type"),
                             "exchange": _EXCHANGE_MAP.get(row.get("primary_exchange", ""), "")}
    return labels


# ── normalization ────────────────────────────────────────────────────────────

def _classification_by_month(vintage: str) -> dict[tuple[int, int], dict[str, dict]]:
    """Forward-held monthly labels: {(y, m): {ticker: {type, exchange}}}."""
    out: dict[tuple[int, int], dict[str, dict]] = {}
    ref_root = ROOT / vintage / "raw" / "reference"
    v3 = contract_version(vintage) == "v3"
    for month_dir in sorted(ref_root.glob("*")):
        if not month_dir.is_dir():
            continue
        year, month = (int(x) for x in month_dir.name.split("-"))
        if v3:
            out[(year, month)] = _read_snapshot(month_dir, require_complete=True)
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
    """(ticker, month) pairs needing day resolution, with the sessions to search."""
    labels_by_month = _classification_by_month(vintage)
    months = sorted(labels_by_month)
    grouped_dir = ROOT / vintage / "raw" / "grouped"
    _warmup, sessions = validated_sessions(vintage)
    trailing = _trailing_labels(vintage)
    if trailing is None:
        raise RuntimeError("no trailing comparison snapshot — the final month's transitions "
                           "cannot be discovered; re-run `spine`")
    pre_by_date: dict[date, set[str]] = {}
    for d in sessions:
        rows = _read_raw(grouped_dir / f"{d}.json.gz").get("results", []) or []
        pre_by_date[d] = {
            r["T"] for r in rows
            if r.get("T") and r.get("c") is not None and r.get("v") is not None
            and r["c"] > MIN_PRICE and r["v"] > MIN_SHARE_VOLUME
        }
    out = []
    pairs = [(a, labels_by_month[b]) for a, b in zip(months, months[1:])]
    if months:
        pairs.append((months[-1], trailing))
    for a, new_l in pairs:
        window = [d for d in sessions if (d.year, d.month) == a]
        if not window:
            continue
        relevant = set().union(*(pre_by_date[d] for d in window))
        old_l = labels_by_month[a]
        for t in sorted(relevant):
            if _membership_relevant_change(old_l.get(t), new_l.get(t)):
                out.append({"ticker": t, "month": a, "old": old_l.get(t), "new": new_l.get(t),
                            "sessions": window})
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
            lo, hi = 0, len(window)        # hi == len(window): new from the next snapshot on
            probes: list[dict] = []
            ambiguous = probe_failed = False
            while hi - lo > 1:             # <= ceil(log2(len(window))) probes
                mid = (lo + hi) // 2
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["spine", "transitions", "audit", "report", "verify",
                                     "package", "divergence-fetch"])
    ap.add_argument("--manifest", help="verify against this manifest instead of the vintage's own")
    ap.add_argument("--years", type=float, default=None)
    ap.add_argument("--start", type=date.fromisoformat, default=None,
                    help=f"first ET session (contract v3 default {DEFAULT_START})")
    ap.add_argument("--vintage", default=None, help="ET date tag; defaults to today ET")
    args = ap.parse_args()

    vintage = args.vintage or str(_today_et())
    logger.info("vintage %s  step %s", vintage, args.step)

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
