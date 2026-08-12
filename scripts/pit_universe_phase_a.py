"""Phase A of the PIT universe build — the membership spine.

Contract: outputs/research/PIT_UNIVERSE_CONTRACT.md (frozen, #77 + #79).

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
    python scripts/pit_universe_phase_a.py spine   [--years 3] [--vintage ET-DATE]
    python scripts/pit_universe_phase_a.py audit   [--vintage ET-DATE]
    python scripts/pit_universe_phase_a.py report  [--vintage ET-DATE]
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
_EXCHANGE_MAP = {
    "XNYS": "NYSE", "XNAS": "NASDAQ", "XASE": "AMEX",
    "ARCX": "NYSE", "BATS": "NASDAQ",
    "XNGS": "NASDAQ", "XNCM": "NASDAQ", "XNMS": "NASDAQ",
}

# §3b audit. Sampler version is recorded in the manifest; changing any of these
# constants is a version bump, never a silent edit.
SAMPLER_VERSION = "phase-a/1"
AUDIT_SEED = 20260812
AUDIT_PAIRS_PER_MONTH = 200
AUDIT_BUCKETS = ("common_stock", "etf_fund_other", "unknown")

# Conservative pacing. The contract requires one request in flight per endpoint
# family and no burst parallelism (§A.3).
REQUEST_DELAY_S = 0.12


# ── raw layer ────────────────────────────────────────────────────────────────

def _raw_path(vintage: str, *parts: str) -> Path:
    return ROOT / vintage / "raw" / Path(*parts)


def _write_raw(path: Path, payload: dict) -> str:
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


async def _get(client: httpx.AsyncClient, url: str, params: dict) -> dict:
    """One GET with backoff. Never parallel — see §A.3."""
    settings = get_settings()
    # Bearer header, not a query parameter. A URL carrying the key ends up in
    # client logs, proxy logs, and exception messages; a header does not.
    headers = {"Authorization": f"Bearer {settings.polygon_api_key}"}
    for attempt in range(1, 6):
        try:
            resp = await client.get(url, params=params, headers=headers, timeout=60)
        except httpx.HTTPError as e:
            if attempt == 5:
                raise
            logger.warning("network error (%s), retry %d", e, attempt)
            await asyncio.sleep(2 ** attempt)
            continue
        if resp.status_code == 429:
            wait = min(60, 2 ** attempt)
            logger.warning("429 rate limited, sleeping %ss", wait)
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        await asyncio.sleep(REQUEST_DELAY_S)
        return resp.json()
    raise RuntimeError(f"exhausted retries for {url}")


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


def et_sessions(years: float) -> list[date]:
    """Actual NYSE sessions, so holidays cost no calls.

    Ends at the last COMPLETED session: today's bars do not exist until the
    session closes, and a partial file frozen into a vintage would be worse
    than a missing one.
    """
    import pandas_market_calendars as mcal

    end = _today_et() - timedelta(days=1)
    start = end - timedelta(days=int(365.25 * years))
    sched = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    return [d.date() for d in sched.index]


# ── step 1: spine ────────────────────────────────────────────────────────────

async def fetch_spine(vintage: str, years: float) -> None:
    sessions = et_sessions(years)
    months = sorted({(d.year, d.month) for d in sessions})
    logger.info(
        "spine: %d ET sessions, %d monthly reference snapshots",
        len(sessions), len(months),
    )

    async with httpx.AsyncClient() as client:
        fetched = skipped = 0
        for d in sessions:
            path = _raw_path(vintage, "grouped", f"{d}.json.gz")
            if path.exists():
                skipped += 1
                continue
            payload = await _get(
                client, f"{BASE}/v2/aggs/grouped/locale/us/market/stocks/{d}",
                {"adjusted": "true"},
            )
            _write_raw(path, payload)
            fetched += 1
            if fetched % 50 == 0:
                logger.info("  grouped: %d fetched, %d already present", fetched, skipped)
        logger.info("grouped daily done: %d fetched, %d resumed", fetched, skipped)

        # Monthly classification snapshot, taken on the first session of each
        # month and applied FORWARD ONLY (§3a).
        for year, month in months:
            snap = next(d for d in sessions if (d.year, d.month) == (year, month))
            page, cursor = 1, None
            while True:
                path = _raw_path(vintage, "reference", f"{year:04d}-{month:02d}", f"page-{page}.json.gz")
                if path.exists():
                    payload = _read_raw(path)
                else:
                    params = {"market": "stocks", "date": str(snap), "limit": 1000}
                    if cursor:
                        params["cursor"] = cursor
                    payload = await _get(client, f"{BASE}/v3/reference/tickers", params)
                    _write_raw(path, payload)
                nxt = payload.get("next_url")
                if not nxt:
                    break
                cursor = nxt.split("cursor=")[-1]
                page += 1
            logger.info("  reference %04d-%02d: %d page(s)", year, month, page)


# ── normalization ────────────────────────────────────────────────────────────

def _classification_by_month(vintage: str) -> dict[tuple[int, int], dict[str, dict]]:
    """Forward-held monthly labels: {(y, m): {ticker: {type, exchange}}}."""
    out: dict[tuple[int, int], dict[str, dict]] = {}
    ref_root = ROOT / vintage / "raw" / "reference"
    for month_dir in sorted(ref_root.glob("*")):
        year, month = (int(x) for x in month_dir.name.split("-"))
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
    grouped_dir = ROOT / vintage / "raw" / "grouped"
    per_date: dict[date, dict] = {}

    for path in sorted(grouped_dir.glob("*.json.gz")):
        d = date.fromisoformat(path.stem.replace(".json", ""))
        results = _read_raw(path).get("results", []) or []
        labels = _label_for(labels_by_month, d)

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
            if label["type"] != "CS":
                reasons["not_common_stock"] += 1
                continue
            if label["exchange"] not in ALLOWED_EXCHANGES:
                reasons["failed_exchange"] += 1
                continue
            eligible.append(ticker)

        per_date[d] = {
            "traded": traded,
            "pre_classification": pre_class,
            "eligible_pre_mcap": eligible,
            "exclusions": dict(reasons),
        }
    return per_date


# ── step 2: classification drift audit (§3b) ─────────────────────────────────

def _bucket(label: dict | None) -> str:
    if label is None or not label.get("type"):
        return "unknown"
    return "common_stock" if label["type"] == "CS" else "etf_fund_other"


def audit_sample(vintage: str, membership: dict[date, dict]) -> dict[tuple[int, int], list]:
    """Deterministic stratified sample from the PRE-classification population.

    Canonical (ET date, ticker) ordering before any seeded draw, per the
    sampling discipline: a seed over an unordered population is not
    reproducible.
    """
    labels_by_month = _classification_by_month(vintage)
    by_month: dict[tuple[int, int], list[tuple[date, str]]] = defaultdict(list)
    for d, rec in membership.items():
        for t in rec["pre_classification"]:
            by_month[(d.year, d.month)].append((d, t))

    sampled: dict[tuple[int, int], list] = {}
    for month, pairs in sorted(by_month.items()):
        labels = labels_by_month.get(month, {})
        strata: dict[str, list] = {b: [] for b in AUDIT_BUCKETS}
        for pair in sorted(pairs):                      # canonical order
            strata[_bucket(labels.get(pair[1]))].append(pair)

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
    membership = build_membership(vintage)
    sample = audit_sample(vintage, membership)
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
                )
                payload["_audit"] = {"bucket": bucket, "date": str(d), "ticker": ticker}
                _write_raw(path, payload)
                done += 1
                if done % 100 == 0:
                    logger.info("  audit: %d/%d", done, total)
    logger.info("audit fetch complete: %d pairs", total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["spine", "audit", "report"])
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--vintage", default=None, help="ET date tag; defaults to today ET")
    args = ap.parse_args()

    vintage = args.vintage or str(_today_et())
    logger.info("vintage %s  step %s", vintage, args.step)

    if args.step == "spine":
        asyncio.run(fetch_spine(vintage, args.years))
    elif args.step == "audit":
        asyncio.run(run_audit(vintage))
    else:
        from scripts.pit_universe_report import write_report  # noqa: PLC0415

        write_report(vintage)


if __name__ == "__main__":
    main()
