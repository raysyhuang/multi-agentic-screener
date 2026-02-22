"""Engine Collector — fetches results from all external engines in parallel.

Supports two modes controlled by ``engine_run_mode`` setting:
  - ``"local"``: runs KooCore-D and Gemini STST in-process (default)
  - ``"http"``: fetches from remote Heroku apps (legacy)

Fail-open per engine: one engine being down does not block the others.
Returns a list of validated EngineResultPayload objects.
Quality validation rejects stale, degenerate, or mock-looking data.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import time
from datetime import date, datetime, timedelta
from typing import Literal

import aiohttp
from pydantic import ValidationError

from src.config import get_settings
from src.contracts import EngineResultPayload
from src.engines.koocore_adapter import fetch_koocore

logger = logging.getLogger(__name__)


EngineFailureKind = Literal[
    "exception",
    "no_output",
    "no_response",
    "quality_rejected",
]


@dataclass(frozen=True)
class EngineFailure:
    """Structured description of why an engine was not usable this cycle."""

    engine_name: str
    kind: EngineFailureKind
    detail: str = ""


# Maximum age of engine run_date in *trading days* before it's considered stale.
# Using trading days instead of calendar days avoids rejecting Friday data on
# Monday (which is 3 calendar days but only 1 trading day).
_MAX_STALENESS_TRADING_DAYS = 2

# Map engine name -> settings attribute for URL (HTTP mode only)
_ENGINE_CONFIG = {
    "koocore_d": "koocore_api_url",
    "gemini_stst": "gemini_api_url",
}


async def _fetch_engine(
    session: aiohttp.ClientSession,
    engine_name: str,
    base_url: str,
    api_key: str,
    timeout_s: float = 30.0,
) -> EngineResultPayload | None:
    """Fetch results from a single engine. Returns None on failure."""
    headers = {}
    if api_key:
        headers["X-Engine-Key"] = api_key

    candidate_paths = (
        "/api/engine/results",
        "/api/engine/results/latest",
        "/api/results",
    )

    for path in candidate_paths:
        url = f"{base_url.rstrip('/')}{path}"
        try:
            start = time.monotonic()
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
                elapsed = time.monotonic() - start
                if resp.status == 404:
                    continue
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "Engine %s returned HTTP %d on %s (%.1fs): %s",
                        engine_name, resp.status, path, elapsed, body[:200],
                    )
                    return None

                data = await resp.json()
                payload = EngineResultPayload.model_validate(data)
                logger.info(
                    "Engine %s: %d picks, status=%s via %s (%.1fs)",
                    engine_name, len(payload.picks), payload.status, path, elapsed,
                )
                return payload

        except ValidationError as e:
            logger.warning("Engine %s returned invalid payload on %s: %s", engine_name, path, e)
            return None
        except asyncio.TimeoutError:
            logger.warning("Engine %s timed out after %.0fs on %s", engine_name, timeout_s, path)
            return None
        except Exception as e:
            logger.warning(
                "Engine %s fetch failed on %s: %s: %s",
                engine_name, path, type(e).__name__, e,
            )
            return None

    logger.warning("Engine %s has no recognized results endpoint", engine_name)
    return None


async def _fetch_with_custom_fallback(
    session: aiohttp.ClientSession,
    engine_name: str,
    base_url: str,
    api_key: str,
    timeout_s: float,
    custom_fetcher,
) -> EngineResultPayload | None:
    """Try engine-specific adapter first, then generic endpoint fallback."""
    result = await custom_fetcher(
        session,
        base_url,
        api_key,
        timeout_s=timeout_s,
    )
    if result is not None and not (
        result.candidates_screened == 0 and not result.picks
    ):
        return result

    logger.warning(
        "Engine %s custom adapter returned no usable data; trying generic results endpoint",
        engine_name,
    )
    return await _fetch_engine(
        session=session,
        engine_name=engine_name,
        base_url=base_url,
        api_key=api_key,
        timeout_s=timeout_s,
    )


async def _fetch_with_generic_then_custom(
    session: aiohttp.ClientSession,
    engine_name: str,
    base_url: str,
    api_key: str,
    timeout_s: float,
    custom_fetcher,
) -> EngineResultPayload | None:
    """Try generic endpoint first, then fall back to engine-specific adapter.

    This avoids noisy legacy-path failures when an engine already exposes
    a standardized `/api/engine/results` payload.
    """
    generic = await _fetch_engine(
        session=session,
        engine_name=engine_name,
        base_url=base_url,
        api_key=api_key,
        timeout_s=timeout_s,
    )
    if generic is not None:
        return generic

    logger.warning(
        "Engine %s generic endpoint unavailable; falling back to custom adapter",
        engine_name,
    )
    return await custom_fetcher(
        session,
        base_url,
        api_key,
        timeout_s=timeout_s,
    )


def _trading_days_between(start: date, end: date) -> int:
    """Count weekday (Mon-Fri) days between two dates, exclusive of start."""
    if start >= end:
        return 0
    count = 0
    current = start
    one_day = timedelta(days=1)
    while current < end:
        current += one_day
        if current.weekday() < 5:  # Mon=0 … Fri=4
            count += 1
    return count


def _validate_payload_quality(engine_name: str, payload: EngineResultPayload) -> list[str]:
    """Check an engine payload for signs of stale, mock, or degenerate data.

    Returns a list of warning strings. Empty list means the payload is clean.
    """
    warnings: list[str] = []

    # 0. Non-success status
    status = (payload.status or "").strip().lower()
    if status not in {"success", "ok"}:
        warnings.append(f"non-success status={payload.status}")

    # 1. Stale run_date (trading-day aware)
    try:
        run_date = datetime.strptime(payload.run_date, "%Y-%m-%d").date()
        trading_days = _trading_days_between(run_date, date.today())
        if trading_days > _MAX_STALENESS_TRADING_DAYS:
            calendar_days = (date.today() - run_date).days
            warnings.append(
                f"stale run_date ({payload.run_date}, {calendar_days}d / {trading_days} trading days old)"
            )
    except ValueError:
        warnings.append(f"unparseable run_date: {payload.run_date}")

    # 1b. Pipeline-failure signature: screened nothing and emitted nothing
    if payload.candidates_screened == 0 and not payload.picks:
        warnings.append("zero candidates screened and zero picks")
        return warnings  # nothing else to validate

    # 2. Zero or negative entry prices
    bad_prices = [p.ticker for p in payload.picks if p.entry_price <= 0]
    if bad_prices:
        warnings.append(f"{len(bad_prices)} picks with zero/negative entry price: {bad_prices[:5]}")

    # 3. All confidence scores identical (suggests mock/hardcoded data)
    confidences = [p.confidence for p in payload.picks]
    if len(set(confidences)) == 1 and len(confidences) > 1:
        warnings.append(f"all {len(confidences)} picks have identical confidence={confidences[0]}")

    # 4. Unreasonable pick count (too many suggests no filtering)
    if len(payload.picks) > 20:
        warnings.append(f"unusually high pick count ({len(payload.picks)})")

    # 5. Stop/target direction sanity (for long picks: stop < entry < target)
    inverted = []
    for p in payload.picks:
        if p.stop_loss is not None and p.stop_loss >= p.entry_price:
            inverted.append(f"{p.ticker}(stop={p.stop_loss}>=entry={p.entry_price})")
        if p.target_price is not None and p.target_price <= p.entry_price:
            inverted.append(f"{p.ticker}(target={p.target_price}<=entry={p.entry_price})")
    if inverted:
        warnings.append(f"inverted stop/target: {inverted[:5]}")

    # 6. Stop loss too far from entry (>30% away suggests bad data)
    wide_stops = []
    for p in payload.picks:
        if p.stop_loss is not None and p.entry_price > 0:
            pct_distance = abs(p.entry_price - p.stop_loss) / p.entry_price
            if pct_distance > 0.30:
                wide_stops.append(f"{p.ticker}({pct_distance:.0%})")
    if wide_stops:
        warnings.append(f"stop loss >30% from entry: {wide_stops[:5]}")

    # 7. Missing risk parameters (strict): picks without stop/target are not tradable
    missing_stop = [
        p.ticker for p in payload.picks
        if p.stop_loss is None or p.stop_loss <= 0
    ]
    if missing_stop:
        warnings.append(
            f"{len(missing_stop)} picks missing stop_loss: {missing_stop[:5]}"
        )

    missing_target = [
        p.ticker for p in payload.picks
        if p.target_price is None or p.target_price <= 0
    ]
    if missing_target:
        warnings.append(
            f"{len(missing_target)} picks missing target_price: {missing_target[:5]}"
        )

    # 8. Duplicate full risk tuples across different tickers (data-mapping smell)
    tuple_to_tickers: dict[tuple[float, float, float], list[str]] = {}
    for p in payload.picks:
        key = (
            round(float(p.entry_price), 2),
            round(float(p.stop_loss or 0), 2),
            round(float(p.target_price or 0), 2),
        )
        tuple_to_tickers.setdefault(key, []).append(p.ticker)

    duplicate_tuples = [
        (k, v) for k, v in tuple_to_tickers.items() if len(set(v)) > 1
    ]
    if duplicate_tuples:
        sample = [
            f"{tickers}->{prices}"
            for prices, tickers in [
                (t[0], t[1][:5]) for t in duplicate_tuples[:3]
            ]
        ]
        warnings.append(f"duplicate price tuples across tickers: {sample}")

    # 9. Optional quality hint: all picks have empty score metadata
    if engine_name == "koocore_d" and payload.picks and all(
        not ((p.metadata or {}).get("scores"))
        for p in payload.picks
    ):
        warnings.append("all picks have empty metadata.scores")

    return warnings


def _is_critical_quality_issue(warnings: list[str]) -> bool:
    """Return True if any warning should reject an engine payload."""
    return any(
        w.startswith("stale")
        or w.startswith("unparseable")
        or "zero/negative" in w
        or w.startswith("non-success status")
        or w.startswith("zero candidates screened and zero picks")
        or "missing stop_loss" in w
        or "missing target_price" in w
        or "duplicate price tuples across tickers" in w
        for w in warnings
    )


async def _collect_local() -> tuple[list[EngineResultPayload], list[EngineFailure]]:
    """Run engines locally in parallel and return validated payloads + failure list."""
    from src.engines.koocore_runner import run_koocore_locally
    from src.engines.gemini_runner import run_gemini_locally

    logger.info("Collecting engine results in LOCAL mode")

    tasks = {
        "koocore_d": asyncio.create_task(run_koocore_locally()),
        "gemini_stst": asyncio.create_task(run_gemini_locally()),
    }

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    payloads: list[EngineResultPayload] = []
    failed_engines: list[EngineFailure] = []
    for engine_name, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            logger.warning("Engine %s local run raised exception: %s", engine_name, result)
            failed_engines.append(
                EngineFailure(
                    engine_name=engine_name,
                    kind="exception",
                    detail=type(result).__name__,
                ),
            )
        elif result is None:
            logger.warning(
                "Engine %s returned None — engine produced no output this cycle",
                engine_name,
            )
            failed_engines.append(EngineFailure(engine_name=engine_name, kind="no_output"))
        else:
            quality_warnings = _validate_payload_quality(engine_name, result)
            if quality_warnings:
                has_critical = _is_critical_quality_issue(quality_warnings)
                for w in quality_warnings:
                    logger.warning("Engine %s quality issue: %s", engine_name, w)
                if has_critical:
                    logger.warning("Engine %s REJECTED due to critical quality issues", engine_name)
                    failed_engines.append(
                        EngineFailure(
                            engine_name=engine_name,
                            kind="quality_rejected",
                            detail=quality_warnings[0] if quality_warnings else "",
                        ),
                    )
                    continue
            payloads.append(result)

    logger.info(
        "Local engine collection complete: %d/%d engines passed quality checks",
        len(payloads), len(tasks),
    )
    return payloads, failed_engines


async def _collect_http() -> tuple[list[EngineResultPayload], list[EngineFailure]]:
    """Fetch results from remote engines via HTTP (legacy mode)."""
    settings = get_settings()
    api_key = settings.engine_api_key
    timeout_s = settings.engine_fetch_timeout_s

    # Engines with custom adapters (don't use generic /api/engine/results)
    _CUSTOM_ADAPTERS = {
        "koocore_d": fetch_koocore,
    }
    _PREFER_GENERIC_THEN_CUSTOM = {"koocore_d"}

    # Build tasks for configured engines only
    tasks: dict[str, asyncio.Task] = {}
    async with aiohttp.ClientSession() as session:
        for engine_name, url_attr in _ENGINE_CONFIG.items():
            base_url = getattr(settings, url_attr, "")
            if not base_url:
                logger.debug("Engine %s not configured (no URL), skipping", engine_name)
                continue

            if engine_name in _CUSTOM_ADAPTERS:
                if engine_name in _PREFER_GENERIC_THEN_CUSTOM:
                    tasks[engine_name] = asyncio.create_task(
                        _fetch_with_generic_then_custom(
                            session=session,
                            engine_name=engine_name,
                            base_url=base_url,
                            api_key=api_key,
                            timeout_s=timeout_s,
                            custom_fetcher=_CUSTOM_ADAPTERS[engine_name],
                        ),
                    )
                else:
                    tasks[engine_name] = asyncio.create_task(
                        _fetch_with_custom_fallback(
                            session=session,
                            engine_name=engine_name,
                            base_url=base_url,
                            api_key=api_key,
                            timeout_s=timeout_s,
                            custom_fetcher=_CUSTOM_ADAPTERS[engine_name],
                        ),
                    )
            else:
                tasks[engine_name] = asyncio.create_task(
                    _fetch_engine(
                        session, engine_name, base_url, api_key, timeout_s=timeout_s,
                    )
                )

        if not tasks:
            logger.info("No external engines configured")
            return [], []

        # Wait for all engines in parallel
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    payloads: list[EngineResultPayload] = []
    failed_engines: list[EngineFailure] = []
    for engine_name, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            logger.warning("Engine %s raised exception: %s", engine_name, result)
            failed_engines.append(
                EngineFailure(
                    engine_name=engine_name,
                    kind="exception",
                    detail=type(result).__name__,
                ),
            )
        elif result is None:
            logger.warning("Engine %s returned no response in HTTP mode", engine_name)
            failed_engines.append(EngineFailure(engine_name=engine_name, kind="no_response"))
        elif result is not None:
            # Quality validation — reject payloads with critical issues
            quality_warnings = _validate_payload_quality(engine_name, result)
            if quality_warnings:
                has_critical = _is_critical_quality_issue(quality_warnings)
                for w in quality_warnings:
                    logger.warning("Engine %s quality issue: %s", engine_name, w)
                if has_critical:
                    logger.warning("Engine %s REJECTED due to critical quality issues", engine_name)
                    failed_engines.append(
                        EngineFailure(
                            engine_name=engine_name,
                            kind="quality_rejected",
                            detail=quality_warnings[0] if quality_warnings else "",
                        ),
                    )
                    continue
            payloads.append(result)

    logger.info(
        "Engine collection complete: %d/%d engines responded and passed quality checks",
        len(payloads), len(tasks),
    )
    return payloads, failed_engines


async def collect_engine_results() -> tuple[list[EngineResultPayload], list[EngineFailure]]:
    """Collect results from all engines in parallel.

    Mode is controlled by ``engine_run_mode`` setting:
      - ``"local"`` (default): runs engines in-process
      - ``"http"``: fetches from remote Heroku apps

    Returns (payloads, failed_engines) where failed_engines is a list of
    human-readable descriptions of engines that failed to report.
    """
    settings = get_settings()
    mode = (settings.engine_run_mode or "local").strip().lower()

    if mode == "local":
        return await _collect_local()
    else:
        return await _collect_http()
