"""Engine Collector — fetches results from all external engines in parallel.

Fail-open per engine: one engine being down does not block the others.
Returns a list of validated EngineResultPayload objects.
Quality validation rejects stale, degenerate, or mock-looking data.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime

import aiohttp
from pydantic import ValidationError

from src.config import get_settings
from src.contracts import EngineResultPayload

logger = logging.getLogger(__name__)

# Maximum age of engine run_date before it's considered stale
_MAX_STALENESS_DAYS = 2

# Map engine name -> settings attribute for URL
_ENGINE_CONFIG = {
    "koocore_d": "koocore_api_url",
    "gemini_stst": "gemini_api_url",
    "top3_7d": "top3_api_url",
}


async def _fetch_engine(
    session: aiohttp.ClientSession,
    engine_name: str,
    base_url: str,
    api_key: str,
    timeout_s: float = 30.0,
) -> EngineResultPayload | None:
    """Fetch results from a single engine. Returns None on failure."""
    url = f"{base_url.rstrip('/')}/api/engine/results"
    headers = {}
    if api_key:
        headers["X-Engine-Key"] = api_key

    try:
        start = time.monotonic()
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            elapsed = time.monotonic() - start
            if resp.status != 200:
                body = await resp.text()
                logger.warning(
                    "Engine %s returned HTTP %d (%.1fs): %s",
                    engine_name, resp.status, elapsed, body[:200],
                )
                return None

            data = await resp.json()
            payload = EngineResultPayload.model_validate(data)
            logger.info(
                "Engine %s: %d picks, status=%s (%.1fs)",
                engine_name, len(payload.picks), payload.status, elapsed,
            )
            return payload

    except ValidationError as e:
        logger.warning("Engine %s returned invalid payload: %s", engine_name, e)
        return None
    except asyncio.TimeoutError:
        logger.warning("Engine %s timed out after %.0fs", engine_name, timeout_s)
        return None
    except Exception as e:
        logger.warning("Engine %s fetch failed: %s: %s", engine_name, type(e).__name__, e)
        return None


def _validate_payload_quality(engine_name: str, payload: EngineResultPayload) -> list[str]:
    """Check an engine payload for signs of stale, mock, or degenerate data.

    Returns a list of warning strings. Empty list means the payload is clean.
    """
    warnings: list[str] = []

    # 1. Stale run_date
    try:
        run_date = datetime.strptime(payload.run_date, "%Y-%m-%d").date()
        staleness = (date.today() - run_date).days
        if staleness > _MAX_STALENESS_DAYS:
            warnings.append(f"stale run_date ({payload.run_date}, {staleness}d old)")
    except ValueError:
        warnings.append(f"unparseable run_date: {payload.run_date}")

    if not payload.picks:
        return warnings  # no picks to validate further

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

    return warnings


async def collect_engine_results() -> list[EngineResultPayload]:
    """Fetch results from all configured external engines in parallel.

    Returns only the engines that responded successfully with valid payloads.
    Payloads are quality-validated; those with critical issues are rejected.
    """
    settings = get_settings()
    api_key = settings.engine_api_key

    # Build tasks for configured engines only
    tasks: dict[str, asyncio.Task] = {}
    async with aiohttp.ClientSession() as session:
        for engine_name, url_attr in _ENGINE_CONFIG.items():
            base_url = getattr(settings, url_attr, "")
            if not base_url:
                logger.debug("Engine %s not configured (no URL), skipping", engine_name)
                continue

            tasks[engine_name] = asyncio.create_task(
                _fetch_engine(session, engine_name, base_url, api_key)
            )

        if not tasks:
            logger.info("No external engines configured")
            return []

        # Wait for all engines in parallel
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    payloads: list[EngineResultPayload] = []
    for engine_name, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            logger.warning("Engine %s raised exception: %s", engine_name, result)
        elif result is not None:
            # Quality validation — reject payloads with critical issues
            quality_warnings = _validate_payload_quality(engine_name, result)
            if quality_warnings:
                has_critical = any(
                    w.startswith("stale") or w.startswith("unparseable") or "zero/negative" in w
                    for w in quality_warnings
                )
                for w in quality_warnings:
                    logger.warning("Engine %s quality issue: %s", engine_name, w)
                if has_critical:
                    logger.warning("Engine %s REJECTED due to critical quality issues", engine_name)
                    continue
            payloads.append(result)

    logger.info(
        "Engine collection complete: %d/%d engines responded and passed quality checks",
        len(payloads), len(tasks),
    )
    return payloads
