"""Engine Collector — fetches results from all external engines in parallel.

Fail-open per engine: one engine being down does not block the others.
Returns a list of validated EngineResultPayload objects.
"""

from __future__ import annotations

import asyncio
import logging
import time

import aiohttp
from pydantic import ValidationError

from src.config import get_settings
from src.contracts import EngineResultPayload

logger = logging.getLogger(__name__)

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


async def collect_engine_results() -> list[EngineResultPayload]:
    """Fetch results from all configured external engines in parallel.

    Returns only the engines that responded successfully with valid payloads.
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
            payloads.append(result)

    logger.info(
        "Engine collection complete: %d/%d engines responded",
        len(payloads), len(tasks),
    )
    return payloads
