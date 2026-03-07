"""Engine Collector — fetches results from all external engines in parallel.

Supports three modes controlled by ``engine_run_mode`` setting:
  - ``"local"``: runs KooCore-D and Gemini STST in-process
  - ``"http"``: fetches from remote Heroku apps (legacy)
  - ``"hybrid"`` (default): KooCore-D via HTTP, Gemini STST locally

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
from src.utils.trading_calendar import trading_days_between

logger = logging.getLogger(__name__)


EngineFailureKind = Literal[
    "exception",
    "no_output",
    "no_response",
    "quality_rejected",
]

CollectionTime = Literal["morning", "evening"]

EngineFailureReasonCode = Literal[
    "stale",
    "expected_stale",
    "no_artifacts",
    "schema_invalid",
    "risk_invalid",
    "timeout",
    "no_response",
    "exception",
]


@dataclass(frozen=True)
class EngineFailure:
    """Structured description of why an engine was not usable this cycle."""

    engine_name: str
    kind: EngineFailureKind
    reason_code: EngineFailureReasonCode = "schema_invalid"
    detail: str = ""


# Maximum age of engine run_date in *trading days* before it's considered stale.
# Using trading days instead of calendar days avoids rejecting Friday data on
# Monday (which is 3 calendar days but only 1 trading day).
_MAX_STALENESS_TRADING_DAYS = 2

# Map engine name -> settings attribute for URL (HTTP mode only)
_ENGINE_CONFIG = {
    "koocore_d": "koocore_api_url",
    "gemini_stst": "gemini_api_url",
    "top3_7d": "top3_7d_api_url",
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


def _format_quality_issue(code: str, message: str) -> str:
    return f"{code}: {message}"


def _reason_code_from_issue(issue: str) -> str:
    if ":" not in issue:
        return "schema_invalid"
    return issue.split(":", 1)[0].strip().lower() or "schema_invalid"


def _validate_payload_quality(
    engine_name: str,
    payload: EngineResultPayload,
    *,
    collection_time: CollectionTime = "morning",
    asof_date: date | None = None,
) -> list[str]:
    """Check an engine payload for signs of stale, mock, or degenerate data.

    Returns a list of warning strings. Empty list means the payload is clean.
    """
    warnings: list[str] = []
    reference_date = asof_date or date.today()
    status = (payload.status or "").strip().lower()

    run_date: date | None = None
    trading_days: int | None = None
    calendar_days: int | None = None

    # 0. Parse run_date once for status/staleness decisions.
    try:
        run_date = datetime.strptime(payload.run_date, "%Y-%m-%d").date()
        trading_days = trading_days_between(run_date, reference_date)
        calendar_days = (reference_date - run_date).days
    except ValueError:
        warnings.append(
            _format_quality_issue("schema_invalid", f"unparseable run_date: {payload.run_date}")
        )

    # 1. Non-success status.
    if status not in {"success", "ok"}:
        if (
            engine_name == "top3_7d"
            and status == "no_artifacts"
            and collection_time == "morning"
            and trading_days is not None
            and trading_days <= 1
        ):
            warnings.append(
                _format_quality_issue(
                    "expected_stale",
                    (
                        f"Top3-7D no_artifacts is expected during morning collection "
                        f"(run_date={payload.run_date}, lag={trading_days} trading day)"
                    ),
                )
            )
        elif status == "no_artifacts":
            warnings.append(
                _format_quality_issue("no_artifacts", f"non-success status={payload.status}")
            )
        else:
            warnings.append(
                _format_quality_issue("schema_invalid", f"non-success status={payload.status}")
            )

    # 2. Stale run_date (trading-day aware)
    if trading_days is not None:
        if trading_days > _MAX_STALENESS_TRADING_DAYS:
            warnings.append(
                _format_quality_issue(
                    "stale",
                    (
                        f"stale run_date ({payload.run_date}, {calendar_days}d / "
                        f"{trading_days} trading days old)"
                    ),
                )
            )

    # 2b. Pipeline-failure signature: screened nothing and emitted nothing.
    #     Skip if check #1 already classified this as expected_stale.
    if payload.candidates_screened == 0 and not payload.picks:
        already_expected = any(w.startswith("expected_stale:") for w in warnings)
        if not already_expected:
            if status == "no_artifacts":
                warnings.append(
                    _format_quality_issue("no_artifacts", "zero candidates screened and zero picks")
                )
            else:
                warnings.append(
                    _format_quality_issue("schema_invalid", "zero candidates screened and zero picks")
                )
        return warnings  # nothing else to validate

    # 3. Zero or negative entry prices
    bad_prices = [p.ticker for p in payload.picks if p.entry_price <= 0]
    if bad_prices:
        warnings.append(
            _format_quality_issue(
                "risk_invalid",
                f"{len(bad_prices)} picks with zero/negative entry price: {bad_prices[:5]}",
            )
        )

    # 4. All confidence scores identical (suggests mock/hardcoded data)
    confidences = [p.confidence for p in payload.picks]
    if len(set(confidences)) == 1 and len(confidences) > 1:
        warnings.append(
            _format_quality_issue(
                "hint",
                f"all {len(confidences)} picks have identical confidence={confidences[0]}",
            )
        )

    # 5. Unreasonable pick count (too many suggests no filtering)
    if len(payload.picks) > 20:
        warnings.append(
            _format_quality_issue("hint", f"unusually high pick count ({len(payload.picks)})")
        )

    # 6. Stop/target direction sanity (for long picks: stop < entry < target)
    inverted = []
    for p in payload.picks:
        if p.stop_loss is not None and p.stop_loss >= p.entry_price:
            inverted.append(f"{p.ticker}(stop={p.stop_loss}>=entry={p.entry_price})")
        if p.target_price is not None and p.target_price <= p.entry_price:
            inverted.append(f"{p.ticker}(target={p.target_price}<=entry={p.entry_price})")
    if inverted:
        warnings.append(
            _format_quality_issue("risk_invalid", f"inverted stop/target: {inverted[:5]}")
        )

    # 7. Stop loss too far from entry (>30% away suggests bad data)
    wide_stops = []
    for p in payload.picks:
        if p.stop_loss is not None and p.entry_price > 0:
            pct_distance = abs(p.entry_price - p.stop_loss) / p.entry_price
            if pct_distance > 0.30:
                wide_stops.append(f"{p.ticker}({pct_distance:.0%})")
    if wide_stops:
        warnings.append(
            _format_quality_issue(
                "risk_invalid",
                f"stop loss >30% from entry: {wide_stops[:5]}",
            )
        )

    # 8. Missing risk parameters (strict): picks without stop/target are not tradable
    missing_stop = [
        p.ticker for p in payload.picks
        if p.stop_loss is None or p.stop_loss <= 0
    ]
    if missing_stop:
        warnings.append(
            _format_quality_issue(
                "risk_invalid",
                f"{len(missing_stop)} picks missing stop_loss: {missing_stop[:5]}",
            )
        )

    missing_target = [
        p.ticker for p in payload.picks
        if p.target_price is None or p.target_price <= 0
    ]
    if missing_target:
        warnings.append(
            _format_quality_issue(
                "risk_invalid",
                f"{len(missing_target)} picks missing target_price: {missing_target[:5]}",
            )
        )

    # 9. Duplicate full risk tuples across different tickers (data-mapping smell)
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
        warnings.append(
            _format_quality_issue(
                "schema_invalid",
                f"duplicate price tuples across tickers: {sample}",
            )
        )

    # 10. Optional quality hint: all picks have empty score metadata
    if engine_name == "koocore_d" and payload.picks and all(
        not ((p.metadata or {}).get("scores"))
        for p in payload.picks
    ):
        warnings.append(_format_quality_issue("hint", "all picks have empty metadata.scores"))

    return warnings


def _is_critical_quality_issue(warnings: list[str]) -> bool:
    """Return True if any warning should reject an engine payload.

    All warnings are formatted via ``_format_quality_issue("code", msg)`` →
    ``"code: msg"``.  We match on the prefix code only.  The ``hint`` code is
    intentionally excluded — hints don't reject payloads.
    """
    _CRITICAL_CODES = {"expected_stale", "stale", "schema_invalid", "risk_invalid", "no_artifacts"}
    return any(
        _reason_code_from_issue(w) in _CRITICAL_CODES
        for w in warnings
    )


async def _collect_local(
    target_date: date | None = None,
    *,
    collection_time: CollectionTime = "morning",
) -> tuple[list[EngineResultPayload], list[EngineFailure]]:
    """Run engines locally in parallel and return validated payloads + failure list."""
    from src.engines.koocore_runner import run_koocore_locally
    from src.engines.gemini_runner import run_gemini_locally

    logger.info("Collecting engine results in LOCAL mode")

    tasks = {
        "koocore_d": asyncio.create_task(run_koocore_locally()),
        "gemini_stst": asyncio.create_task(run_gemini_locally(target_date=target_date)),
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
                    reason_code="exception",
                    detail=type(result).__name__,
                ),
            )
        elif result is None:
            logger.warning(
                "Engine %s returned None — engine produced no output this cycle",
                engine_name,
            )
            failed_engines.append(
                EngineFailure(
                    engine_name=engine_name,
                    kind="no_output",
                    reason_code="no_response",
                )
            )
        else:
            quality_warnings = _validate_payload_quality(
                engine_name,
                result,
                collection_time=collection_time,
                asof_date=target_date,
            )
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
                            reason_code=_reason_code_from_issue(quality_warnings[0]),
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


async def _collect_hybrid(
    target_date: date | None = None,
    *,
    collection_time: CollectionTime = "morning",
) -> tuple[list[EngineResultPayload], list[EngineFailure]]:
    """Hybrid mode: KooCore-D via HTTP (GitHub Action pushes results), Gemini STST locally."""
    from src.engines.gemini_runner import run_gemini_locally

    settings = get_settings()
    api_key = settings.engine_api_key
    timeout_s = settings.engine_fetch_timeout_s
    koocore_url = settings.koocore_api_url
    top3_7d_url = settings.top3_7d_api_url

    logger.info("Collecting engine results in HYBRID mode")

    # KooCore-D: fetch via HTTP (results pushed by GitHub Action)
    async def _fetch_koocore_http() -> EngineResultPayload | None:
        if not koocore_url:
            logger.warning("KooCore-D URL not configured; skipping HTTP fetch")
            return None
        async with aiohttp.ClientSession() as session:
            return await _fetch_with_generic_then_custom(
                session=session,
                engine_name="koocore_d",
                base_url=koocore_url,
                api_key=api_key,
                timeout_s=timeout_s,
                custom_fetcher=fetch_koocore,
            )

    # Top3-7D: fetch via HTTP (standard EngineResultPayload, no custom adapter)
    async def _fetch_top3_7d_http() -> EngineResultPayload | None:
        if not top3_7d_url:
            logger.warning("Top3-7D URL not configured; skipping HTTP fetch")
            return None
        async with aiohttp.ClientSession() as session:
            return await _fetch_engine(
                session=session,
                engine_name="top3_7d",
                base_url=top3_7d_url,
                api_key=api_key,
                timeout_s=timeout_s,
            )

    # Run KooCore-D (HTTP), Gemini STST (local), and Top3-7D (HTTP) in parallel
    koocore_task = asyncio.create_task(_fetch_koocore_http())
    gemini_task = asyncio.create_task(run_gemini_locally(target_date=target_date))
    top3_7d_task = asyncio.create_task(_fetch_top3_7d_http())
    results = await asyncio.gather(koocore_task, gemini_task, top3_7d_task, return_exceptions=True)

    engine_names = ["koocore_d", "gemini_stst", "top3_7d"]
    payloads: list[EngineResultPayload] = []
    failed_engines: list[EngineFailure] = []

    for engine_name, result in zip(engine_names, results):
        if isinstance(result, Exception):
            logger.warning("Engine %s raised exception in hybrid mode: %s", engine_name, result)
            failed_engines.append(
                EngineFailure(
                    engine_name=engine_name,
                    kind="exception",
                    reason_code="exception",
                    detail=type(result).__name__,
                ),
            )
        elif result is None:
            kind: EngineFailureKind = "no_response" if engine_name in ("koocore_d", "top3_7d") else "no_output"
            logger.warning("Engine %s returned no data in hybrid mode", engine_name)
            failed_engines.append(
                EngineFailure(
                    engine_name=engine_name,
                    kind=kind,
                    reason_code="no_response",
                )
            )
        else:
            quality_warnings = _validate_payload_quality(
                engine_name,
                result,
                collection_time=collection_time,
                asof_date=target_date,
            )
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
                            reason_code=_reason_code_from_issue(quality_warnings[0]),
                            detail=quality_warnings[0] if quality_warnings else "",
                        ),
                    )
                    continue
            payloads.append(result)

    logger.info(
        "Hybrid engine collection complete: %d/%d engines passed quality checks",
        len(payloads), len(engine_names),
    )
    return payloads, failed_engines


async def _collect_http(
    target_date: date | None = None,
    *,
    collection_time: CollectionTime = "morning",
) -> tuple[list[EngineResultPayload], list[EngineFailure]]:
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
                    reason_code="exception",
                    detail=type(result).__name__,
                ),
            )
        elif result is None:
            logger.warning("Engine %s returned no response in HTTP mode", engine_name)
            failed_engines.append(
                EngineFailure(
                    engine_name=engine_name,
                    kind="no_response",
                    reason_code="no_response",
                )
            )
        elif result is not None:
            # Quality validation — reject payloads with critical issues
            quality_warnings = _validate_payload_quality(
                engine_name,
                result,
                collection_time=collection_time,
                asof_date=target_date,
            )
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
                            reason_code=_reason_code_from_issue(quality_warnings[0]),
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


async def collect_engine_results(
    target_date: date | None = None,
    *,
    collection_time: CollectionTime = "morning",
) -> tuple[list[EngineResultPayload], list[EngineFailure]]:
    """Collect results from all engines in parallel.

    Mode is controlled by ``engine_run_mode`` setting:
      - ``"local"``: runs engines in-process
      - ``"http"``: fetches from remote Heroku apps (legacy)
      - ``"hybrid"`` (default): KooCore-D via HTTP, Gemini STST locally

    Returns (payloads, failed_engines) where failed_engines is a list of
    human-readable descriptions of engines that failed to report.
    """
    settings = get_settings()
    mode = (settings.engine_run_mode or "hybrid").strip().lower()

    if mode == "local":
        return await _collect_local(target_date=target_date, collection_time=collection_time)
    elif mode == "hybrid":
        return await _collect_hybrid(target_date=target_date, collection_time=collection_time)
    else:
        return await _collect_http(target_date=target_date, collection_time=collection_time)
