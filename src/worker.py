"""Heroku worker process — runs the APScheduler for pipeline jobs.

This runs as a separate dyno from the web process:
  - Morning pipeline (6:00 AM ET)
  - Afternoon position check (4:30 PM ET)
  - Weekly meta-analyst review (Sunday 7:00 PM ET)
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from src.config import get_settings
from src.db.session import init_db
from src.main import (
    run_morning_pipeline, run_afternoon_check, run_weekly_meta_review,
    _setup_logging,
)
from src.output.telegram import send_alert

_setup_logging()
logger = logging.getLogger(__name__)


async def start_worker() -> None:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.events import EVENT_JOB_ERROR

    await init_db()

    # Validate API keys at startup
    settings = get_settings()
    try:
        settings.validate_keys_for_mode()
    except ValueError as e:
        logger.error("Startup validation failed:\n%s", e)
        sys.exit(1)

    scheduler = AsyncIOScheduler(timezone="US/Eastern")

    # Error listener — log and alert on job failures
    def _job_error_handler(event):
        job_id = event.job_id
        exc = event.exception
        logger.error("Scheduler job '%s' FAILED: %s", job_id, exc, exc_info=exc)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(send_alert(
                    f"WORKER JOB FAILED\nJob: {job_id}\nError: {type(exc).__name__}: {exc}"
                ))
        except Exception:
            pass

    scheduler.add_listener(_job_error_handler, EVENT_JOB_ERROR)

    # Morning pipeline
    scheduler.add_job(
        run_morning_pipeline,
        CronTrigger(
            hour=settings.morning_run_hour,
            minute=settings.morning_run_minute,
            day_of_week="mon-fri",
            timezone="US/Eastern",
        ),
        id="morning_pipeline",
        name="Daily Morning Pipeline",
        max_instances=1,
        misfire_grace_time=300,
        coalesce=True,
    )

    # Afternoon check
    scheduler.add_job(
        run_afternoon_check,
        CronTrigger(
            hour=settings.afternoon_check_hour,
            minute=settings.afternoon_check_minute,
            day_of_week="mon-fri",
            timezone="US/Eastern",
        ),
        id="afternoon_check",
        name="Afternoon Position Check",
        max_instances=1,
        misfire_grace_time=300,
        coalesce=True,
    )

    # Weekly meta-analyst review (Sunday 7 PM ET)
    scheduler.add_job(
        run_weekly_meta_review,
        CronTrigger(
            day_of_week="sun",
            hour=19,
            minute=0,
            timezone="US/Eastern",
        ),
        id="weekly_meta_review",
        name="Weekly Meta-Analyst Review",
        max_instances=1,
        misfire_grace_time=300,
        coalesce=True,
    )

    scheduler.start()
    logger.info(
        "Worker started: morning=%02d:%02d (Mon-Fri), afternoon=%02d:%02d (Mon-Fri), weekly=Sun 19:00 ET",
        settings.morning_run_hour, settings.morning_run_minute,
        settings.afternoon_check_hour, settings.afternoon_check_minute,
    )

    # Handle one-off commands
    if "--run-now" in sys.argv:
        await run_morning_pipeline()
    elif "--check-now" in sys.argv:
        await run_afternoon_check()
    elif "--meta-now" in sys.argv:
        await run_weekly_meta_review()

    # Keep alive
    stop_event = asyncio.Event()

    def _shutdown(signum, frame):
        logger.info("Received signal %s, initiating graceful shutdown...", signum)
        scheduler.shutdown(wait=True)
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    await stop_event.wait()
    logger.info("Worker stopped.")


if __name__ == "__main__":
    asyncio.run(start_worker())
