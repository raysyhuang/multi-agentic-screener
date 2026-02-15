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
from src.main import run_morning_pipeline, run_afternoon_check, run_weekly_meta_review

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def start_worker() -> None:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    await init_db()

    settings = get_settings()
    scheduler = AsyncIOScheduler(timezone="US/Eastern")

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
        logger.info("Received signal %s, shutting down...", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    await stop_event.wait()
    scheduler.shutdown()
    logger.info("Worker stopped.")


if __name__ == "__main__":
    asyncio.run(start_worker())
