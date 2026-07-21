"""Heroku worker process — runs the APScheduler for pipeline jobs.

This runs as a separate dyno from the web process:
  - Morning pipeline (6:00 AM ET)
  - Afternoon position check (4:30 PM ET)
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from src.config import get_settings
from src.db.session import init_db
from src.main import (
    run_morning_pipeline, run_afternoon_check, _setup_logging,
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

    # One-off mode: run exactly one job and exit (no scheduler loop).
    one_off_jobs = {
        "--run-now": run_morning_pipeline,
        "--check-now": run_afternoon_check,
    }
    for flag, job in one_off_jobs.items():
        if flag in sys.argv:
            logger.info("Running one-off job %s", flag)
            await job()
            logger.info("One-off job %s complete; exiting worker", flag)
            return

    # Guard: an unrecognized flag (e.g. a removed one like --collect-now or
    # --meta-now) must fail loudly, NOT fall through and start the blocking
    # scheduler loop — in CI that would hang the job until the runner timeout.
    unknown_flags = [a for a in sys.argv[1:] if a.startswith("--")]
    if unknown_flags:
        logger.error(
            "Unknown worker flag(s) %s. Valid one-off flags: %s. "
            "The evening (--collect-now) and weekly (--meta-now) jobs were "
            "removed with the cross-engine/LLM subsystems.",
            unknown_flags, ", ".join(one_off_jobs),
        )
        sys.exit(2)

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
        except Exception as alert_exc:
            logger.error("Failed to send worker failure alert for job '%s': %s", job_id, alert_exc)

    scheduler.add_listener(_job_error_handler, EVENT_JOB_ERROR)

    # Morning pipeline — 1h grace so R14 restarts don't silently skip the run
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
        misfire_grace_time=3600,
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
        misfire_grace_time=3600,
        coalesce=True,
    )

    scheduler.start()

    # Log next scheduled run times for operational visibility
    for job in scheduler.get_jobs():
        logger.info("Scheduled job '%s' — next run: %s", job.name, job.next_run_time)

    logger.info(
        "Worker started: morning=%02d:%02d, afternoon=%02d:%02d (all ET, Mon-Fri)",
        settings.morning_run_hour, settings.morning_run_minute,
        settings.afternoon_check_hour, settings.afternoon_check_minute,
    )

    # Keep alive
    stop_event = asyncio.Event()

    def _shutdown(signum, frame):
        logger.info("Received signal %s, initiating graceful shutdown...", signum)
        # Use wait=False so we don't block past Heroku's 30s SIGTERM grace period.
        # Running jobs will be interrupted, but the pipeline is idempotent (re-run safe).
        scheduler.shutdown(wait=False)
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    await stop_event.wait()
    logger.info("Worker stopped.")


if __name__ == "__main__":
    asyncio.run(start_worker())
