"""Daily HTML report generation — Jinja2-based signal cards served by FastAPI."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

TEMPLATES_DIR = PROJECT_ROOT / "templates"


def get_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
    )


def generate_daily_report(
    run_date: date,
    regime: str,
    regime_details: dict,
    picks: list[dict],
    vetoed: list[str],
    pipeline_stats: dict,
) -> str:
    """Generate the daily HTML report."""
    env = get_jinja_env()
    template = env.get_template("daily_report.html")

    return template.render(
        run_date=str(run_date),
        regime=regime,
        regime_details=regime_details,
        picks=picks,
        vetoed=vetoed,
        pipeline_stats=pipeline_stats,
        total_picks=len(picks),
    )


def generate_performance_report(
    performance_data: dict,
    period_days: int = 30,
) -> str:
    """Generate a performance summary HTML page."""
    env = get_jinja_env()
    template = env.get_template("performance_report.html")

    return template.render(
        performance=performance_data,
        period_days=period_days,
    )
