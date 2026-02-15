"""Central configuration — loads .env and exposes typed settings."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    # --- API keys ---
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    polygon_api_key: str = ""
    fmp_api_key: str = ""
    financial_datasets_api_key: str = ""
    fred_api_key: str = ""  # https://api.stlouisfed.org — optional, yfinance fallback used

    # --- Telegram ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- Database ---
    database_url: str = ""

    # --- Model configs ---
    signal_interpreter_model: str = "claude-sonnet-4-5-20250929"
    adversarial_model: str = "gpt-5.2"
    risk_gate_model: str = "claude-opus-4-6"
    meta_analyst_model: str = "claude-opus-4-6"

    # --- Pipeline parameters ---
    min_price: float = 5.0
    min_avg_daily_volume: int = 500_000
    top_n_for_interpretation: int = 10
    top_n_for_debate: int = 5
    max_final_picks: int = 2
    holding_periods: list[int] = Field(default=[5, 10, 15])
    slippage_pct: float = 0.001  # 0.10%
    commission_per_trade: float = 1.0  # dollars

    # --- Trading mode ---
    trading_mode: str = "PAPER"  # PAPER or LIVE — paper trading until 30-day gate passes

    # --- Schedule (ET) ---
    morning_run_hour: int = 6
    morning_run_minute: int = 0
    afternoon_check_hour: int = 16
    afternoon_check_minute: int = 30

    # --- Planner / Verifier ---
    planner_model: str = "gpt-5.2-mini"
    verifier_model: str = "gpt-5.2-mini"

    # --- Retry settings ---
    agent_max_retry_attempts: int = 2
    agent_retry_cost_cap_usd: float = 0.50
    agent_retry_on_low_quality: bool = False
    max_run_cost_usd: float = 2.00
    max_verifier_retries: int = 2

    # --- Regime thresholds ---
    vix_high_threshold: float = 25.0
    vix_low_threshold: float = 15.0
    breadth_bullish_threshold: float = 0.60
    breadth_bearish_threshold: float = 0.40

    model_config = {
        "env_file": str(ENV_PATH),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
