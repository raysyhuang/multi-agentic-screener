"""Central configuration — loads .env and exposes typed settings."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator

logger = logging.getLogger(__name__)

# Known-good model names for validation
KNOWN_MODELS = {
    # Anthropic
    "claude-opus-4-6", "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001",
    "claude-sonnet-4-20250514",
    # OpenAI
    "gpt-5.2", "gpt-5.2-nano",
    "o1", "o1-mini", "o1-preview", "o3-mini",
}


class ExecutionMode(str, Enum):
    """Pipeline execution mode — controls how much LLM processing is used."""
    QUANT_ONLY = "quant_only"      # L1-L3 only: data/features/signals/rank. Zero LLM cost.
    HYBRID = "hybrid"              # Quant shortlist + interpreter only. No debate/risk gate.
    AGENTIC_FULL = "agentic_full"  # Full pipeline with all LLM agents.

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

    # --- API auth ---
    api_secret_key: str = ""  # Bearer token for /api/* routes
    allowed_origins: str = ""  # Comma-separated CORS origins (empty = same-origin only)

    # --- Model configs ---
    signal_interpreter_model: str = "claude-sonnet-4-5-20250929"
    adversarial_model: str = "gpt-5.2"
    risk_gate_model: str = "claude-opus-4-6"
    meta_analyst_model: str = "claude-opus-4-6"

    # --- Pipeline parameters ---
    min_price: float = 5.0
    min_avg_daily_volume: int = 500_000
    max_ohlcv_tickers: int = 1000
    top_n_for_interpretation: int = 10
    top_n_for_debate: int = 5
    max_final_picks: int = 2
    holding_periods: list[int] = Field(default=[3, 5, 7])
    slippage_pct: float = 0.001  # 0.10%
    commission_per_trade: float = 1.0  # dollars
    fmp_daily_call_budget: int = 750
    fmp_budget_warn_threshold_pct: float = 0.80
    fmp_enforce_daily_budget: bool = False
    fmp_fundamentals_max_tickers_per_run: int = 150

    # --- Execution mode ---
    execution_mode: str = "agentic_full"  # quant_only | hybrid | agentic_full

    # --- Trading mode ---
    trading_mode: str = "PAPER"  # PAPER or LIVE — paper trading until 30-day gate passes
    force_live: bool = False  # Override 30-day paper trading gate (emergency use)

    # --- Schedule (ET) ---
    morning_run_hour: int = 6
    morning_run_minute: int = 0
    afternoon_check_hour: int = 16
    afternoon_check_minute: int = 30

    # --- Planner / Verifier ---
    planner_model: str = "gpt-5.2"
    verifier_model: str = "gpt-5.2"

    # --- Retry settings ---
    agent_max_retry_attempts: int = 2
    agent_retry_cost_cap_usd: float = 0.50
    agent_retry_on_low_quality: bool = True
    max_run_cost_usd: float = 2.00
    max_verifier_retries: int = 2

    # --- Logging ---
    log_format: str = "text"  # "text" or "json" — use json for Heroku log drains

    # --- Regime thresholds ---
    vix_high_threshold: float = 25.0
    vix_low_threshold: float = 15.0
    breadth_bullish_threshold: float = 0.60
    breadth_bearish_threshold: float = 0.40

    # --- External Engine URLs ---
    koocore_api_url: str = ""
    gemini_api_url: str = ""
    engine_api_key: str = ""

    # --- Local Engine Runners ---
    engine_run_mode: str = "local"  # "local" (run in-process) or "http" (legacy remote fetch)
    koocore_config_path: str = "KooCore-D/config/default.yaml"

    # --- Cross-Engine System ---
    cross_engine_enabled: bool = True
    cross_engine_model: str = "claude-opus-4-6"
    cross_engine_max_cost_usd: float = 0.50
    cross_engine_verify_before_synthesize: bool = True
    engine_fetch_timeout_s: float = 30.0
    llm_request_timeout_s: float = 90.0

    # --- Credibility Tracking ---
    credibility_lookback_days: int = 30
    credibility_min_picks_for_weight: int = 10
    convergence_2_engine_multiplier: float = 1.3
    convergence_3_engine_multiplier: float = 1.0
    convergence_4_engine_multiplier: float = 1.0
    convergence_1_engine_multiplier: float = 0.9

    # --- Capital Guardian (portfolio-level risk defense) ---
    guardian_enabled: bool = True
    guardian_max_drawdown_pct: float = 20.0       # Halt all trading beyond this drawdown
    guardian_streak_reduction_after: int = 3       # Start reducing size after N consecutive losses
    guardian_halt_after_consecutive_losses: int = 6  # Full halt after N consecutive losses
    guardian_max_portfolio_heat_pct: float = 12.0  # Soft cap for total open portfolio heat
    guardian_halt_portfolio_heat_pct: float = 25.0  # Hard halt only beyond this heat level
    guardian_overheat_sizing_floor: float = 0.35    # Sizing floor when above soft heat cap
    guardian_max_sector_concentration: int = 3     # Max positions in any single sector
    guardian_per_trade_risk_cap_pct: float = 2.0   # Max risk per trade as % of portfolio
    guardian_bear_sizing: float = 0.65             # Position size multiplier in bear regime
    guardian_choppy_sizing: float = 0.85           # Position size multiplier in choppy regime

    # --- Low-overlap portfolio guardrails ---
    low_overlap_max_positions: int = 3
    low_overlap_max_total_weight_pct: float = 30.0

    # --- Gemini STST filter thresholds (backtest adapter) ---
    gemini_momentum_adv_min: float = 300_000
    gemini_momentum_rvol_min: float = 1.0
    gemini_reversion_adv_min: float = 750_000
    gemini_reversion_rsi2_max: float = 15.0

    # --- Execution Gates (pre-synthesis safety checks) ---
    min_engines_for_trade: int = 2  # Require N engines reporting before allowing synthesis
    require_known_regime: bool = False  # Block trades when regime is "unknown"

    # --- Cross-Engine Alert Cooldown ---
    cross_engine_alert_cooldown_hours: int = 4
    engine_drop_alert_cooldown_minutes: int = 60

    # --- Regime Strategy Gate ---
    regime_strategy_gate_enabled: bool = True
    regime_gate_bear_blocked_strategies: str = "momentum"
    regime_gate_bear_penalized_strategies: str = "breakout,swing"
    regime_gate_bear_penalty_multiplier: float = 0.65

    @model_validator(mode="after")
    def _validate_model_names(self) -> "Settings":
        """Warn on unrecognized model names at startup."""
        model_fields = [
            "signal_interpreter_model", "adversarial_model", "risk_gate_model",
            "meta_analyst_model", "planner_model", "verifier_model",
            "cross_engine_model",
        ]
        for field_name in model_fields:
            value = getattr(self, field_name, "")
            if value and value not in KNOWN_MODELS:
                logger.warning(
                    "Unrecognized model '%s' in %s — may cause runtime errors. "
                    "Known models: %s",
                    value, field_name, ", ".join(sorted(KNOWN_MODELS)),
                )
        return self

    def validate_keys_for_mode(self) -> None:
        """Validate that required API keys are present for the configured execution mode.

        Raises ValueError with a clear message listing all missing keys.
        """
        missing: list[str] = []

        # Data keys are always required (all modes fetch market data)
        if not self.polygon_api_key and not self.fmp_api_key:
            missing.append("polygon_api_key or fmp_api_key (at least one data provider)")

        mode = ExecutionMode(self.execution_mode)
        if mode in (ExecutionMode.HYBRID, ExecutionMode.AGENTIC_FULL):
            if not self.anthropic_api_key and not self.openai_api_key:
                missing.append(
                    "anthropic_api_key or openai_api_key (required for LLM agents)"
                )

        if missing:
            raise ValueError(
                f"Missing required API keys for execution_mode={self.execution_mode}:\n"
                + "\n".join(f"  - {k}" for k in missing)
            )

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
