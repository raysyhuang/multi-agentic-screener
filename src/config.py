"""Central configuration — loads .env and exposes typed settings."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field

logger = logging.getLogger(__name__)


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
    polygon_api_key: str = ""
    fmp_api_key: str = ""
    financial_datasets_api_key: str = ""
    fred_api_key: str = ""  # https://api.stlouisfed.org — optional, yfinance fallback used

    # --- Telegram ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_alert_prefix: str = "MAS"  # Label shown in alert headers, e.g. [MAS] or [IBKR]
    telegram_source_id: str = "mas"  # Source tag written to telegram_log rows

    # --- Database ---
    database_url: str = ""

    # --- API auth ---
    api_secret_key: str = ""  # Bearer token for /api/* routes
    allowed_origins: str = ""  # Comma-separated CORS origins (empty = same-origin only)

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
    trail_activate_pct: float = 0.5   # activate trailing stop after +0.5% MFE
    trail_distance_pct: float = 0.3   # trail 0.3% below high watermark

    # --- Two-Leg Trade Engine (V1.2) ---
    # Disabled: backtest showed partial TP dilutes winners (-0.16% avg return)
    # while trailing stop alone captures MFE efficiently. Keep code for future tuning.
    partial_tp_enabled: bool = False
    partial_tp_fraction: float = 0.5       # close 50% at partial target
    partial_tp_atr_multiple: float = 1.0   # Leg 1 target = entry + 1.0×ATR
    breakeven_after_partial: bool = True    # move stop to entry after Leg 1 fills

    # --- Entry Refinement (V1.2) ---
    entry_gap_max_atr: float = 0.2         # reject if T+1 open > close + 0.2×ATR
    volume_slope_lookback: int = 3          # bars for volume slope calculation

    # --- Regime + Volatility Gate (V1.2) ---
    choppy_min_score: int = 75             # score floor in choppy regime
    min_atr_percentile_252: float = 0.10   # block if ATR14 in bottom decile
    earnings_blackout_days: int = 2        # block picks within N days of earnings

    # --- Phase 2: Win-Rate Lift ---
    weekly_trend_gate_enabled: bool = False   # require close > 150-day SMA (30-week proxy)
    weekly_trend_sma_days: int = 150          # SMA period for weekly trend gate
    shock_killswitch_enabled: bool = False    # block when 1D true range > k × ATR14
    shock_killswitch_atr_mult: float = 3.0   # k multiplier for shock detection
    confirm_entry_enabled: bool = False       # require bullish confirmation on entry day
    confirm_mode: str = "close_gt_open"       # close_gt_open | low_gt_open_minus_atr
    blocked_entry_weekdays: str = ""          # comma-separated: 0=Mon..4=Fri (empty=none)
    early_exit_mfe_pct: float = 0.0           # exit at close when MFE exceeds this % (0=disabled)
    score_tiered_stops_enabled: bool = True    # score≥85→1.25×ATR, 70-84→0.85×ATR, <70→0.50×ATR

    # --- Veto Architecture (V1.2) ---
    veto_board_enabled: bool = True
    veto_penalty: float = 0.5              # confidence multiplier for vetoed picks
    idiosyncratic_bonus_enabled: bool = True
    idiosyncratic_bonus_multiplier: float = 1.10

    # Ticker blacklist: mean reversion backtest (S&P500, 2yr, 24K trades) showed
    # these tickers have <35% win rate with >=50 trades. Comma-separated.
    # See outputs/research/sweep_trailing_best_trades.csv for data.
    mean_reversion_blacklist: str = (
        "NKE,DECK,TTD,GNRC,CDW,DOW,CPRT,CDNS,EXE,LULU,STZ,BIIB,HSY,APD,"
        "INTC,COST,CL,PSA,AMT,LHX,TSN,PAYX,MTB,VRSK"
    )
    fmp_daily_call_budget: int = 750
    fmp_budget_warn_threshold_pct: float = 0.80
    fmp_enforce_daily_budget: bool = False
    fmp_fundamentals_max_tickers_per_run: int = 150
    # Comma-separated endpoints considered required for health checks.
    # Keep this aligned with plan-supported endpoints to avoid persistent WARNs.
    fmp_health_check_endpoints: str = "profile,earnings,insider_trading,screener,ratios,analyst_estimates"

    # --- Execution mode ---
    # Default changed to quant_only: backtest proved mean reversion signal
    # works without LLM layers (Sharpe 2.47 with trailing stop). LLM debate
    # added cost ($0.15/run) without measurable improvement.
    execution_mode: str = "quant_only"  # quant_only | hybrid | agentic_full

    # --- Trading mode ---
    trading_mode: str = "PAPER"  # PAPER or LIVE — paper trading until 30-day gate passes
    force_live: bool = False  # Override 30-day paper trading gate (emergency use)

    # --- Schedule (ET) ---
    morning_run_hour: int = 6
    morning_run_minute: int = 0
    afternoon_check_hour: int = 16
    afternoon_check_minute: int = 30

    # --- Logging ---
    log_format: str = "text"  # "text" or "json" — use json for Heroku log drains

    # --- Regime thresholds ---
    vix_high_threshold: float = 25.0
    vix_low_threshold: float = 15.0
    breadth_bullish_threshold: float = 0.60
    breadth_bearish_threshold: float = 0.40

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

    # --- MCP Data Connectors ---
    mcp_enabled: bool = True  # Master toggle for MCP enrichment layer
    mcp_enabled_providers: str = ""  # Comma-separated list (empty = all from .mcp.json)
    mcp_enrich_top_n: int = 10  # Only enrich top N candidates (controls cost)
    mcp_request_timeout_s: float = 30.0

    # --- Production Profile ---
    production_profile: str = "balanced"  # "balanced" (default) or future challenger profiles

    # --- Sniper Track ---
    sniper_enabled: bool = True
    sniper_min_score: int = 70
    sniper_atr_pct_floor: float = 5.0
    sniper_stop_atr_mult: float = 1.5
    sniper_target_atr_mult: float = 3.0
    sniper_holding_period: int = 7
    sniper_max_positions: int = 3
    sniper_time_stop_days: int = 1

    def validate_keys_for_mode(self) -> None:
        """Validate that required API keys are present.

        Raises ValueError with a clear message listing all missing keys.
        """
        missing: list[str] = []

        # Data keys are always required (the pipeline fetches market data)
        if not self.polygon_api_key and not self.fmp_api_key:
            missing.append("polygon_api_key or fmp_api_key (at least one data provider)")

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
