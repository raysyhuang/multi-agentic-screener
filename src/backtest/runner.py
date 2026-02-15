"""Reproducible backtest runner — config-driven wrapper around walk-forward engine.

Usage:
    python -m src.backtest.runner --config configs/backtest_default.yaml --mode quant_only
    python -m src.backtest.runner --start 2024-06-01 --end 2025-01-31
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.metrics import PerformanceMetrics, compute_metrics
from src.backtest.walk_forward import WalkForwardResult, run_walk_forward
from src.config import get_settings, ExecutionMode
from src.data.aggregator import DataAggregator
from src.features.technical import compute_all_technical_features, compute_rsi2_features, latest_features
from src.features.fundamental import score_earnings_surprise, score_insider_activity, days_to_next_earnings
from src.signals.breakout import score_breakout
from src.signals.mean_reversion import score_mean_reversion
from src.signals.catalyst import score_catalyst
from src.signals.ranker import rank_candidates, deduplicate_signals, MODEL_MAP
from src.features.regime import classify_regime, get_regime_allowed_models, compute_breadth_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Pinned backtest configuration for reproducibility."""
    start_date: date
    end_date: date
    universe: list[str] | None = None
    mode: str = "quant_only"
    slippage_pct: float = 0.001
    commission_per_trade: float = 1.0
    holding_periods: list[int] = field(default_factory=lambda: [5, 10, 15])
    min_price: float = 5.0
    min_avg_daily_volume: int = 500_000
    top_n_candidates: int = 10
    output_dir: str = "backtest_results"
    version_tag: str = "v1"

    @property
    def config_hash(self) -> str:
        """Deterministic hash of this config for version comparison."""
        serialized = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    @classmethod
    def from_yaml(cls, path: str | Path) -> BacktestConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls(
            start_date=date.fromisoformat(raw["start_date"]),
            end_date=date.fromisoformat(raw["end_date"]),
            universe=raw.get("universe"),
            mode=raw.get("mode", "quant_only"),
            slippage_pct=raw.get("slippage_pct", 0.001),
            commission_per_trade=raw.get("commission_per_trade", 1.0),
            holding_periods=raw.get("holding_periods", [5, 10, 15]),
            min_price=raw.get("min_price", 5.0),
            min_avg_daily_volume=raw.get("min_avg_daily_volume", 500_000),
            top_n_candidates=raw.get("top_n_candidates", 10),
            output_dir=raw.get("output_dir", "backtest_results"),
            version_tag=raw.get("version_tag", "v1"),
        )


@dataclass
class BacktestReport:
    """Output of a reproducible backtest run."""
    config_hash: str
    config: dict
    walk_forward: dict  # WalkForwardResult as dict
    metrics: dict  # PerformanceMetrics as dict
    dsr: float  # Deflated Sharpe Ratio (placeholder — requires multiple strategies)
    elapsed_s: float
    timestamp: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


async def generate_backtest_signals(
    config: BacktestConfig,
    aggregator: DataAggregator,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Generate signals for the backtest period — shared quant logic (Steps 3-6).

    Returns (signals_df, price_data) for walk-forward consumption.
    """
    settings = get_settings()
    settings.slippage_pct = config.slippage_pct
    settings.commission_per_trade = config.commission_per_trade

    # Determine universe
    if config.universe:
        tickers = config.universe
    else:
        raw_universe = await aggregator.get_universe()
        from src.signals.filter import filter_universe
        filtered = filter_universe(raw_universe)
        tickers = [s["symbol"] for s in filtered[:200]]

    logger.info("Backtest universe: %d tickers", len(tickers))

    # Fetch OHLCV — extend lookback for indicator warmup
    lookback_start = config.start_date - timedelta(days=300)
    price_data = await aggregator.get_bulk_ohlcv(tickers, lookback_start, config.end_date)

    # Macro context for regime detection
    macro = await aggregator.get_macro_context()
    regime_assessment = classify_regime(
        spy_df=macro.get("spy_prices"),
        qqq_df=macro.get("qqq_prices"),
        vix=macro.get("vix"),
        yield_spread=macro.get("yield_spread_10y2y"),
    )

    breadth = compute_breadth_score(price_data)
    if breadth is not None:
        regime_assessment = classify_regime(
            spy_df=macro.get("spy_prices"),
            qqq_df=macro.get("qqq_prices"),
            vix=macro.get("vix"),
            yield_spread=macro.get("yield_spread_10y2y"),
            breadth_score=breadth,
        )

    # Feature engineering + signal generation (per ticker)
    all_signals = []
    features_by_ticker = {}
    earnings_calendar = await aggregator.get_upcoming_earnings()

    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue

        df = compute_all_technical_features(df)
        df = compute_rsi2_features(df)
        feat = latest_features(df)
        feat["ticker"] = ticker
        feat["days_to_earnings"] = days_to_next_earnings(earnings_calendar, ticker)
        features_by_ticker[ticker] = feat
        price_data[ticker] = df

        breakout = score_breakout(ticker, df, feat)
        if breakout:
            all_signals.append(breakout)

        mean_rev = score_mean_reversion(ticker, df, feat)
        if mean_rev:
            all_signals.append(mean_rev)

        catalyst = score_catalyst(
            ticker, feat,
            fundamental_data=feat.get("fundamental", {}),
            days_to_earnings=feat.get("days_to_earnings"),
            sentiment=feat.get("sentiment"),
        )
        if catalyst:
            all_signals.append(catalyst)

    logger.info("Generated %d raw signals", len(all_signals))

    all_signals = deduplicate_signals(all_signals)

    ranked = rank_candidates(
        all_signals,
        regime=regime_assessment.regime,
        features_by_ticker=features_by_ticker,
        top_n=config.top_n_candidates,
    )

    # Convert ranked candidates to signals DataFrame for walk-forward
    signal_records = []
    for c in ranked:
        signal_records.append({
            "date": config.end_date,  # signal date = latest available
            "ticker": c.ticker,
            "signal_model": c.signal_model,
            "direction": c.direction,
            "entry_price": c.entry_price,
            "stop_loss": c.stop_loss,
            "target_1": c.target_1,
        })

    signals_df = pd.DataFrame(signal_records) if signal_records else pd.DataFrame()
    return signals_df, price_data


async def run_backtest(config: BacktestConfig) -> BacktestReport:
    """Execute a full reproducible backtest."""
    start_time = time.monotonic()
    logger.info("=" * 60)
    logger.info("Starting backtest: %s → %s (mode=%s, hash=%s)",
                config.start_date, config.end_date, config.mode, config.config_hash)
    logger.info("=" * 60)

    aggregator = DataAggregator()

    # Generate signals using shared quant pipeline
    signals_df, price_data = await generate_backtest_signals(config, aggregator)

    if signals_df.empty:
        logger.warning("No signals generated — empty backtest")
        elapsed = time.monotonic() - start_time
        return BacktestReport(
            config_hash=config.config_hash,
            config=asdict(config),
            walk_forward={"total_trades": 0},
            metrics={"total_trades": 0},
            dsr=0.0,
            elapsed_s=round(elapsed, 2),
            timestamp=str(date.today()),
        )

    # Run walk-forward
    wf_result = run_walk_forward(
        signals_df=signals_df,
        price_data=price_data,
        holding_periods=config.holding_periods,
    )

    # Compute metrics from walk-forward returns
    pnls = [t.pnl_after_costs for t in wf_result.trades]
    perf_metrics = compute_metrics(pnls)

    elapsed = time.monotonic() - start_time

    # Build report
    wf_dict = {
        "total_trades": wf_result.total_trades,
        "wins": wf_result.wins,
        "losses": wf_result.losses,
        "win_rate": wf_result.win_rate,
        "avg_pnl_pct": wf_result.avg_pnl_pct,
        "avg_pnl_after_costs": wf_result.avg_pnl_after_costs,
        "total_return_pct": wf_result.total_return_pct,
        "max_drawdown_pct": wf_result.max_drawdown_pct,
        "sharpe_ratio": wf_result.sharpe_ratio,
        "sortino_ratio": wf_result.sortino_ratio,
        "profit_factor": wf_result.profit_factor,
        "avg_holding_days": wf_result.avg_holding_days,
        "by_holding_period": wf_result.by_holding_period,
    }

    report = BacktestReport(
        config_hash=config.config_hash,
        config=asdict(config),
        walk_forward=wf_dict,
        metrics=asdict(perf_metrics),
        dsr=wf_result.sharpe_ratio * 0.9,  # Placeholder DSR haircut
        elapsed_s=round(elapsed, 2),
        timestamp=str(date.today()),
    )

    # Save report
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"backtest_{config.version_tag}_{config.config_hash}.json"
    report_path.write_text(report.to_json())

    logger.info("Backtest complete in %.1fs: %d trades, Sharpe=%.2f, report=%s",
                elapsed, wf_result.total_trades, wf_result.sharpe_ratio, report_path)

    return report


def main():
    parser = argparse.ArgumentParser(description="Reproducible backtest runner")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--mode", type=str, default="quant_only",
                        choices=["quant_only", "hybrid", "agentic_full"])
    parser.add_argument("--version-tag", type=str, default="v1")
    args = parser.parse_args()

    if args.config:
        config = BacktestConfig.from_yaml(args.config)
        # CLI overrides
        if args.mode:
            config.mode = args.mode
        if args.version_tag:
            config.version_tag = args.version_tag
    elif args.start and args.end:
        config = BacktestConfig(
            start_date=date.fromisoformat(args.start),
            end_date=date.fromisoformat(args.end),
            mode=args.mode,
            version_tag=args.version_tag,
        )
    else:
        parser.error("Provide --config or both --start and --end")

    asyncio.run(run_backtest(config))


if __name__ == "__main__":
    main()
