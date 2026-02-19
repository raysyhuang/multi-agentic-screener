"""Multi-engine backtest orchestrator.

Main loop: pre-fetches OHLCV for the full date range, iterates each trading
day sequentially, calls all enabled engine adapters, synthesizes picks, and
runs the portfolio simulation.

Usage::

    python -m src.backtest.multi_engine.orchestrator \\
        --config configs/multi_engine_backtest.yaml

    python -m src.backtest.multi_engine.orchestrator \\
        --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.multi_engine.adapters.base import EngineAdapter, NormalizedPick
from src.backtest.multi_engine.adapters.mas_adapter import MASAdapter
from src.backtest.multi_engine.adapters.koocore_adapter import KooCoreAdapter
from src.backtest.multi_engine.adapters.gemini_adapter import GeminiAdapter
from src.backtest.multi_engine.regime_tracker import classify_regime_for_date
from src.backtest.multi_engine.synthesizer import (
    RollingCredibilityTracker,
    SynthesisConfig,
    synthesize_picks,
)
from src.backtest.multi_engine.portfolio_sim import (
    DailyPickRecord,
    PortfolioConfig,
    run_portfolio_simulation,
)
from src.backtest.multi_engine.report_generator import generate_report
from src.data.aggregator import DataAggregator
from src.data.universe_selection import select_ohlcv_tickers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_adapters(engine_cfg: dict) -> list[EngineAdapter]:
    """Instantiate enabled engine adapters from config."""
    adapters: list[EngineAdapter] = []

    if engine_cfg.get("mas", {}).get("enabled", True):
        top_n = engine_cfg.get("mas", {}).get("top_n", 10)
        adapters.append(MASAdapter(top_n=top_n))

    if engine_cfg.get("koocore_d", {}).get("enabled", True):
        adapters.append(KooCoreAdapter())

    if engine_cfg.get("gemini_stst", {}).get("enabled", True):
        adapters.append(GeminiAdapter())

    return adapters


def _parse_convergence_overrides(
    raw: dict | None,
) -> dict[str, dict[int, float]]:
    """Parse regime_convergence_overrides from YAML config.

    YAML keys like ``2_engines`` are converted to integer keys (2).
    """
    if not raw:
        return {}
    result: dict[str, dict[int, float]] = {}
    for regime, mult_dict in raw.items():
        result[regime] = {
            int(k.split("_")[0]): v for k, v in mult_dict.items()
        }
    return result


# ── Trading Calendar ───────────────────────────────────────────────────────


def build_trading_calendar(
    spy_df: pd.DataFrame, start: date, end: date
) -> list[date]:
    """Extract trading days from SPY OHLCV dates within the given range."""
    if spy_df.empty:
        return []

    dates = pd.to_datetime(spy_df["date"]).dt.date
    mask = (dates >= start) & (dates <= end)
    trading_days = sorted(dates[mask].unique())
    logger.info(
        "Trading calendar: %d days from %s to %s",
        len(trading_days), start, end,
    )
    return trading_days


# ── Point-in-Time Slicing ─────────────────────────────────────────────────


def _slice_to_date(
    price_data: dict[str, pd.DataFrame], as_of: date
) -> dict[str, pd.DataFrame]:
    """Return a shallow copy of price_data with each DataFrame sliced to <= as_of."""
    sliced: dict[str, pd.DataFrame] = {}
    for ticker, df in price_data.items():
        if df is None or df.empty:
            continue
        if not isinstance(df["date"].iloc[0], date):
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"]).dt.date
        mask = df["date"] <= as_of
        sub = df[mask]
        if not sub.empty:
            sliced[ticker] = sub
    return sliced


# ── Main Orchestrator ─────────────────────────────────────────────────────


async def run_multi_engine_backtest(
    start_date: date,
    end_date: date,
    engine_cfg: dict | None = None,
    synthesis_cfg: dict | None = None,
    portfolio_cfg: dict | None = None,
    output_dir: str = "backtest_results/multi_engine",
    universe: list[str] | None = None,
) -> dict:
    """Execute the full multi-engine cross-backtest.

    Steps:
      1. Pre-fetch all OHLCV for the date range + lookback.
      2. Build trading calendar from SPY dates.
      3. For each trading day, slice data, call adapters, synthesize.
      4. Run walk-forward simulation on accumulated signals.
      5. Generate report.
    """
    t0 = time.monotonic()
    engine_cfg = engine_cfg or {}
    synthesis_cfg_dict = synthesis_cfg or {}
    portfolio_cfg_dict = portfolio_cfg or {}

    adapters = _build_adapters(engine_cfg)
    if not adapters:
        return {"error": "No adapters enabled"}

    logger.info("=" * 70)
    logger.info(
        "Multi-Engine Backtest: %s → %s (%d adapters: %s)",
        start_date,
        end_date,
        len(adapters),
        [a.engine_name for a in adapters],
    )
    logger.info("=" * 70)

    # ── Step 1: Pre-fetch OHLCV ──
    max_lookback = max(a.required_lookback_days() for a in adapters)
    fetch_start = start_date - timedelta(days=max_lookback + 30)

    aggregator = DataAggregator()

    # Determine universe
    if universe is None:
        raw_universe = await aggregator.get_universe()
        from src.signals.filter import filter_universe

        filtered = filter_universe(raw_universe)
        tickers = select_ohlcv_tickers(filtered, max_tickers=200)
    else:
        tickers = universe

    # Always include SPY and QQQ for regime classification
    for idx_ticker in ("SPY", "QQQ"):
        if idx_ticker not in tickers:
            tickers.append(idx_ticker)

    logger.info(
        "Pre-fetching OHLCV for %d tickers (%s → %s)…",
        len(tickers), fetch_start, end_date,
    )
    price_data = await aggregator.get_bulk_ohlcv(tickers, fetch_start, end_date)

    # Normalize date columns once
    for ticker in list(price_data.keys()):
        df = price_data[ticker]
        if df is not None and not df.empty and not isinstance(df["date"].iloc[0], date):
            price_data[ticker] = df.assign(
                date=pd.to_datetime(df["date"]).dt.date
            )

    logger.info("Pre-fetch complete: %d tickers with data", len(price_data))

    # ── Step 2: Trading calendar ──
    spy_full = price_data.get("SPY")
    if spy_full is None or spy_full.empty:
        return {"error": "SPY data not available — cannot build trading calendar"}

    trading_days = build_trading_calendar(spy_full, start_date, end_date)
    if not trading_days:
        return {"error": "No trading days in the given range"}

    # ── Step 3: Day-by-day signal generation ──
    synth_config = SynthesisConfig(
        initial_weights=synthesis_cfg_dict.get(
            "initial_weights", {"mas": 1.0, "koocore_d": 1.0, "gemini_stst": 1.0}
        ),
        convergence_multipliers={
            int(k.split("_")[0]): v
            for k, v in synthesis_cfg_dict.get(
                "convergence_multipliers", {"2_engines": 1.5, "3_engines": 2.0}
            ).items()
        } if synthesis_cfg_dict.get("convergence_multipliers") else {2: 1.3, 3: 1.0},
        top_n_per_day=synthesis_cfg_dict.get("top_n_per_day", 5),
        rolling_credibility=synthesis_cfg_dict.get("rolling_credibility", False),
        min_confidence=synthesis_cfg_dict.get("min_confidence", 35.0),
        regime_convergence_overrides=_parse_convergence_overrides(
            synthesis_cfg_dict.get("regime_convergence_overrides")
        ),
    )

    # Rolling credibility tracker (fed with resolved trade outcomes)
    cred_tracker = RollingCredibilityTracker() if synth_config.rolling_credibility else None

    daily_records: list[DailyPickRecord] = []
    total_picks = 0

    for i, day in enumerate(trading_days):
        if i % 50 == 0:
            elapsed = time.monotonic() - t0
            logger.info(
                "Processing day %d/%d (%s) — %.0fs elapsed",
                i + 1, len(trading_days), day, elapsed,
            )

        # Slice price data to <= day (point-in-time)
        pit_data = _slice_to_date(price_data, day)
        spy_pit = pit_data.get("SPY", pd.DataFrame())
        qqq_pit = pit_data.get("QQQ", pd.DataFrame())

        if spy_pit.empty:
            continue

        # Regime classification — BEFORE synthesis so we can pass regime in
        regime = classify_regime_for_date(
            screen_date=day,
            spy_df=spy_pit,
            qqq_df=qqq_pit,
            price_data=pit_data,
        )
        regime_label = regime.regime.value  # "bull", "bear", "choppy"

        # Call all adapters
        engine_picks: dict[str, list[NormalizedPick]] = {}
        all_picks: list[NormalizedPick] = []

        for adapter in adapters:
            try:
                picks = await adapter.generate_picks(
                    screen_date=day,
                    price_data=pit_data,
                    spy_df=spy_pit,
                    qqq_df=qqq_pit,
                )
                engine_picks[adapter.engine_name] = picks
                all_picks.extend(picks)
            except Exception as e:
                logger.warning(
                    "Adapter %s failed on %s: %s", adapter.engine_name, day, e
                )
                engine_picks[adapter.engine_name] = []

        total_picks += len(all_picks)

        # Synthesize — equal-weight (regime-adaptive)
        synth_eq = synthesize_picks(
            all_picks, synth_config, regime=regime_label,
        )

        # Synthesize — credibility-weight (rolling weights if available)
        rolling_weights = None
        if cred_tracker is not None:
            rolling_weights = cred_tracker.get_rolling_weights(
                synth_config.initial_weights
            )
        synth_cred = synthesize_picks(
            all_picks, synth_config,
            rolling_weights=rolling_weights,
            regime=regime_label,
        )

        daily_records.append(DailyPickRecord(
            screen_date=day,
            regime=regime_label,
            engine_picks=engine_picks,
            synthesis_eq=synth_eq,
            synthesis_cred=synth_cred,
        ))

        # Feed resolved outcomes from previous days into credibility tracker.
        # We check trades that entered ~holding_period days ago and are now
        # resolved.  We use a simple heuristic: look up the price change from
        # signal_date+1 to signal_date+median_holding_period for each engine's
        # picks from that day.
        if cred_tracker is not None:
            _feed_credibility_outcomes(
                cred_tracker, daily_records, price_data, day,
                lookback_days=10,
            )

    logger.info(
        "Signal generation complete: %d days, %d total picks",
        len(daily_records), total_picks,
    )

    # ── Step 4: Portfolio simulation ──
    port_config = PortfolioConfig(
        capital=portfolio_cfg_dict.get("capital", 100_000),
        max_positions=portfolio_cfg_dict.get("max_positions", 10),
        slippage_pct=portfolio_cfg_dict.get("slippage_pct", 0.001),
        commission_per_trade=portfolio_cfg_dict.get("commission_per_trade", 1.0),
        holding_periods=portfolio_cfg_dict.get("holding_periods", [5, 10, 15]),
        confidence_sizing=portfolio_cfg_dict.get("confidence_sizing", False),
    )

    track_results = run_portfolio_simulation(daily_records, price_data, port_config)

    # ── Step 5: Generate report ──
    elapsed = time.monotonic() - t0
    report = generate_report(
        track_results=track_results,
        daily_records=daily_records,
        start_date=start_date,
        end_date=end_date,
        engine_names=[a.engine_name for a in adapters],
        config={
            "engines": engine_cfg,
            "synthesis": synthesis_cfg_dict,
            "portfolio": portfolio_cfg_dict,
        },
        elapsed_s=elapsed,
    )

    # Save report
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / f"multi_engine_{start_date}_{end_date}.json"
    report_file.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Report saved to %s", report_file)

    logger.info(
        "Multi-engine backtest complete in %.1f min (%d trading days)",
        elapsed / 60, len(trading_days),
    )

    return report


# ── Credibility feedback ──────────────────────────────────────────────────


def _feed_credibility_outcomes(
    tracker: RollingCredibilityTracker,
    daily_records: list[DailyPickRecord],
    price_data: dict[str, pd.DataFrame],
    current_date: date,
    lookback_days: int = 10,
) -> None:
    """Feed resolved trade outcomes into the rolling credibility tracker.

    For each daily record from *lookback_days* ago, look at each engine's
    raw picks and compute the realized PnL using the median holding period
    (10 days by default).  This gives the tracker a stream of outcomes to
    compute rolling win rates and avg returns.
    """
    if len(daily_records) <= lookback_days:
        return

    target_record = daily_records[-(lookback_days + 1)]
    signal_date = target_record.screen_date

    for engine_name, picks in target_record.engine_picks.items():
        for pick in picks:
            df = price_data.get(pick.ticker)
            if df is None or df.empty:
                continue

            # Find entry (T+1) and exit (T+lookback_days) prices
            dates_col = df["date"]
            after_signal = df[dates_col > signal_date].head(lookback_days + 1)
            if len(after_signal) < 2:
                continue

            entry_price = float(after_signal.iloc[0]["open"])
            exit_price = float(after_signal.iloc[-1]["close"])
            if entry_price <= 0:
                continue

            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            if pick.direction == "short":
                pnl_pct = -pnl_pct

            tracker.record_outcome(engine_name, pnl_pct)


# ── CLI entry point ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Multi-engine cross-backtest orchestrator"
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="backtest_results/multi_engine",
    )
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        dr = cfg.get("date_range", {})
        # CLI --start/--end override config date_range when provided
        start = date.fromisoformat(args.start if args.start else dr.get("start", "2024-01-01"))
        end = date.fromisoformat(args.end if args.end else dr.get("end", "2026-02-17"))
        output_dir = cfg.get("output_dir", args.output_dir)

        asyncio.run(
            run_multi_engine_backtest(
                start_date=start,
                end_date=end,
                engine_cfg=cfg.get("engines"),
                synthesis_cfg=cfg.get("synthesis"),
                portfolio_cfg=cfg.get("portfolio"),
                output_dir=output_dir,
                universe=cfg.get("universe"),
            )
        )
    elif args.start and args.end:
        asyncio.run(
            run_multi_engine_backtest(
                start_date=date.fromisoformat(args.start),
                end_date=date.fromisoformat(args.end),
                output_dir=args.output_dir,
            )
        )
    else:
        parser.error("Provide --config or both --start and --end")


if __name__ == "__main__":
    main()
