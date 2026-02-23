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
import re
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
    SynthesisPick,
    synthesize_picks,
)
from src.engines.regime_gate import apply_regime_strategy_gate
from src.backtest.multi_engine.portfolio_sim import (
    DailyPickRecord,
    PortfolioConfig,
    run_portfolio_simulation,
)
from src.backtest.multi_engine.report_generator import generate_report
from src.backtest.multi_engine.persistence import persist_multi_engine_backtest_report
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


def _slugify_label(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_").lower()
    return slug[:40] or "candidate"


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
    regime_gate_backtest_cfg: dict | None = None,
    output_dir: str = "backtest_results/multi_engine",
    universe: list[str] | None = None,
    tuning_meta: dict | None = None,
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
    regime_gate_cfg = regime_gate_backtest_cfg or {}

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
        rolling_credibility_window=synthesis_cfg_dict.get(
            "rolling_credibility_window", 20
        ),
        rolling_credibility_min_trades=synthesis_cfg_dict.get(
            "rolling_credibility_min_trades", 10
        ),
        rolling_credibility_weight_floor=synthesis_cfg_dict.get(
            "rolling_credibility_weight_floor", 0.3
        ),
        rolling_credibility_weight_cap=synthesis_cfg_dict.get(
            "rolling_credibility_weight_cap", 2.5
        ),
        min_confidence=synthesis_cfg_dict.get("min_confidence", 35.0),
        diversity_enabled=synthesis_cfg_dict.get("diversity_enabled", True),
        diversity_boost_multi_category=synthesis_cfg_dict.get(
            "diversity_boost_multi_category", 1.15
        ),
        diversity_penalty_homogeneous_3plus=synthesis_cfg_dict.get(
            "diversity_penalty_homogeneous_3plus", 0.70
        ),
        regime_convergence_overrides=_parse_convergence_overrides(
            synthesis_cfg_dict.get("regime_convergence_overrides")
        ),
    )

    # Rolling credibility tracker (fed with resolved trade outcomes)
    cred_tracker = None
    if synth_config.rolling_credibility:
        cred_tracker = RollingCredibilityTracker(
            window=synth_config.rolling_credibility_window,
            min_trades=synth_config.rolling_credibility_min_trades,
            weight_floor=synth_config.rolling_credibility_weight_floor,
            weight_cap=synth_config.rolling_credibility_weight_cap,
        )

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

        # Regime-gated synthesis: apply strategy weighting + bear blocking
        synth_regime_gated = _apply_regime_gate_to_synthesis(
            synth_eq,
            regime_label,
            overrides=regime_gate_cfg if regime_gate_cfg.get("enabled", True) else {},
        )

        daily_records.append(DailyPickRecord(
            screen_date=day,
            regime=regime_label,
            engine_picks=engine_picks,
            synthesis_eq=synth_eq,
            synthesis_cred=synth_cred,
            synthesis_regime_gated=synth_regime_gated,
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
            "regime_gate_backtest": regime_gate_cfg,
            "tuning_meta": tuning_meta or {},
        },
        elapsed_s=elapsed,
    )

    # Save report
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename_label = None
    if tuning_meta and isinstance(tuning_meta, dict):
        raw_label = tuning_meta.get("config_label")
        if raw_label:
            filename_label = _slugify_label(str(raw_label))
    report_name = (
        f"multi_engine_{start_date}_{end_date}_{filename_label}.json"
        if filename_label
        else f"multi_engine_{start_date}_{end_date}.json"
    )
    report_file = out_path / report_name
    report_file.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Report saved to %s", report_file)
    persisted = await persist_multi_engine_backtest_report(report, report_file.name)
    if persisted:
        logger.info("Persisted backtest report to database: %s", report_file.name)

    logger.info(
        "Multi-engine backtest complete in %.1f min (%d trading days)",
        elapsed / 60, len(trading_days),
    )

    return report


# ── Regime gating for backtest A/B ────────────────────────────────────


def _apply_regime_gate_to_synthesis(
    synth_picks: list[SynthesisPick],
    regime: str,
    overrides: dict | None = None,
) -> list[SynthesisPick]:
    """Apply regime strategy gate to synthesized picks for backtest comparison.

    Converts SynthesisPick → dict, applies the gate, then maps back to
    SynthesisPick with updated scores.
    """
    if not synth_picks:
        return []

    pick_dicts = []
    for sp in synth_picks:
        d = sp.to_dict()
        d["strategy_tags"] = sp.strategies
        pick_dicts.append(d)

    gated, _meta = apply_regime_strategy_gate(
        pick_dicts, regime=regime, overrides=overrides or {}
    )

    result: list[SynthesisPick] = []
    for gd in gated:
        ticker = gd["ticker"]
        # Find the original SynthesisPick for non-score fields
        original = next((sp for sp in synth_picks if sp.ticker == ticker), None)
        if original is None:
            continue
        result.append(SynthesisPick(
            ticker=ticker,
            combined_score=gd.get("combined_score", original.combined_score),
            avg_weighted_confidence=gd.get("avg_weighted_confidence", original.avg_weighted_confidence),
            convergence_multiplier=original.convergence_multiplier,
            diversity_multiplier=original.diversity_multiplier,
            engine_count=original.engine_count,
            engines=original.engines,
            strategies=original.strategies,
            entry_price=original.entry_price,
            stop_loss=original.stop_loss,
            target_price=original.target_price,
            holding_period_days=original.holding_period_days,
            direction=original.direction,
            source_picks=original.source_picks,
        ))

    return result


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


# ── Multi-Horizon Backtest ────────────────────────────────────────────────


async def run_multi_horizon_backtest(
    end_date: date,
    backtest_years: list[int],
    engine_cfg: dict | None = None,
    synthesis_cfg: dict | None = None,
    portfolio_cfg: dict | None = None,
    regime_gate_backtest_cfg: dict | None = None,
    output_dir: str = "backtest_results/multi_engine",
    universe: list[str] | None = None,
    tuning_meta: dict | None = None,
) -> dict:
    """Run backtests across multiple time horizons and compare results.

    For each entry in *backtest_years*, a separate multi-engine backtest is
    executed with ``start_date = end_date - N years``.  The individual reports
    are collected and a cross-horizon comparison is appended.

    Returns a combined report with per-horizon results and comparison.
    """
    from dateutil.relativedelta import relativedelta

    t0 = time.monotonic()
    horizon_reports: dict[str, dict] = {}

    for years in sorted(backtest_years):
        start = end_date - relativedelta(years=years)
        label = f"{years}y"
        horizon_dir = f"{output_dir}/{label}"
        logger.info("=" * 70)
        logger.info("Starting %s horizon backtest: %s → %s", label, start, end_date)
        logger.info("=" * 70)

        report = await run_multi_engine_backtest(
            start_date=start,
            end_date=end_date,
            engine_cfg=engine_cfg,
            synthesis_cfg=synthesis_cfg,
            portfolio_cfg=portfolio_cfg,
            regime_gate_backtest_cfg=regime_gate_backtest_cfg,
            output_dir=horizon_dir,
            universe=universe,
            tuning_meta=tuning_meta,
        )
        horizon_reports[label] = report

    # Build cross-horizon comparison
    comparison = _compare_horizons(horizon_reports)

    elapsed = time.monotonic() - t0
    combined = {
        "run_date": str(date.today()),
        "end_date": str(end_date),
        "backtest_years": backtest_years,
        "elapsed_total_s": round(elapsed, 1),
        "horizons": horizon_reports,
        "cross_horizon_comparison": comparison,
    }

    # Save combined report
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    combined_file = out_path / f"multi_horizon_{'_'.join(str(y) for y in backtest_years)}y_{end_date}.json"
    combined_file.write_text(json.dumps(combined, indent=2, default=str))
    logger.info("Combined multi-horizon report saved to %s", combined_file)

    return combined


def _compare_horizons(horizon_reports: dict[str, dict]) -> dict:
    """Build a cross-horizon comparison from individual backtest reports.

    Extracts key metrics from each horizon's synthesis and per-engine results
    and presents them side-by-side for easy comparison.
    """
    if not horizon_reports:
        return {"error": "No horizon reports to compare"}

    # Per-engine comparison across horizons
    engine_comparison: dict[str, dict[str, dict]] = {}
    synthesis_comparison: dict[str, dict[str, dict]] = {}

    for horizon, report in sorted(horizon_reports.items()):
        # Per-engine metrics
        per_engine = report.get("per_engine", {})
        for engine, data in per_engine.items():
            summary = data.get("summary", {})
            engine_comparison.setdefault(engine, {})[horizon] = {
                "total_trades": summary.get("total_trades", 0),
                "win_rate": summary.get("win_rate", 0),
                "avg_return_pct": summary.get("avg_return_pct", 0),
                "sharpe_ratio": summary.get("sharpe_ratio", 0),
                "sortino_ratio": summary.get("sortino_ratio", 0),
                "max_drawdown_pct": summary.get("max_drawdown_pct", 0),
                "profit_factor": summary.get("profit_factor", 0),
                "calmar_ratio": summary.get("calmar_ratio", 0),
                "total_return_pct": summary.get("total_return_pct", 0),
            }

        # Synthesis metrics
        synthesis = report.get("synthesis", {})
        for track_name in ("equal_weight", "credibility_weight", "regime_gated"):
            track_data = synthesis.get(track_name, {})
            summary = track_data.get("summary", {})
            if summary:
                synthesis_comparison.setdefault(track_name, {})[horizon] = {
                    "total_trades": summary.get("total_trades", 0),
                    "win_rate": summary.get("win_rate", 0),
                    "avg_return_pct": summary.get("avg_return_pct", 0),
                    "sharpe_ratio": summary.get("sharpe_ratio", 0),
                    "sortino_ratio": summary.get("sortino_ratio", 0),
                    "max_drawdown_pct": summary.get("max_drawdown_pct", 0),
                    "profit_factor": summary.get("profit_factor", 0),
                    "calmar_ratio": summary.get("calmar_ratio", 0),
                    "total_return_pct": summary.get("total_return_pct", 0),
                }

    # Stability analysis: how consistent are metrics across horizons?
    stability = _compute_stability(engine_comparison, synthesis_comparison)

    return {
        "engine_comparison": engine_comparison,
        "synthesis_comparison": synthesis_comparison,
        "stability_analysis": stability,
    }


def _compute_stability(
    engine_comparison: dict[str, dict[str, dict]],
    synthesis_comparison: dict[str, dict[str, dict]],
) -> dict:
    """Measure metric stability across horizons.

    A strategy that performs consistently across 1y, 3y, and 5y windows is
    more trustworthy than one that only shines in a single window.

    Returns coefficient of variation (CV) for key metrics per engine/track.
    Lower CV = more stable.
    """
    import numpy as np

    result: dict[str, dict] = {}
    key_metrics = ["win_rate", "sharpe_ratio", "avg_return_pct", "profit_factor"]

    all_sources = {**engine_comparison, **{f"synth_{k}": v for k, v in synthesis_comparison.items()}}

    for source_name, horizons in all_sources.items():
        if len(horizons) < 2:
            continue

        stability: dict[str, float | str] = {}
        for metric in key_metrics:
            values = [h.get(metric, 0) for h in horizons.values()]
            arr = np.array(values)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            cv = abs(std / mean) if mean != 0 else float("inf")
            stability[metric] = {
                "values": {h: v for h, v in zip(horizons.keys(), values)},
                "mean": round(mean, 4),
                "std": round(std, 4),
                "cv": round(cv, 4),
                "stable": cv < 0.5,  # CV < 50% = reasonably stable
            }

        result[source_name] = stability

    return result


# ── CLI entry point ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Multi-engine cross-backtest orchestrator"
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--backtest-years",
        type=str,
        help="Comma-separated list of horizon years (e.g. 1,3,5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="backtest_results/multi_engine",
    )
    args = parser.parse_args()

    # Parse --backtest-years override
    cli_years = None
    if args.backtest_years:
        cli_years = [int(y.strip()) for y in args.backtest_years.split(",")]

    if args.config:
        cfg = load_config(args.config)
        dr = cfg.get("date_range", {})
        end = date.fromisoformat(args.end if args.end else dr.get("end", "2026-02-17"))
        output_dir = cfg.get("output_dir", args.output_dir)

        # Determine backtest years: CLI flag > config > None (single run)
        backtest_years = cli_years or cfg.get("backtest_years")

        if backtest_years:
            asyncio.run(
                run_multi_horizon_backtest(
                    end_date=end,
                    backtest_years=backtest_years,
                    engine_cfg=cfg.get("engines"),
                    synthesis_cfg=cfg.get("synthesis"),
                    portfolio_cfg=cfg.get("portfolio"),
                    regime_gate_backtest_cfg=cfg.get("regime_gate_backtest"),
                    output_dir=output_dir,
                    universe=cfg.get("universe"),
                    tuning_meta=cfg.get("tuning_meta"),
                )
            )
        else:
            # Single date-range backtest (original behavior)
            start = date.fromisoformat(
                args.start if args.start else dr.get("start", "2024-01-01")
            )
            asyncio.run(
                run_multi_engine_backtest(
                    start_date=start,
                    end_date=end,
                    engine_cfg=cfg.get("engines"),
                    synthesis_cfg=cfg.get("synthesis"),
                    portfolio_cfg=cfg.get("portfolio"),
                    regime_gate_backtest_cfg=cfg.get("regime_gate_backtest"),
                    output_dir=output_dir,
                    universe=cfg.get("universe"),
                    tuning_meta=cfg.get("tuning_meta"),
                )
            )
    elif cli_years and args.end:
        asyncio.run(
            run_multi_horizon_backtest(
                end_date=date.fromisoformat(args.end),
                backtest_years=cli_years,
                output_dir=args.output_dir,
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
        parser.error("Provide --config, both --start and --end, or --backtest-years with --end")


if __name__ == "__main__":
    main()
