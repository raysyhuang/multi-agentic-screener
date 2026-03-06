"""Standalone signal model backtester.

Runs each signal model (breakout, mean_reversion) independently on historical
OHLCV data, simulates every pick with T+1 execution, and reports per-model
performance metrics. Supports parameter grid sweeps.

Usage:
    python -m src.research.signal_backtest --tickers SPY,AAPL,MSFT --years 2
    python -m src.research.signal_backtest --sp500 --years 2
    python -m src.research.signal_backtest --sp500 --years 2 --sweep breakout
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.backtest.metrics import PerformanceMetrics, compute_metrics, deflated_sharpe_ratio
from src.features.technical import (
    compute_all_technical_features,
    compute_rsi2_features,
    latest_features,
)
from src.signals.breakout import BreakoutSignal, score_breakout
from src.signals.mean_reversion import MeanReversionSignal, score_mean_reversion

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_SLIPPAGE_PCT = 0.001
DEFAULT_COMMISSION = 1.0
MIN_HISTORY_BARS = 60  # need at least this many bars for features


# ── Data fetching ─────────────────────────────────────────────────────────────

CACHE_DIR = Path("data/cache/ohlcv")


def _cache_key(tickers: list[str], years: float) -> Path:
    """Generate a cache file path based on ticker list hash and date range."""
    import hashlib
    ticker_hash = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:12]
    end = date.today()
    start = end - timedelta(days=int(years * 365.25))
    return CACHE_DIR / f"ohlcv_{ticker_hash}_{start}_{end}.parquet"


def fetch_ohlcv(tickers: list[str], years: float = 2, no_cache: bool = False) -> dict[str, pd.DataFrame]:
    """Download historical OHLCV data via yfinance, with disk cache.

    Cache is keyed on (sorted tickers hash, start date, end date).
    Pass no_cache=True to force a fresh download.
    """
    cache_path = _cache_key(tickers, years)

    # Try cache first
    if not no_cache and cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        combined = pd.read_parquet(cache_path)
        result: dict[str, pd.DataFrame] = {}
        for ticker, grp in combined.groupby("_ticker"):
            result[ticker] = grp.drop(columns=["_ticker"]).reset_index(drop=True)
        print(f"Loaded {len(result)} tickers from cache")
        return result

    end = date.today()
    start = end - timedelta(days=int(years * 365.25))

    print(f"Downloading {len(tickers)} tickers, {start} to {end} ...")
    raw = yf.download(tickers, start=str(start), end=str(end), auto_adjust=True, threads=True)

    result: dict[str, pd.DataFrame] = {}

    if len(tickers) == 1:
        ticker = tickers[0]
        if raw.empty:
            return result
        df = raw.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        result[ticker] = df
    elif not raw.empty:
        for ticker in tickers:
            try:
                cols = raw.xs(ticker, axis=1, level=1) if isinstance(raw.columns, pd.MultiIndex) else raw
                df = cols.reset_index()
                df.columns = [c.lower() for c in df.columns]
                if "date" not in df.columns and "price" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "date"})
                df["date"] = pd.to_datetime(df["date"]).dt.date
                if df["close"].dropna().empty:
                    continue
                result[ticker] = df.dropna(subset=["close"])
            except (KeyError, ValueError):
                continue

    print(f"Got data for {len(result)}/{len(tickers)} tickers")

    # Save to cache
    if result:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frames = []
        for ticker, df in result.items():
            df_copy = df.copy()
            df_copy["_ticker"] = ticker
            frames.append(df_copy)
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(cache_path, index=False)
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"Cached to {cache_path} ({size_mb:.1f}MB)")

    return result


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia."""
    import urllib.request

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8")
        table = pd.read_html(html)[0]
        return sorted(table["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception as e:
        print(f"Failed to fetch S&P 500 list: {e}")
        return []


# ── Trade simulation (from walk_forward.py, simplified) ──────────────────────

@dataclass
class Trade:
    ticker: str
    model: str
    signal_date: date
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    exit_reason: str
    holding_days: int
    pnl_pct: float
    mfe_pct: float
    mae_pct: float
    score: float
    regime: str


def simulate_trade(
    df: pd.DataFrame,
    signal_date: date,
    stop_loss: float,
    target: float,
    max_hold: int,
    slippage: float = DEFAULT_SLIPPAGE_PCT,
    trail_activate_pct: float = 0.0,
    trail_distance_pct: float = 0.0,
) -> dict | None:
    """Simulate a LONG trade with T+1 open entry and optional trailing stop.

    Trailing stop: once unrealized gain reaches trail_activate_pct, a trailing
    stop activates at (high_watermark * (1 - trail_distance_pct/100)). It only
    ratchets up, never down. Overrides the fixed stop when higher.
    Set both to 0 to disable.
    """
    future = df[df["date"] > signal_date].sort_values("date")
    if len(future) < 2:
        return None

    entry_row = future.iloc[0]
    entry_price = float(entry_row["open"]) * (1 + slippage)
    entry_date = entry_row["date"]

    window = future.iloc[:max_hold + 1]
    mfe = 0.0
    mae = 0.0
    high_watermark = entry_price
    trailing_active = False
    use_trailing = trail_activate_pct > 0 and trail_distance_pct > 0

    for i in range(1, len(window)):
        row = window.iloc[i]
        high, low = float(row["high"]), float(row["low"])

        # Update high watermark (intraday)
        high_watermark = max(high_watermark, high)

        # Activate trailing stop once MFE threshold reached
        if use_trailing and not trailing_active:
            gain_pct = (high_watermark - entry_price) / entry_price * 100
            if gain_pct >= trail_activate_pct:
                trailing_active = True

        # Compute effective stop: max of fixed stop and trailing stop
        effective_stop = stop_loss
        if trailing_active:
            trail_stop = high_watermark * (1 - trail_distance_pct / 100)
            effective_stop = max(stop_loss, trail_stop)

        # Check stop (fixed or trailing)
        if low <= effective_stop:
            exit_price = effective_stop * (1 - slippage)
            pnl = (exit_price - entry_price) / entry_price * 100
            exit_reason = "trail_stop" if trailing_active and effective_stop > stop_loss else "stop"
            return dict(
                entry_date=entry_date, entry_price=round(entry_price, 2),
                exit_date=row["date"], exit_price=round(exit_price, 2),
                exit_reason=exit_reason, holding_days=(row["date"] - entry_date).days,
                pnl_pct=round(pnl, 4), mfe_pct=round(mfe, 4), mae_pct=round(mae, 4),
            )

        if high >= target:
            exit_price = target * (1 - slippage)
            pnl = (exit_price - entry_price) / entry_price * 100
            return dict(
                entry_date=entry_date, entry_price=round(entry_price, 2),
                exit_date=row["date"], exit_price=round(exit_price, 2),
                exit_reason="target", holding_days=(row["date"] - entry_date).days,
                pnl_pct=round(pnl, 4), mfe_pct=round(mfe, 4), mae_pct=round(mae, 4),
            )

        day_best = (high - entry_price) / entry_price * 100
        day_worst = (low - entry_price) / entry_price * 100
        mfe = max(mfe, day_best)
        mae = min(mae, day_worst)

    # Expiry
    last = window.iloc[-1]
    exit_price = float(last["close"]) * (1 - slippage)
    pnl = (exit_price - entry_price) / entry_price * 100
    return dict(
        entry_date=entry_date, entry_price=round(entry_price, 2),
        exit_date=last["date"], exit_price=round(exit_price, 2),
        exit_reason="expiry", holding_days=(last["date"] - entry_date).days,
        pnl_pct=round(pnl, 4), mfe_pct=round(mfe, 4), mae_pct=round(mae, 4),
    )


# ── Signal scanning ──────────────────────────────────────────────────────────

def classify_regime(df: pd.DataFrame) -> str:
    """Simple regime classification from price data."""
    if len(df) < 50:
        return "unknown"
    close = df["close"].astype(float)
    sma50 = close.rolling(50).mean().iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    current = close.iloc[-1]

    if current > sma50 and sma20 > sma50:
        return "bull"
    elif current < sma50 and sma20 < sma50:
        return "bear"
    return "choppy"


def scan_breakout(
    ticker: str,
    df: pd.DataFrame,
    min_score: float = 50.0,
    stop_atr_mult: float = 2.0,
    target_atr_mult: float = 2.0,
    holding_period: int = 7,
) -> list[tuple[date, BreakoutSignal]]:
    """Scan every trading day for breakout signals."""
    if len(df) < MIN_HISTORY_BARS:
        return []

    enriched = compute_all_technical_features(df)
    signals = []

    for i in range(MIN_HISTORY_BARS, len(enriched)):
        window = enriched.iloc[:i + 1].copy()
        feat = latest_features(window)
        feat["close"] = float(window["close"].iloc[-1])

        sig = score_breakout(ticker, window, feat)
        if sig is None:
            continue
        if sig.score < min_score:
            continue

        # Apply parameter overrides
        atr = feat.get("atr_14")
        if atr and atr > 0:
            atr = max(atr, feat["close"] * 0.005)
            sig.stop_loss = round(feat["close"] - stop_atr_mult * atr, 2)
            sig.target_1 = round(feat["close"] + target_atr_mult * atr, 2)
        sig.holding_period = holding_period

        signal_date = window["date"].iloc[-1]
        signals.append((signal_date, sig))

    return signals


def scan_mean_reversion(
    ticker: str,
    df: pd.DataFrame,
    min_score: float = 50.0,
    rsi2_threshold: float = 20.0,
    stop_atr_mult: float = 1.0,
    target_atr_mult: float = 1.0,
    holding_period: int = 3,
) -> list[tuple[date, MeanReversionSignal]]:
    """Scan every trading day for mean reversion signals."""
    if len(df) < MIN_HISTORY_BARS:
        return []

    enriched = compute_all_technical_features(df)
    enriched = compute_rsi2_features(enriched)
    signals = []

    for i in range(MIN_HISTORY_BARS, len(enriched)):
        window = enriched.iloc[:i + 1].copy()
        feat = latest_features(window)
        feat["close"] = float(window["close"].iloc[-1])

        # Apply RSI threshold override
        rsi_2 = feat.get("rsi_2")
        if rsi_2 is None or rsi_2 > rsi2_threshold:
            continue

        sig = score_mean_reversion(ticker, window, feat)
        if sig is None:
            continue
        if sig.score < min_score:
            continue

        # Apply parameter overrides
        atr = feat.get("atr_14")
        close_price = feat["close"]
        if atr and atr > 0:
            atr = max(atr, close_price * 0.005)
            sig.stop_loss = round(close_price - stop_atr_mult * atr, 2)
            sig.target_1 = round(max(feat.get("sma_10", close_price * 1.03) or close_price * 1.03,
                                     close_price + target_atr_mult * atr), 2)
        sig.holding_period = holding_period

        signal_date = window["date"].iloc[-1]
        signals.append((signal_date, sig))

    return signals


# ── Backtest runner ──────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    model: str
    params: dict
    metrics: PerformanceMetrics
    trades: list[Trade]
    by_regime: dict[str, PerformanceMetrics]
    by_exit_reason: dict[str, int]
    avg_mfe_pct: float
    avg_mae_pct: float
    dsr: float  # deflated sharpe ratio (only meaningful in sweeps)


def run_model_backtest(
    model: str,
    price_data: dict[str, pd.DataFrame],
    params: dict | None = None,
) -> ModelResult:
    """Run a single model across all tickers and return results."""
    params = params or {}
    all_trades: list[Trade] = []

    for ticker, df in price_data.items():
        if len(df) < MIN_HISTORY_BARS:
            continue

        # Classify regime from full history
        regime = classify_regime(df)

        # Scan for signals
        if model == "breakout":
            raw_signals = scan_breakout(
                ticker, df,
                min_score=params.get("min_score", 50.0),
                stop_atr_mult=params.get("stop_atr_mult", 2.0),
                target_atr_mult=params.get("target_atr_mult", 2.0),
                holding_period=params.get("holding_period", 7),
            )
        elif model == "mean_reversion":
            raw_signals = scan_mean_reversion(
                ticker, df,
                min_score=params.get("min_score", 50.0),
                rsi2_threshold=params.get("rsi2_threshold", 20.0),
                stop_atr_mult=params.get("stop_atr_mult", 1.0),
                target_atr_mult=params.get("target_atr_mult", 1.0),
                holding_period=params.get("holding_period", 3),
            )
        else:
            raise ValueError(f"Unknown model: {model}")

        # Simulate trades
        trail_activate = params.get("trail_activate_pct", 0.0)
        trail_distance = params.get("trail_distance_pct", 0.0)
        for signal_date, sig in raw_signals:
            result = simulate_trade(
                df, signal_date,
                stop_loss=sig.stop_loss,
                target=sig.target_1,
                max_hold=sig.holding_period,
                trail_activate_pct=trail_activate,
                trail_distance_pct=trail_distance,
            )
            if result is None:
                continue

            all_trades.append(Trade(
                ticker=ticker,
                model=model,
                signal_date=signal_date,
                score=sig.score,
                regime=regime,
                **result,
            ))

    # Compute metrics
    returns = [t.pnl_pct for t in all_trades]
    metrics = compute_metrics(returns)

    # By regime
    by_regime: dict[str, PerformanceMetrics] = {}
    for regime_name in ["bull", "bear", "choppy", "unknown"]:
        regime_returns = [t.pnl_pct for t in all_trades if t.regime == regime_name]
        if regime_returns:
            by_regime[regime_name] = compute_metrics(regime_returns)

    # By exit reason
    by_exit: dict[str, int] = {}
    for t in all_trades:
        by_exit[t.exit_reason] = by_exit.get(t.exit_reason, 0) + 1

    # MFE / MAE
    avg_mfe = float(np.mean([t.mfe_pct for t in all_trades])) if all_trades else 0.0
    avg_mae = float(np.mean([t.mae_pct for t in all_trades])) if all_trades else 0.0

    return ModelResult(
        model=model,
        params=params,
        metrics=metrics,
        trades=all_trades,
        by_regime=by_regime,
        by_exit_reason=by_exit,
        avg_mfe_pct=round(avg_mfe, 4),
        avg_mae_pct=round(avg_mae, 4),
        dsr=0.0,
    )


# ── Parameter sweep ──────────────────────────────────────────────────────────

BREAKOUT_GRID = {
    "min_score": [40, 50, 60],
    "stop_atr_mult": [1.5, 2.0, 2.5],
    "target_atr_mult": [1.5, 2.0, 3.0],
    "holding_period": [5, 7, 10],
}

MEAN_REVERSION_GRID = {
    "min_score": [40, 50, 60],
    "rsi2_threshold": [10, 15, 20],
    "stop_atr_mult": [0.75, 1.0, 1.5],
    "target_atr_mult": [0.75, 1.0, 1.5],
    "holding_period": [3, 5],
}

# Focused grid: fix best base params, sweep trailing stop only
TRAILING_STOP_GRID = {
    "rsi2_threshold": [10],
    "stop_atr_mult": [0.75],
    "target_atr_mult": [1.5],
    "holding_period": [3],
    "min_score": [40],
    "trail_activate_pct": [0.0, 0.5, 0.75, 1.0, 1.5],
    "trail_distance_pct": [0.0, 0.3, 0.5, 0.75, 1.0],
}


def _grid_combos(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_parameter_sweep(
    model: str,
    price_data: dict[str, pd.DataFrame],
    grid_override: dict | None = None,
) -> list[ModelResult]:
    """Sweep parameter grid for a model, compute DSR to penalize overfitting."""
    if grid_override:
        grid = grid_override
    else:
        grid = BREAKOUT_GRID if model == "breakout" else MEAN_REVERSION_GRID
    combos = _grid_combos(grid)
    print(f"\nSweeping {len(combos)} parameter combinations for {model}...")

    results: list[ModelResult] = []
    for i, params in enumerate(combos):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1}/{len(combos)}] {params}")
        result = run_model_backtest(model, price_data, params)
        results.append(result)

    # Compute DSR for each result
    num_trials = len(combos)
    for r in results:
        returns = [t.pnl_pct for t in r.trades]
        if r.metrics.sharpe_ratio > 0 and len(returns) >= 10:
            r.dsr = deflated_sharpe_ratio(r.metrics.sharpe_ratio, num_trials, returns)

    # Sort by DSR (penalized Sharpe)
    results.sort(key=lambda r: r.dsr, reverse=True)
    return results


# ── Reporting ────────────────────────────────────────────────────────────────

def format_model_report(result: ModelResult) -> str:
    """Format a single model result as text."""
    m = result.metrics
    lines = [
        f"\n{'='*60}",
        f"  MODEL: {result.model.upper()}",
        f"  Params: {result.params}" if result.params else "",
        f"{'='*60}",
        f"  Trades:       {m.total_trades}",
        f"  Win Rate:     {m.win_rate:.1%}",
        f"  Avg Return:   {m.avg_return_pct:+.2f}%",
        f"  Median Return:{m.median_return_pct:+.2f}%",
        f"  Total Return: {m.total_return_pct:+.2f}%",
        f"  Expectancy:   {m.expectancy:+.3f}%",
        f"  Sharpe:       {m.sharpe_ratio:.2f}",
        f"  Sortino:      {m.sortino_ratio:.2f}",
        f"  Profit Factor:{m.profit_factor:.2f}",
        f"  Max Drawdown: {m.max_drawdown_pct:.2f}%",
        f"  Avg Win:      {m.avg_win_pct:+.2f}%  |  Avg Loss: {m.avg_loss_pct:+.2f}%",
        f"  Payoff Ratio: {m.payoff_ratio:.2f}",
        f"  Max Consec W: {m.max_consecutive_wins}  |  Max Consec L: {m.max_consecutive_losses}",
        f"  Avg MFE:      {result.avg_mfe_pct:+.2f}%  |  Avg MAE: {result.avg_mae_pct:+.2f}%",
    ]

    if result.dsr > 0:
        lines.append(f"  DSR (p-value): {result.dsr:.4f}  {'PASS' if result.dsr > 0.95 else 'FAIL'}")

    # Exit reasons
    lines.append(f"\n  Exit Reasons:")
    for reason, count in sorted(result.by_exit_reason.items()):
        pct = count / m.total_trades * 100 if m.total_trades else 0
        lines.append(f"    {reason:8s}: {count:4d} ({pct:.1f}%)")

    # By regime
    if result.by_regime:
        lines.append(f"\n  By Regime:")
        for regime, rm in sorted(result.by_regime.items()):
            lines.append(
                f"    {regime:8s}: {rm.total_trades:3d} trades, "
                f"WR={rm.win_rate:.1%}, "
                f"avg={rm.avg_return_pct:+.2f}%, "
                f"Sharpe={rm.sharpe_ratio:.2f}"
            )

    return "\n".join(lines)


def format_sweep_summary(results: list[ModelResult], top_n: int = 10) -> str:
    """Format top N sweep results."""
    lines = [
        f"\n{'='*60}",
        f"  TOP {top_n} PARAMETER COMBINATIONS (by DSR)",
        f"{'='*60}",
    ]

    for i, r in enumerate(results[:top_n]):
        m = r.metrics
        lines.append(
            f"\n  #{i+1}  DSR={r.dsr:.4f}  Sharpe={m.sharpe_ratio:.2f}  "
            f"WR={m.win_rate:.1%}  N={m.total_trades}  "
            f"Avg={m.avg_return_pct:+.2f}%  PF={m.profit_factor:.2f}"
        )
        lines.append(f"      Params: {r.params}")

    # Also show worst
    if len(results) > top_n:
        worst = results[-1]
        wm = worst.metrics
        lines.append(
            f"\n  WORST: DSR={worst.dsr:.4f}  Sharpe={wm.sharpe_ratio:.2f}  "
            f"WR={wm.win_rate:.1%}  N={wm.total_trades}  "
            f"Avg={wm.avg_return_pct:+.2f}%"
        )
        lines.append(f"      Params: {worst.params}")

    return "\n".join(lines)


def save_trades_csv(trades: list[Trade], path: Path) -> None:
    """Save trade log to CSV for further analysis."""
    if not trades:
        return
    rows = [
        {
            "ticker": t.ticker, "model": t.model, "signal_date": t.signal_date,
            "entry_date": t.entry_date, "entry_price": t.entry_price,
            "exit_date": t.exit_date, "exit_price": t.exit_price,
            "exit_reason": t.exit_reason, "holding_days": t.holding_days,
            "pnl_pct": t.pnl_pct, "mfe_pct": t.mfe_pct, "mae_pct": t.mae_pct,
            "score": t.score, "regime": t.regime,
        }
        for t in trades
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Trade log saved to {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Signal model backtester")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers")
    parser.add_argument("--sp500", action="store_true", help="Use S&P 500 universe")
    parser.add_argument("--years", type=float, default=2.0, help="Years of history (default: 2)")
    parser.add_argument("--models", type=str, default="breakout,mean_reversion",
                        help="Comma-separated models to test")
    parser.add_argument("--sweep", type=str, help="Model to run parameter sweep on")
    parser.add_argument("--sweep-trailing", action="store_true",
                        help="Sweep trailing stop params (uses best mean_reversion base)")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh data download")
    parser.add_argument("--output-dir", type=str, default="outputs/research",
                        help="Directory for output files")
    args = parser.parse_args()

    # Resolve tickers
    if args.sp500:
        tickers = get_sp500_tickers()
        if not tickers:
            print("Failed to get S&P 500 tickers")
            sys.exit(1)
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        print("Must specify --tickers or --sp500")
        sys.exit(1)

    # Fetch data
    t0 = time.time()
    price_data = fetch_ohlcv(tickers, years=args.years, no_cache=args.no_cache)
    if not price_data:
        print("No data fetched")
        sys.exit(1)
    print(f"Data fetch: {time.time() - t0:.1f}s")

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Trailing stop sweep mode
    if args.sweep_trailing:
        t1 = time.time()
        results = run_parameter_sweep("mean_reversion", price_data, grid_override=TRAILING_STOP_GRID)
        print(f"Trailing stop sweep completed in {time.time() - t1:.1f}s")
        print(format_sweep_summary(results))

        if results:
            print(format_model_report(results[0]))
            save_trades_csv(results[0].trades, out_dir / "sweep_trailing_best_trades.csv")

            sweep_rows = []
            for r in results:
                row = {**r.params, "dsr": r.dsr}
                row.update(asdict(r.metrics))
                sweep_rows.append(row)
            pd.DataFrame(sweep_rows).to_csv(out_dir / "sweep_trailing_all.csv", index=False)
            print(f"Full sweep saved to {out_dir / 'sweep_trailing_all.csv'}")

        return

    # Parameter sweep mode
    if args.sweep:
        t1 = time.time()
        results = run_parameter_sweep(args.sweep, price_data)
        print(f"Sweep completed in {time.time() - t1:.1f}s")
        print(format_sweep_summary(results))

        if results:
            print(format_model_report(results[0]))  # best result details
            save_trades_csv(results[0].trades, out_dir / f"sweep_{args.sweep}_best_trades.csv")

            # Save full sweep results
            sweep_rows = []
            for r in results:
                row = {**r.params, "dsr": r.dsr}
                row.update(asdict(r.metrics))
                sweep_rows.append(row)
            pd.DataFrame(sweep_rows).to_csv(out_dir / f"sweep_{args.sweep}_all.csv", index=False)
            print(f"Full sweep saved to {out_dir / f'sweep_{args.sweep}_all.csv'}")

        return

    # Normal mode: run each model with defaults
    models = [m.strip() for m in args.models.split(",")]
    for model in models:
        t1 = time.time()
        print(f"\nRunning {model}...")
        result = run_model_backtest(model, price_data)
        print(f"  Completed in {time.time() - t1:.1f}s")
        print(format_model_report(result))
        save_trades_csv(result.trades, out_dir / f"backtest_{model}_trades.csv")


if __name__ == "__main__":
    main()
