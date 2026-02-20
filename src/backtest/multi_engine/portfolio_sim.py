"""Portfolio-level trade simulation for the multi-engine backtest.

Runs **7 parallel simulation tracks**:
  1. MAS standalone
  2. KooCore-D standalone
  3. Gemini STST standalone
  4. Equal-weight synthesis
  5. Credibility-weight synthesis
  6. Confidence-sized synthesis (PnL scaled by conviction)
  7. SPY buy-and-hold benchmark

Reuses :func:`src.backtest.walk_forward._simulate_trade` for T+1 entry,
stop/target/expiry exits, and MFE/MAE tracking.

Position management: each track maintains a set of open positions keyed by
(ticker, signal_date).  A position is considered "open" from signal_date
until the exit_date of the *longest* holding-period variant.  Once a
position exits, its slot is freed for new picks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta

import pandas as pd

from src.backtest.walk_forward import _simulate_trade
from src.backtest.multi_engine.adapters.base import NormalizedPick
from src.backtest.multi_engine.synthesizer import SynthesisPick

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Simulation parameters."""

    capital: float = 100_000.0
    max_positions: int = 10
    max_per_engine: int = 5
    risk_per_trade_pct: float = 2.0
    slippage_pct: float = 0.001
    commission_per_trade: float = 1.0
    holding_periods: list[int] = field(default_factory=lambda: [5, 10, 15])
    confidence_sizing: bool = False


@dataclass
class SimulatedTrade:
    """Record of a single simulated trade."""

    ticker: str
    engine_name: str
    strategy: str
    track: str  # "mas", "koocore_d", "gemini_stst", "eq_synth", "cred_synth", "sized_synth"
    signal_date: date
    entry_date: date | None
    entry_price: float
    exit_date: date | None
    exit_price: float | None
    exit_reason: str | None
    holding_days: int
    holding_period: int  # the configured period used
    pnl_pct: float
    pnl_after_costs: float
    mfe_pct: float
    mae_pct: float
    confidence: float
    direction: str

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("signal_date", "entry_date", "exit_date"):
            if d[k] is not None:
                d[k] = str(d[k])
        return d


@dataclass
class TrackResult:
    """Aggregated result for one simulation track."""

    track_name: str
    trades: list[SimulatedTrade]
    pnl_series: list[float]
    equity_curve: list[dict]  # [{date, cumulative_pnl_pct}]


@dataclass
class DailyPickRecord:
    """All picks and regime for a single trading day."""

    screen_date: date
    regime: str
    engine_picks: dict[str, list[NormalizedPick]]  # engine_name -> picks
    synthesis_eq: list[SynthesisPick]  # equal-weight synthesis
    synthesis_cred: list[SynthesisPick]  # credibility-weight synthesis
    synthesis_regime_gated: list[SynthesisPick] = field(default_factory=list)  # regime-weighted


@dataclass
class _OpenPosition:
    """Tracks an open position slot so we know when it frees up."""

    ticker: str
    signal_date: date
    latest_exit_date: date  # exit date of the longest holding period variant


def run_portfolio_simulation(
    daily_records: list[DailyPickRecord],
    price_data: dict[str, pd.DataFrame],
    config: PortfolioConfig,
) -> dict[str, TrackResult]:
    """Simulate all tracks across the full set of daily pick records.

    Returns:
        Dict of track_name -> TrackResult.
    """
    tracks = {
        "mas": TrackResult("mas", [], [], []),
        "koocore_d": TrackResult("koocore_d", [], [], []),
        "gemini_stst": TrackResult("gemini_stst", [], [], []),
        "eq_synth": TrackResult("eq_synth", [], [], []),
        "cred_synth": TrackResult("cred_synth", [], [], []),
        "regime_gated": TrackResult("regime_gated", [], [], []),
        "spy_benchmark": TrackResult("spy_benchmark", [], [], []),
    }

    # Add confidence-sized track if enabled
    if config.confidence_sizing:
        tracks["sized_synth"] = TrackResult("sized_synth", [], [], [])

    # Simulate per-engine tracks
    for engine_name in ("mas", "koocore_d", "gemini_stst"):
        _simulate_engine_track(
            engine_name, daily_records, price_data, config, tracks[engine_name]
        )

    # Simulate synthesis tracks
    _simulate_synthesis_track(
        "eq_synth", daily_records, price_data, config, tracks["eq_synth"],
        use_eq=True,
    )
    _simulate_synthesis_track(
        "cred_synth", daily_records, price_data, config, tracks["cred_synth"],
        use_eq=False,
    )

    # Regime-gated synthesis track (eq picks + regime strategy weighting)
    _simulate_synthesis_track(
        "regime_gated", daily_records, price_data, config,
        tracks["regime_gated"],
        use_eq=True,
        use_regime_gated=True,
    )

    # Confidence-sized synthesis track (uses eq picks, scales PnL by confidence)
    if config.confidence_sizing:
        _simulate_synthesis_track(
            "sized_synth", daily_records, price_data, config,
            tracks["sized_synth"],
            use_eq=True,
            confidence_sizing=True,
        )

    # SPY buy-and-hold benchmark
    _simulate_spy_benchmark(daily_records, price_data, tracks["spy_benchmark"])

    for name, track in tracks.items():
        logger.info(
            "Track %s: %d trades, total PnL=%.2f%%",
            name,
            len(track.trades),
            sum(track.pnl_series),
        )

    return tracks


# ── Position management helpers ───────────────────────────────────────────


def _expire_positions(
    open_positions: dict[str, _OpenPosition], current_date: date
) -> None:
    """Remove positions whose latest exit date is before *current_date*."""
    expired = [
        ticker
        for ticker, pos in open_positions.items()
        if pos.latest_exit_date < current_date
    ]
    for ticker in expired:
        del open_positions[ticker]


# ── Per-engine simulation ─────────────────────────────────────────────────


def _simulate_engine_track(
    engine_name: str,
    daily_records: list[DailyPickRecord],
    price_data: dict[str, pd.DataFrame],
    config: PortfolioConfig,
    track: TrackResult,
) -> None:
    """Simulate trades for a single engine's picks with position management."""
    open_positions: dict[str, _OpenPosition] = {}  # ticker -> _OpenPosition

    for record in daily_records:
        # Free slots for positions that have exited before this date
        _expire_positions(open_positions, record.screen_date)

        picks = record.engine_picks.get(engine_name, [])
        for pick in picks:
            # Skip if already holding this ticker
            if pick.ticker in open_positions:
                continue
            # Respect max concurrent positions
            if len(open_positions) >= config.max_positions:
                break

            trades = _simulate_pick(
                pick=pick,
                signal_date=record.screen_date,
                price_data=price_data,
                config=config,
                track_name=engine_name,
            )
            if not trades:
                continue

            for t in trades:
                track.trades.append(t)
                track.pnl_series.append(t.pnl_after_costs)

            # Track position using the latest exit date across holding periods
            latest_exit = max(
                (t.exit_date for t in trades if t.exit_date is not None),
                default=record.screen_date + timedelta(days=max(config.holding_periods)),
            )
            open_positions[pick.ticker] = _OpenPosition(
                ticker=pick.ticker,
                signal_date=record.screen_date,
                latest_exit_date=latest_exit,
            )

    # Build equity curve
    track.equity_curve = _build_equity_curve(track.trades)


def _simulate_synthesis_track(
    track_name: str,
    daily_records: list[DailyPickRecord],
    price_data: dict[str, pd.DataFrame],
    config: PortfolioConfig,
    track: TrackResult,
    use_eq: bool,
    confidence_sizing: bool = False,
    use_regime_gated: bool = False,
) -> None:
    """Simulate trades from synthesis picks with position management."""
    open_positions: dict[str, _OpenPosition] = {}

    for record in daily_records:
        _expire_positions(open_positions, record.screen_date)

        if use_regime_gated:
            # When regime gate blocks all picks, skip the day (don't fall back)
            synth_picks = record.synthesis_regime_gated
        else:
            synth_picks = record.synthesis_eq if use_eq else record.synthesis_cred
        for sp in synth_picks:
            if sp.ticker in open_positions:
                continue
            if len(open_positions) >= config.max_positions:
                break

            # Convert SynthesisPick to NormalizedPick for simulation
            pick = NormalizedPick(
                ticker=sp.ticker,
                engine_name=",".join(sp.engines),
                strategy=",".join(sp.strategies),
                entry_price=sp.entry_price,
                stop_loss=sp.stop_loss,
                target_price=sp.target_price,
                confidence=sp.combined_score,
                holding_period_days=sp.holding_period_days,
                direction=sp.direction,
            )

            trades = _simulate_pick(
                pick=pick,
                signal_date=record.screen_date,
                price_data=price_data,
                config=config,
                track_name=track_name,
            )
            if not trades:
                continue

            for t in trades:
                if confidence_sizing:
                    # Scale PnL by confidence: 0.5x at conf=0, 1.5x at conf=100
                    scale = 0.5 + (sp.avg_weighted_confidence / 100.0)
                    scale = max(0.5, min(1.5, scale))
                    t = SimulatedTrade(
                        ticker=t.ticker,
                        engine_name=t.engine_name,
                        strategy=t.strategy,
                        track=t.track,
                        signal_date=t.signal_date,
                        entry_date=t.entry_date,
                        entry_price=t.entry_price,
                        exit_date=t.exit_date,
                        exit_price=t.exit_price,
                        exit_reason=t.exit_reason,
                        holding_days=t.holding_days,
                        holding_period=t.holding_period,
                        pnl_pct=round(t.pnl_pct * scale, 4),
                        pnl_after_costs=round(t.pnl_after_costs * scale, 4),
                        mfe_pct=t.mfe_pct,
                        mae_pct=t.mae_pct,
                        confidence=t.confidence,
                        direction=t.direction,
                    )
                track.trades.append(t)
                track.pnl_series.append(t.pnl_after_costs)

            latest_exit = max(
                (t.exit_date for t in trades if t.exit_date is not None),
                default=record.screen_date + timedelta(days=max(config.holding_periods)),
            )
            open_positions[sp.ticker] = _OpenPosition(
                ticker=sp.ticker,
                signal_date=record.screen_date,
                latest_exit_date=latest_exit,
            )

    track.equity_curve = _build_equity_curve(track.trades)


# ── Single-pick simulation ────────────────────────────────────────────────


def _simulate_pick(
    pick: NormalizedPick,
    signal_date: date,
    price_data: dict[str, pd.DataFrame],
    config: PortfolioConfig,
    track_name: str,
) -> list[SimulatedTrade]:
    """Simulate a single pick across all holding periods."""
    df = price_data.get(pick.ticker)
    if df is None or df.empty:
        return []

    # Ensure date column is date type
    df = df.copy()
    if "date" in df.columns and not df.empty:
        if not isinstance(df["date"].iloc[0], date):
            df["date"] = pd.to_datetime(df["date"]).dt.date

    stop = pick.stop_loss if pick.stop_loss is not None else pick.entry_price * 0.95
    target = pick.target_price if pick.target_price is not None else pick.entry_price * 1.10

    trades: list[SimulatedTrade] = []

    for period in config.holding_periods:
        result = _simulate_trade(
            df=df,
            signal_date=signal_date,
            direction=pick.direction,
            stop_loss=stop,
            target=target,
            max_holding_days=period,
            slippage_pct=config.slippage_pct,
            commission=config.commission_per_trade,
            entry_price_hint=pick.entry_price,
        )
        if result is None:
            continue

        trades.append(SimulatedTrade(
            ticker=pick.ticker,
            engine_name=pick.engine_name,
            strategy=pick.strategy,
            track=track_name,
            signal_date=signal_date,
            entry_date=result["entry_date"],
            entry_price=result["entry_price"],
            exit_date=result["exit_date"],
            exit_price=result["exit_price"],
            exit_reason=result["exit_reason"],
            holding_days=result["holding_days"],
            holding_period=period,
            pnl_pct=result["pnl_pct"],
            pnl_after_costs=result["pnl_after_costs"],
            mfe_pct=result["max_favorable_excursion"],
            mae_pct=result["max_adverse_excursion"],
            confidence=pick.confidence,
            direction=pick.direction,
        ))

    return trades


# ── SPY benchmark ─────────────────────────────────────────────────────────


def _simulate_spy_benchmark(
    daily_records: list[DailyPickRecord],
    price_data: dict[str, pd.DataFrame],
    track: TrackResult,
) -> None:
    """Simulate SPY buy-and-hold over the backtest period."""
    spy_df = price_data.get("SPY")
    if spy_df is None or spy_df.empty:
        return

    spy_df = spy_df.copy()
    if not isinstance(spy_df["date"].iloc[0], date):
        spy_df["date"] = pd.to_datetime(spy_df["date"]).dt.date

    if not daily_records:
        return

    start_date = daily_records[0].screen_date
    end_date = daily_records[-1].screen_date

    # Filter to backtest period
    mask = (spy_df["date"] >= start_date) & (spy_df["date"] <= end_date)
    period_df = spy_df[mask].sort_values("date")

    if period_df.empty or len(period_df) < 2:
        return

    entry_price = float(period_df.iloc[0]["open"])
    daily_returns = period_df["close"].astype(float).pct_change().dropna() * 100

    cumulative = 0.0
    for i, (idx, row) in enumerate(period_df.iterrows()):
        if i == 0:
            continue
        pnl = float(daily_returns.iloc[i - 1]) if i - 1 < len(daily_returns) else 0
        track.pnl_series.append(pnl)
        cumulative += pnl
        track.equity_curve.append({
            "date": str(row["date"]),
            "cumulative_pnl_pct": round(cumulative, 2),
        })


# ── Equity curve builder ──────────────────────────────────────────────────


def _build_equity_curve(trades: list[SimulatedTrade]) -> list[dict]:
    """Build cumulative equity curve from trade results, sorted by exit date."""
    if not trades:
        return []

    sorted_trades = sorted(trades, key=lambda t: t.exit_date or t.signal_date)
    cumulative = 0.0
    curve = []
    for t in sorted_trades:
        cumulative += t.pnl_after_costs
        curve.append({
            "date": str(t.exit_date or t.signal_date),
            "cumulative_pnl_pct": round(cumulative, 2),
        })
    return curve
