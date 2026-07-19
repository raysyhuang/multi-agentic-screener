"""Pure shared exit-simulation engine.

Single source of truth for the bar-by-bar exit walk shared by the live
outcome tracker (``src/output/performance.py::_evaluate_position``) and the
research backtester (``src/research/signal_backtest.py::simulate_trade``).

Before this module existed the two walks were maintained by hand, which let a
class of backtest-vs-live divergences re-creep every time one side was edited
without the other. Both callers now delegate their inner exit loop here.

Design notes
------------
The canonical semantics are the LIVE walk (the ground truth): entry bar is
eligible for stop/target (``check_entry_bar=True``), open-gap fills are modelled
on both stop and target (``gap_through=True``), MFE/MAE are updated BEFORE the
exit checks, and the "phantom exit" guards are load-bearing:

* A trailing stop that *arms this bar* must not enforce its newly-computed level
  on the same bar (daily OHLC can't prove the high preceded the low). It only
  ratchets/enforces from the next bar. An already-active trail still ratchets
  and can exit on the same bar.
* A leg-1 partial fill that happens *this bar* must not raise the breakeven
  floor for the same bar. The floor enforces from the next bar (or immediately
  if leg-1 was filled on a prior bar / preloaded from persistence).

The engine is pure: no I/O, no DB, no async, no pandas. Callers pass plain
``ExitBar`` records and read a plain ``ExitOutcome`` back. Rounding, slippage on
partial legs, pnl-weighting, and date/price stamping stay in the callers so each
side keeps its own (documented) conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable


@dataclass(frozen=True)
class ExitBar:
    """A single OHLC bar. ``date`` is informational (engine keys off index)."""

    date: date
    open: float
    high: float
    low: float
    close: float


@dataclass
class ExitParams:
    """Exit-walk configuration.

    Defaults describe the canonical LIVE walk (``check_entry_bar=True``,
    ``gap_through=True``). The legacy research backtester flips
    ``check_entry_bar`` and ``gap_through`` to ``False`` to approximate its
    historical ``i=1`` / no-open-gap behaviour.
    """

    stop: float
    target: float
    max_hold: int  # expire once this many bars have been held (entry bar = day 1)
    slippage: float
    trail_activate_pct: float = 0.0
    trail_distance_pct: float = 0.0
    partial_tp_target: float = 0.0  # absolute price; 0 disables two-leg
    partial_tp_fraction: float = 0.5  # informational; caller weights the pnl
    time_stop_days: int = 0  # 0 disables
    time_stop_eligible: bool = False  # True only for sniper
    early_exit_mfe_pct: float = 0.0  # 0 disables
    gap_through: bool = True  # model open-gap fills on stop AND target
    check_entry_bar: bool = True  # if False, bars[0] is not eligible for exits
    leg1_prefilled: bool = False  # leg-1 already filled before this walk
    # Optional same-bar tie-breaker. When a daily bar's low breaches the stop AND
    # its high reaches the target (neither via an opening gap), the daily OHLC
    # cannot say which came first, so walk_exit conservatively takes the stop.
    # If set, this callable — (bar_date, stop_level, target_level) -> "stop" |
    # "target" | None — resolves the tie from intraday (minute) data. None means
    # "unresolved"; the conservative stop is kept. walk_exit stays pure: the
    # caller wires the minute-bar lookup (see resolve_first_touch).
    same_bar_resolver: Callable[[date, float, float], str | None] | None = None


@dataclass
class ExitOutcome:
    """Result of :func:`walk_exit`.

    ``pnl_pct`` is the leg-2 / full pnl ``(exit_price - entry) / entry * 100``;
    the caller applies any two-leg weighting. ``mfe_pct`` / ``mae_pct`` are the
    running max-favorable / max-adverse excursions in percent.
    """

    exited: bool
    exit_index: int | None
    exit_price: float | None
    exit_reason: str | None  # stop | trail_stop | target | time_stop | expiry | early_exit
    pnl_pct: float | None
    mfe_pct: float
    mae_pct: float
    leg1_filled: bool
    leg1_index: int | None  # bar index where the partial filled this walk (None if preloaded)
    days_held: int | None


def resolve_first_touch(
    minute_bars: Iterable[tuple[float, float]],
    stop: float,
    target: float,
) -> str | None:
    """Given a day's ordered (low, high) minute bars, return which level was hit
    first: "stop", "target", or None if neither was touched.

    Used to break the same-bar stop-vs-target tie that daily OHLC can't. If a
    single minute bar straddles both levels it stays ambiguous, so we keep the
    conservative "stop" for that minute. Pure and pandas-free; the caller adapts
    its minute frame to (low, high) tuples.
    """
    for low, high in minute_bars:
        hit_stop = low <= stop
        hit_target = high >= target
        if hit_stop:
            return "stop"          # conservative when a single minute hits both
        if hit_target:
            return "target"
    return None


def walk_exit(bars: list[ExitBar], entry_price: float, params: ExitParams) -> ExitOutcome:
    """Walk ``bars`` and resolve the first exit under ``params``.

    ``bars[0]`` is the entry bar; ``entry_price`` is already slippage-adjusted by
    the caller. ``days_held`` starts at 1 on ``bars[0]``.
    """
    use_trailing = params.trail_activate_pct > 0 and params.trail_distance_pct > 0
    use_two_leg = params.partial_tp_target > 0

    high_watermark = entry_price
    trailing_active = False
    mfe = 0.0
    mae = 0.0
    leg1_filled = params.leg1_prefilled
    leg1_index: int | None = None

    def _pnl(price: float) -> float:
        return (price - entry_price) / entry_price * 100

    def _result(index: int, price: float, reason: str) -> ExitOutcome:
        return ExitOutcome(
            exited=True,
            exit_index=index,
            exit_price=price,
            exit_reason=reason,
            pnl_pct=_pnl(price),
            mfe_pct=mfe,
            mae_pct=mae,
            leg1_filled=leg1_filled,
            leg1_index=leg1_index,
            days_held=index + 1,
        )

    for i, bar in enumerate(bars):
        days_held = i + 1

        # MFE/MAE and high-watermark update BEFORE exit checks (live semantics).
        high_watermark = max(high_watermark, bar.high)
        mfe = max(mfe, _pnl(bar.high))
        mae = min(mae, _pnl(bar.low))

        # Arm the trailing stop. A trail that arms THIS bar must not enforce its
        # freshly-computed level on the same bar — only ratchet from the next bar.
        trail_just_activated = False
        if use_trailing and not trailing_active:
            gain_pct = (high_watermark - entry_price) / entry_price * 100
            if gain_pct >= params.trail_activate_pct:
                trailing_active = True
                trail_just_activated = True

        # Effective stop = max(base stop, active trail level).
        effective_stop = params.stop
        if trailing_active and not trail_just_activated:
            trail_stop = high_watermark * (1 - params.trail_distance_pct / 100)
            effective_stop = max(params.stop, trail_stop)

        # Breakeven floor: only if leg-1 was filled on a PRIOR bar (or preloaded).
        # A leg-1 that fills on THIS bar must not raise the floor this same bar.
        if leg1_filled:
            effective_stop = max(effective_stop, entry_price)

        # Leg-1 partial fill (keyed to current-bar high). Deliberately AFTER the
        # breakeven step so a same-bar fill does not raise the floor this bar.
        if use_two_leg and not leg1_filled and bar.high >= params.partial_tp_target:
            leg1_filled = True
            leg1_index = i

        # On the entry bar in legacy (check_entry_bar=False) mode, skip all exit
        # checks — the walk still updated HWM/MFE/MAE/trail-arming above.
        if i == 0 and not params.check_entry_bar:
            continue

        trailed = trailing_active and effective_stop > params.stop
        stop_reason = "trail_stop" if trailed else "stop"

        # Exit checks, in canonical live order (stop before target — conservative).
        # Opening-gap fills are unambiguous (the open already breached the level).
        if params.gap_through and bar.open <= effective_stop:
            return _result(i, bar.open * (1 - params.slippage), stop_reason)
        if params.gap_through and bar.open >= params.target:
            return _result(i, bar.open * (1 - params.slippage), "target")

        stop_hit = bar.low <= effective_stop
        target_hit = bar.high >= params.target

        # Same-bar tie: both intraday levels reached. Use the resolver if given
        # (minute data says which came first); otherwise keep the conservative
        # stop-first assumption.
        if stop_hit and target_hit and params.same_bar_resolver is not None:
            verdict = params.same_bar_resolver(bar.date, effective_stop, params.target)
            if verdict == "target":
                return _result(i, params.target * (1 - params.slippage), "target")
            # "stop" or None (unresolved) → fall through to conservative stop.

        if stop_hit:
            return _result(i, effective_stop * (1 - params.slippage), stop_reason)
        if target_hit:
            return _result(i, params.target * (1 - params.slippage), "target")

        if params.early_exit_mfe_pct > 0 and mfe >= params.early_exit_mfe_pct:
            return _result(i, bar.close * (1 - params.slippage), "early_exit")

        # Sniper time stop — never on the entry bar (i == 0).
        if (
            params.time_stop_eligible
            and i > 0
            and params.time_stop_days > 0
            and days_held >= params.time_stop_days
            and bar.close <= entry_price
        ):
            return _result(i, bar.close * (1 - params.slippage), "time_stop")

        if days_held >= params.max_hold:
            return _result(i, bar.close * (1 - params.slippage), "expiry")

    # No exit — position still open.
    return ExitOutcome(
        exited=False,
        exit_index=None,
        exit_price=None,
        exit_reason=None,
        pnl_pct=None,
        mfe_pct=mfe,
        mae_pct=mae,
        leg1_filled=leg1_filled,
        leg1_index=leg1_index,
        days_held=len(bars) if bars else None,
    )
