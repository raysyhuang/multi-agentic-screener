"""Uniform pick interface for all engine adapters.

Every adapter normalizes its engine's output into NormalizedPick objects so the
orchestrator, synthesizer, and portfolio simulator speak a single language.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import date

import pandas as pd


@dataclass
class NormalizedPick:
    """Engine-agnostic representation of a trade signal."""

    ticker: str
    engine_name: str
    strategy: str  # "breakout", "mean_reversion", "momentum", "reversion", etc.
    entry_price: float
    stop_loss: float | None
    target_price: float | None
    confidence: float  # 0-100 unified scale
    holding_period_days: int
    direction: str  # "LONG" | "SHORT"
    raw_score: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class EngineAdapter(ABC):
    """Abstract base for engine adapters.

    Each adapter wraps a single engine's signal-generation logic, accepting
    pre-fetched OHLCV data sliced to ``<= screen_date`` (point-in-time) and
    returning a list of :class:`NormalizedPick` objects.
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Short identifier used in reports and logging."""
        ...

    @abstractmethod
    async def generate_picks(
        self,
        screen_date: date,
        price_data: dict[str, pd.DataFrame],
        spy_df: pd.DataFrame,
        qqq_df: pd.DataFrame,
    ) -> list[NormalizedPick]:
        """Generate picks for a single trading day.

        Args:
            screen_date: The as-of date (no data beyond this date).
            price_data: Ticker -> OHLCV DataFrame, already sliced to <= screen_date.
            spy_df: SPY OHLCV sliced to <= screen_date.
            qqq_df: QQQ OHLCV sliced to <= screen_date.

        Returns:
            List of NormalizedPick from this engine for the given date.
        """
        ...

    @abstractmethod
    def required_lookback_days(self) -> int:
        """Minimum calendar days of OHLCV history the adapter needs before
        ``screen_date`` for indicator warm-up."""
        ...
