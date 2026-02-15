"""Retrain policy — determines when models should be re-evaluated.

Implements dual gates: minimum sample count AND minimum time elapsed
since last training. Generates versioned model identifiers for tracking.

Ported from KooCore-D governance/retrain_policy.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime

logger = logging.getLogger(__name__)


@dataclass
class RetrainDecision:
    """Result of a retrain check."""

    should_retrain: bool
    reason: str
    new_samples_since_last: int
    days_since_last: int
    current_version: str | None = None


def should_retrain(
    last_train_date: date | None,
    new_samples: int,
    min_samples: int = 50,
    min_days: int = 7,
) -> RetrainDecision:
    """Determine whether to retrain based on sample accumulation and time elapsed.

    Both gates must pass:
        1. At least min_samples new trades since last training
        2. At least min_days since last training date

    If last_train_date is None, always recommends retraining.
    """
    if last_train_date is None:
        return RetrainDecision(
            should_retrain=True,
            reason="No previous training — initial training needed",
            new_samples_since_last=new_samples,
            days_since_last=999,
        )

    days_elapsed = (date.today() - last_train_date).days

    if new_samples < min_samples:
        return RetrainDecision(
            should_retrain=False,
            reason=f"Insufficient samples: {new_samples} < {min_samples}",
            new_samples_since_last=new_samples,
            days_since_last=days_elapsed,
        )

    if days_elapsed < min_days:
        return RetrainDecision(
            should_retrain=False,
            reason=f"Too recent: {days_elapsed} days < {min_days} day minimum",
            new_samples_since_last=new_samples,
            days_since_last=days_elapsed,
        )

    return RetrainDecision(
        should_retrain=True,
        reason=f"Ready: {new_samples} samples, {days_elapsed} days elapsed",
        new_samples_since_last=new_samples,
        days_since_last=days_elapsed,
    )


def generate_model_version(
    regime: str,
    train_date: date,
    version_num: int = 1,
) -> str:
    """Generate a versioned model identifier.

    Format: "model_{regime}_{date}_v{version}"
    Example: "model_bull_2025-03-15_v1"
    """
    return f"model_{regime}_{train_date.isoformat()}_v{version_num}"


@dataclass
class VersionMetadata:
    """Comprehensive metadata for a model version."""

    version_id: str
    regime: str
    train_start: date
    train_end: date
    n_samples: int
    n_positive: int
    hit_rate: float
    created_at: str


def build_version_metadata(
    regime: str,
    train_start: date,
    train_end: date,
    n_samples: int,
    n_positive: int,
) -> VersionMetadata:
    """Build comprehensive version metadata for a trained model."""
    version_id = generate_model_version(regime, train_end)
    hit_rate = n_positive / n_samples if n_samples > 0 else 0.0

    return VersionMetadata(
        version_id=version_id,
        regime=regime,
        train_start=train_start,
        train_end=train_end,
        n_samples=n_samples,
        n_positive=n_positive,
        hit_rate=round(hit_rate, 4),
        created_at=datetime.utcnow().isoformat(),
    )
