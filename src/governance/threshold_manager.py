"""Threshold Manager — applies meta-analyst suggestions with guardrails.

Closes the feedback loop: MetaAnalystAgent suggests adjustments,
ThresholdManager validates, versions, and optionally applies them.

Default: dry-run mode (human reviews before applying).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.base import ThresholdAdjustment
from src.config import get_settings

logger = logging.getLogger(__name__)

# Parameters the meta-analyst is allowed to adjust
ADJUSTABLE_PARAMS = {
    "vix_high_threshold",
    "vix_low_threshold",
    "breadth_bullish_threshold",
    "breadth_bearish_threshold",
    "min_price",
    "min_avg_daily_volume",
    "top_n_for_interpretation",
    "top_n_for_debate",
}

# Guardrails
MAX_CHANGE_PCT = 0.20    # Max 20% change per adjustment
MIN_SAMPLE_SIZE = 20     # Min trades to support a suggestion
SNAPSHOT_DIR = Path("data/threshold_snapshots")


@dataclass
class ThresholdSnapshot:
    """Versioned snapshot of threshold parameters."""
    timestamp: str
    run_date: str
    values: dict[str, Any]
    adjustments_applied: list[dict] = field(default_factory=list)
    adjustments_rejected: list[dict] = field(default_factory=list)
    dry_run: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AdjustmentResult:
    """Result of processing a batch of threshold adjustments."""
    applied: list[ThresholdAdjustment]
    rejected: list[tuple[ThresholdAdjustment, str]]  # (adjustment, rejection_reason)
    snapshot: ThresholdSnapshot
    dry_run: bool


def get_current_thresholds() -> dict[str, Any]:
    """Read current threshold values from settings."""
    settings = get_settings()
    return {
        "vix_high_threshold": settings.vix_high_threshold,
        "vix_low_threshold": settings.vix_low_threshold,
        "breadth_bullish_threshold": settings.breadth_bullish_threshold,
        "breadth_bearish_threshold": settings.breadth_bearish_threshold,
        "min_price": settings.min_price,
        "min_avg_daily_volume": settings.min_avg_daily_volume,
        "top_n_for_interpretation": settings.top_n_for_interpretation,
        "top_n_for_debate": settings.top_n_for_debate,
    }


def validate_adjustment(
    adjustment: ThresholdAdjustment,
    current_values: dict[str, Any],
) -> str | None:
    """Validate a single adjustment against guardrails.

    Returns None if valid, or a rejection reason string.
    """
    # Check whitelist
    if adjustment.parameter not in ADJUSTABLE_PARAMS:
        return f"Parameter '{adjustment.parameter}' is not in the adjustable whitelist"

    # Check sample size
    if adjustment.evidence_sample_size < MIN_SAMPLE_SIZE:
        return (
            f"Insufficient evidence: {adjustment.evidence_sample_size} trades "
            f"(minimum {MIN_SAMPLE_SIZE})"
        )

    # Check parameter exists
    current = current_values.get(adjustment.parameter)
    if current is None:
        return f"Parameter '{adjustment.parameter}' not found in current settings"

    # Check max change percentage
    if current != 0:
        change_pct = abs(adjustment.suggested_value - current) / abs(current)
        if change_pct > MAX_CHANGE_PCT:
            return (
                f"Change too large: {change_pct:.1%} "
                f"(max {MAX_CHANGE_PCT:.0%}). "
                f"Current={current}, suggested={adjustment.suggested_value}"
            )

    # Check suggested value is reasonable (not negative for thresholds)
    if adjustment.suggested_value < 0 and adjustment.parameter not in (
        "breadth_bearish_threshold",
    ):
        return f"Suggested value {adjustment.suggested_value} is negative"

    return None  # Valid


def process_adjustments(
    adjustments: list[ThresholdAdjustment],
    run_date: str,
    dry_run: bool = True,
) -> AdjustmentResult:
    """Process a batch of threshold adjustments from the meta-analyst.

    In dry-run mode (default): validates and logs proposals but doesn't apply.
    In live mode: applies validated adjustments to settings and saves snapshot.

    Returns AdjustmentResult with applied/rejected details.
    """
    current = get_current_thresholds()
    applied: list[ThresholdAdjustment] = []
    rejected: list[tuple[ThresholdAdjustment, str]] = []

    for adj in adjustments:
        reason = validate_adjustment(adj, current)
        if reason:
            rejected.append((adj, reason))
            logger.info(
                "Threshold adjustment REJECTED: %s (%.2f → %.2f): %s",
                adj.parameter, adj.current_value, adj.suggested_value, reason,
            )
        else:
            applied.append(adj)
            if dry_run:
                logger.info(
                    "Threshold adjustment PROPOSED (dry-run): %s (%.2f → %.2f) — %s",
                    adj.parameter, adj.current_value, adj.suggested_value, adj.reasoning,
                )
            else:
                _apply_adjustment(adj)
                logger.info(
                    "Threshold adjustment APPLIED: %s (%.2f → %.2f) — %s",
                    adj.parameter, adj.current_value, adj.suggested_value, adj.reasoning,
                )

    # Create snapshot
    snapshot = ThresholdSnapshot(
        timestamp=datetime.utcnow().isoformat(),
        run_date=run_date,
        values=get_current_thresholds(),
        adjustments_applied=[a.model_dump() for a in applied],
        adjustments_rejected=[
            {"adjustment": a.model_dump(), "reason": r} for a, r in rejected
        ],
        dry_run=dry_run,
    )

    # Save snapshot to disk
    _save_snapshot(snapshot)

    return AdjustmentResult(
        applied=applied,
        rejected=rejected,
        snapshot=snapshot,
        dry_run=dry_run,
    )


def apply_snapshot(run_date: str) -> bool:
    """Apply a previously-saved dry-run snapshot (human approval step).

    Loads the snapshot for the given run_date and applies all its proposed adjustments.
    """
    snapshot = load_snapshot(run_date)
    if snapshot is None:
        logger.error("No snapshot found for %s", run_date)
        return False

    if not snapshot.dry_run:
        logger.warning("Snapshot for %s was already applied", run_date)
        return False

    adjustments = [
        ThresholdAdjustment(**adj) for adj in snapshot.adjustments_applied
    ]

    for adj in adjustments:
        _apply_adjustment(adj)
        logger.info(
            "Applied from snapshot: %s → %.2f", adj.parameter, adj.suggested_value,
        )

    # Update snapshot to mark as applied
    snapshot.dry_run = False
    _save_snapshot(snapshot)

    return True


def _apply_adjustment(adj: ThresholdAdjustment) -> None:
    """Apply a single adjustment to the runtime settings."""
    settings = get_settings()
    if hasattr(settings, adj.parameter):
        setattr(settings, adj.parameter, adj.suggested_value)


def _save_snapshot(snapshot: ThresholdSnapshot) -> Path:
    """Save snapshot to disk for versioning and rollback."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"snapshot_{snapshot.run_date}.json"
    path.write_text(json.dumps(snapshot.to_dict(), indent=2, default=str))
    logger.info("Snapshot saved: %s", path)
    return path


def load_snapshot(run_date: str) -> ThresholdSnapshot | None:
    """Load a snapshot by run date."""
    path = SNAPSHOT_DIR / f"snapshot_{run_date}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return ThresholdSnapshot(**data)


def get_snapshot_history(limit: int = 20) -> list[dict]:
    """List recent threshold snapshots."""
    if not SNAPSHOT_DIR.is_dir():
        return []

    snapshots = []
    for path in sorted(SNAPSHOT_DIR.glob("snapshot_*.json"), reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text())
            snapshots.append({
                "run_date": data.get("run_date"),
                "timestamp": data.get("timestamp"),
                "dry_run": data.get("dry_run", True),
                "applied_count": len(data.get("adjustments_applied", [])),
                "rejected_count": len(data.get("adjustments_rejected", [])),
            })
        except Exception:
            continue
    return snapshots
