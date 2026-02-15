"""Governance audit trail — records per-run governance context for forensic analysis.

Every pipeline run captures its governance state: which models were active,
what regime was detected, whether decay was flagged, and any warnings.
This enables post-hoc analysis of decision quality.

Ported from KooCore-D governance/artifacts.py, adapted for multi-agentic contracts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GovernanceRecord:
    """Immutable record of governance state for a single pipeline run."""

    run_id: str
    run_date: str
    regime: str = ""
    trading_mode: str = "PAPER"  # PAPER or LIVE
    execution_mode: str = "agentic_full"  # quant_only | hybrid | agentic_full
    models_active: list[str] = field(default_factory=list)
    decay_detected: bool = False
    decay_reasons: list[str] = field(default_factory=list)
    eligibility_passed: bool = True
    governance_flags: list[str] = field(default_factory=list)
    config_hash: str = ""
    git_commit: str = ""
    universe_size: int = 0
    candidates_scored: int = 0
    picks_approved: int = 0
    pipeline_duration_s: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> GovernanceRecord:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class GovernanceContext:
    """Context manager that collects governance info during pipeline execution.

    Usage:
        with GovernanceContext(run_id="abc123", run_date="2025-03-15") as gov:
            gov.set_regime("bull")
            gov.set_trading_mode("PAPER")
            gov.add_flag("low_breadth_warning")
        record = gov.record
    """

    def __init__(self, run_id: str, run_date: str):
        self._record = GovernanceRecord(run_id=run_id, run_date=run_date)

    def __enter__(self) -> GovernanceContext:
        self._record.git_commit = _get_git_commit()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self._record.governance_flags.append(
                f"pipeline_exception: {exc_type.__name__}: {exc_val}"
            )

    @property
    def record(self) -> GovernanceRecord:
        return self._record

    def set_regime(self, regime: str) -> None:
        self._record.regime = regime

    def set_trading_mode(self, mode: str) -> None:
        self._record.trading_mode = mode

    def set_execution_mode(self, mode: str) -> None:
        self._record.execution_mode = mode

    def set_models_active(self, models: list[str]) -> None:
        self._record.models_active = models

    def set_decay(self, detected: bool, reasons: list[str] | None = None) -> None:
        self._record.decay_detected = detected
        self._record.decay_reasons = reasons or []

    def set_eligibility(self, passed: bool) -> None:
        self._record.eligibility_passed = passed

    def set_config_hash(self, config: dict[str, Any]) -> None:
        serialized = json.dumps(config, sort_keys=True, default=str)
        self._record.config_hash = hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def set_pipeline_stats(
        self,
        universe_size: int = 0,
        candidates_scored: int = 0,
        picks_approved: int = 0,
        duration_s: float = 0.0,
    ) -> None:
        self._record.universe_size = universe_size
        self._record.candidates_scored = candidates_scored
        self._record.picks_approved = picks_approved
        self._record.pipeline_duration_s = round(duration_s, 2)

    def add_flag(self, flag: str) -> None:
        self._record.governance_flags.append(flag)


def _get_git_commit() -> str:
    """Get current git commit hash for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""
