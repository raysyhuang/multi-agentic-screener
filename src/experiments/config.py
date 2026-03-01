"""Shadow track configuration — loader, validator, and sync to DB."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy import select

from src.db.models import ShadowTrack
from src.db.session import get_session

logger = logging.getLogger(__name__)

TRACKS_YAML = Path(__file__).resolve().parent.parent.parent / "configs" / "tracks.yaml"

# Keys that map directly to Settings fields (flat overrides)
FLAT_OVERRIDE_KEYS = {
    "convergence_1_engine_multiplier",
    "convergence_2_engine_multiplier",
    "convergence_3_engine_multiplier",
    "convergence_4_engine_multiplier",
    "convergence_sector_multiplier",
    "credibility_lookback_days",
    "credibility_min_picks_for_weight",
    "guardian_max_drawdown_pct",
    "guardian_bear_sizing",
    "guardian_choppy_sizing",
    "guardian_max_portfolio_heat_pct",
    "guardian_halt_portfolio_heat_pct",
    "guardian_overheat_sizing_floor",
    "guardian_per_trade_risk_cap_pct",
    "guardian_streak_reduction_after",
    "guardian_halt_after_consecutive_losses",
    "guardian_max_sector_concentration",
    "min_confidence",
    "low_overlap_max_positions",
    "low_overlap_max_total_weight_pct",
    "regime_gate_bear_penalty_multiplier",
}

# Special-cased nested keys
NESTED_OVERRIDE_KEYS = {"regime_multipliers"}


@dataclass
class TrackConfig:
    """Parsed track configuration from YAML or DB."""

    name: str
    description: str
    overrides: dict
    generation: int = 1
    parent_track: str | None = None


def load_tracks_from_yaml(path: Path | None = None) -> list[TrackConfig]:
    """Load track definitions from configs/tracks.yaml."""
    yaml_path = path or TRACKS_YAML
    if not yaml_path.exists():
        logger.warning("tracks.yaml not found at %s", yaml_path)
        return []

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not data or "tracks" not in data:
        return []

    tracks: list[TrackConfig] = []
    for entry in data["tracks"]:
        name = entry.get("name")
        if not name:
            logger.warning("Track entry missing 'name', skipping: %s", entry)
            continue

        overrides = entry.get("overrides", {})
        errors = validate_overrides(overrides)
        if errors:
            logger.warning("Track '%s' has invalid overrides: %s", name, errors)
            continue

        tracks.append(TrackConfig(
            name=name,
            description=entry.get("description", ""),
            overrides=overrides,
            generation=entry.get("generation", 1),
            parent_track=entry.get("parent_track"),
        ))

    logger.info("Loaded %d track configs from %s", len(tracks), yaml_path)
    return tracks


def validate_overrides(overrides: dict) -> list[str]:
    """Validate that override keys are recognized. Returns list of error strings."""
    errors: list[str] = []
    for key in overrides:
        if key not in FLAT_OVERRIDE_KEYS and key not in NESTED_OVERRIDE_KEYS:
            errors.append(f"Unknown override key: {key}")
    return errors


async def sync_tracks_to_db(tracks: list[TrackConfig] | None = None) -> list[ShadowTrack]:
    """Sync YAML track configs to DB. Creates new tracks, updates existing.

    Returns list of all active ShadowTrack rows.
    """
    if tracks is None:
        tracks = load_tracks_from_yaml()

    async with get_session() as session:
        # Fetch existing tracks
        result = await session.execute(select(ShadowTrack))
        existing = {t.name: t for t in result.scalars().all()}

        for tc in tracks:
            if tc.name in existing:
                row = existing[tc.name]
                # Update config if changed
                if row.config != tc.overrides:
                    row.config = tc.overrides
                    row.description = tc.description
                    logger.info("Updated track config: %s", tc.name)
            else:
                session.add(ShadowTrack(
                    name=tc.name,
                    generation=tc.generation,
                    parent_track=tc.parent_track,
                    status="active",
                    config=tc.overrides,
                    description=tc.description,
                ))
                logger.info("Created new track: %s", tc.name)

        # Re-fetch all active tracks
        result = await session.execute(
            select(ShadowTrack).where(ShadowTrack.status == "active")
        )
        active = result.scalars().all()

    return active


async def get_active_tracks() -> list[ShadowTrack]:
    """Fetch all active shadow tracks from DB."""
    async with get_session() as session:
        result = await session.execute(
            select(ShadowTrack).where(ShadowTrack.status == "active")
        )
        return result.scalars().all()
