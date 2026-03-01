"""Evolutionary Track Optimizer — selection, mutation, crossover.

Cycle: compute leaderboard → select top N → eliminate bottom → generate
offspring from winners via mutation operators → register new generation.

Mutation operators:
  - Nudge (60%): perturb each numeric param ±10-20%
  - Crossover (20%): merge overrides from two parents
  - Explore (10%): try a random untested parameter
  - Invert (10%): reverse a parent's change direction
"""

from __future__ import annotations

import copy
import logging
import random
from datetime import datetime

import yaml
from sqlalchemy import select

from src.db.models import ShadowTrack
from src.db.session import get_session
from src.experiments.config import FLAT_OVERRIDE_KEYS, TRACKS_YAML, validate_overrides
from src.experiments.leaderboard import compute_leaderboard

logger = logging.getLogger(__name__)

# Default parameter ranges for exploration and clamping
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "convergence_1_engine_multiplier": (0.5, 1.5),
    "convergence_2_engine_multiplier": (0.8, 2.5),
    "convergence_3_engine_multiplier": (0.8, 2.0),
    "convergence_4_engine_multiplier": (0.8, 2.5),
    "convergence_sector_multiplier": (1.0, 1.5),
    "credibility_lookback_days": (14, 90),
    "credibility_min_picks_for_weight": (5, 30),
    "guardian_max_drawdown_pct": (10.0, 30.0),
    "guardian_bear_sizing": (0.3, 0.9),
    "guardian_choppy_sizing": (0.5, 1.0),
    "guardian_max_portfolio_heat_pct": (5.0, 20.0),
    "guardian_per_trade_risk_cap_pct": (1.0, 4.0),
    "guardian_streak_reduction_after": (2, 6),
    "guardian_halt_after_consecutive_losses": (4, 10),
    "min_confidence": (30.0, 60.0),
    "low_overlap_max_positions": (2, 5),
    "regime_gate_bear_penalty_multiplier": (0.3, 0.9),
}


async def evolve_tracks(
    top_n: int = 5,
    offspring_per: int = 4,
    min_resolved_picks: int = 10,
) -> list[dict]:
    """Run one evolutionary cycle.

    1. Compute leaderboard
    2. Select top N by composite score (require >= min_resolved_picks)
    3. Mark bottom tracks as 'eliminated'
    4. Generate offspring via mutation
    5. Create new shadow_tracks rows (generation + 1)
    6. Update configs/tracks.yaml

    Returns list of new offspring config dicts.
    """
    # Compute leaderboard with 30-day lookback for evolution decisions
    scorecards = await compute_leaderboard(lookback_days=30)
    if not scorecards:
        logger.info("Evolution: no tracks to evolve")
        return []

    # Filter to eligible tracks (active + sufficient data)
    eligible = [
        s for s in scorecards
        if s.status == "active" and s.resolved_picks >= min_resolved_picks
    ]

    if len(eligible) < 2:
        logger.info(
            "Evolution: need at least 2 eligible tracks (have %d), skipping",
            len(eligible),
        )
        return []

    # Select winners and losers
    eligible.sort(key=lambda s: s.composite_score, reverse=True)
    winners = eligible[:top_n]
    losers = eligible[top_n:]

    winner_names = {w.name for w in winners}
    loser_names = {l.name for l in losers}

    logger.info(
        "Evolution: %d winners (%s), %d eliminated (%s)",
        len(winners), ", ".join(winner_names),
        len(losers), ", ".join(loser_names),
    )

    # Mark losers as eliminated in DB
    async with get_session() as session:
        result = await session.execute(select(ShadowTrack))
        all_tracks = {t.name: t for t in result.scalars().all()}

        for name in loser_names:
            if name in all_tracks:
                all_tracks[name].status = "eliminated"

        # Determine next generation number
        max_gen = max((t.generation for t in all_tracks.values()), default=1)
        next_gen = max_gen + 1

        # Generate offspring
        offspring_configs: list[dict] = []
        for winner in winners:
            parent_config = winner.config or {}
            for i in range(offspring_per):
                child_config, method = _mutate(
                    parent_config,
                    all_parent_configs=[w.config for w in winners],
                )
                child_name = f"{winner.name}_g{next_gen}_{method[:3]}_{i}"

                # Validate
                errors = validate_overrides(child_config)
                if errors:
                    logger.warning("Skipping invalid offspring %s: %s", child_name, errors)
                    continue

                session.add(ShadowTrack(
                    name=child_name,
                    generation=next_gen,
                    parent_track=winner.name,
                    status="active",
                    config=child_config,
                    description=f"Gen {next_gen} offspring of {winner.name} via {method}",
                ))

                offspring_configs.append({
                    "name": child_name,
                    "parent": winner.name,
                    "method": method,
                    "config": child_config,
                })

    # Update YAML
    _update_tracks_yaml(offspring_configs)

    logger.info(
        "Evolution complete: gen %d, %d new tracks created",
        next_gen, len(offspring_configs),
    )
    return offspring_configs


def _mutate(
    parent_config: dict,
    all_parent_configs: list[dict],
) -> tuple[dict, str]:
    """Apply a random mutation operator to a parent config.

    Returns (child_config, method_name).
    """
    roll = random.random()

    if roll < 0.60:
        return _nudge(parent_config), "nudge"
    elif roll < 0.80:
        other = random.choice(all_parent_configs)
        return _crossover(parent_config, other), "crossover"
    elif roll < 0.90:
        return _explore(parent_config), "explore"
    else:
        return _invert(parent_config), "invert"


def _nudge(config: dict) -> dict:
    """Perturb each numeric parameter by ±10-20%."""
    child = copy.deepcopy(config)

    for key, value in child.items():
        if key == "regime_multipliers":
            # Nudge nested regime multipliers
            for regime_key, model_mults in value.items():
                if isinstance(model_mults, dict):
                    for model, mult in model_mults.items():
                        if isinstance(mult, (int, float)):
                            factor = 1.0 + random.uniform(-0.20, 0.20)
                            model_mults[model] = round(mult * factor, 3)
        elif isinstance(value, (int, float)):
            factor = 1.0 + random.uniform(-0.20, 0.20)
            new_val = value * factor
            new_val = _clamp(key, new_val)
            child[key] = round(new_val, 4) if isinstance(value, float) else int(round(new_val))

    return child


def _crossover(config_a: dict, config_b: dict) -> dict:
    """Merge overrides from two parents — pick each key randomly from either."""
    all_keys = set(config_a.keys()) | set(config_b.keys())
    child: dict = {}

    for key in all_keys:
        if random.random() < 0.5 and key in config_a:
            child[key] = copy.deepcopy(config_a[key])
        elif key in config_b:
            child[key] = copy.deepcopy(config_b[key])
        elif key in config_a:
            child[key] = copy.deepcopy(config_a[key])

    return child


def _explore(config: dict) -> dict:
    """Add a random untested parameter to the config."""
    child = copy.deepcopy(config)

    # Find parameters not already in this config
    untested = [k for k in FLAT_OVERRIDE_KEYS if k not in child]
    if not untested:
        return _nudge(child)  # all params already set, fall back to nudge

    new_key = random.choice(untested)
    lo, hi = PARAM_RANGES.get(new_key, (0.5, 2.0))

    if isinstance(lo, int) and isinstance(hi, int):
        child[new_key] = random.randint(int(lo), int(hi))
    else:
        child[new_key] = round(random.uniform(lo, hi), 4)

    return child


def _invert(config: dict) -> dict:
    """Reverse a parent's change direction — if parent increased a param, decrease it."""
    child = copy.deepcopy(config)

    # Pick a random numeric key to invert
    numeric_keys = [
        k for k, v in child.items()
        if isinstance(v, (int, float)) and k != "regime_multipliers"
    ]
    if not numeric_keys:
        return _nudge(child)

    key = random.choice(numeric_keys)
    lo, hi = PARAM_RANGES.get(key, (0.0, 100.0))
    midpoint = (lo + hi) / 2

    # Reflect value around the midpoint of its range
    current = child[key]
    inverted = midpoint + (midpoint - current)
    inverted = _clamp(key, inverted)

    child[key] = round(inverted, 4) if isinstance(current, float) else int(round(inverted))
    return child


def _clamp(key: str, value: float) -> float:
    """Clamp a value to its defined parameter range."""
    lo, hi = PARAM_RANGES.get(key, (0.0, float("inf")))
    return max(lo, min(hi, value))


def _update_tracks_yaml(offspring: list[dict]) -> None:
    """Append new offspring tracks to configs/tracks.yaml."""
    if not offspring:
        return

    try:
        if TRACKS_YAML.exists():
            with open(TRACKS_YAML) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        if "tracks" not in data:
            data["tracks"] = []

        existing_names = {t["name"] for t in data["tracks"]}
        for child in offspring:
            if child["name"] not in existing_names:
                data["tracks"].append({
                    "name": child["name"],
                    "description": f"Gen offspring of {child['parent']} via {child['method']}",
                    "overrides": child["config"],
                })

        with open(TRACKS_YAML, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Updated %s with %d new tracks", TRACKS_YAML, len(offspring))
    except Exception as e:
        logger.error("Failed to update tracks.yaml: %s", e)
