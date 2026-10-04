"""How each `signal_source` is classified, in one place.

These sets decide whether a row competes for official slots, whether it counts
as a trade, and whether it counts as an idea. They were previously declared in
`src/main.py` and re-declared in `scripts/export_dashboard_data.py`, which is
how `pead_60d_shadow` ended up excluded from the dashboard's open-position
count while still inflating "Picks today", the funnel and the drift monitor's
`total_resolved` — three counts, three call sites, one rule remembered in two
of them. Anything that counts rows imports from here.
"""

from __future__ import annotations

# Recorded, never in the book. Their history must not feed the OFFICIAL
# cooldown — a quarantined pick must not suppress a book pick.
SHADOW_SOURCES: frozenset[str] = frozenset({
    "sniper_shadow", "mr_shadow", "pead_60d_shadow",
    "reclaim_shadow_h5", "reclaim_shadow_h10", "reclaim_shadow_h20",
})

# Streams that re-measure an entry already counted under another stream. The
# rows are real and their outcomes are real, but they are not additional ideas,
# positions, trades or capital, so they never contribute to a COUNT of any of
# those — not picks, not open positions, not the drift monitor's resolved-trade
# total, which gates whether other streams' alerts are sent at all.
PAIRED_OBSERVATION_SOURCES: frozenset[str] = frozenset({"pead_60d_shadow"})

# Capital-like PEAD paper positions: these, and only these, consume the primary
# sleeve's concurrency slots.
PEAD_POSITION_SOURCES: tuple[str, ...] = ("pead_paper", "pead_neglected")

# The one source that IS the book. Official cooldown history is defined from
# this, not from "everything that is not shadow": `pead_paper` and
# `pead_neglected` are quarantined too, and defining the complement let a
# quarantined PEAD row suppress an eligible official pick — the reverse of the
# quarantine. Legacy rows predate `signal_source` and are official, so a null
# source reads as `mas_official` at the call site.
BOOK_SOURCES: frozenset[str] = frozenset({"mas_official"})

# Sources whose open positions consume the sniper concurrency cap.
SNIPER_CAP_SOURCES: tuple[str, ...] = ("mas_official", "sniper_shadow")

# RECLAIM shadow lanes (RECLAIM_MAS_SPEC_v1.1 §8.3). Three fixed-horizon clones
# of one trigger, evaluated ONLY by src/output/reclaim_tracker.py under a
# structural stop. Deliberately NOT in PAIRED_OBSERVATION_SOURCES: that set
# means "zero stop, infinite target" in the generic tracker, which is the wrong
# exit policy here. Never in BOOK_SOURCES, PEAD_POSITION_SOURCES or
# SNIPER_CAP_SOURCES. Excluded from every count of picks, positions, trades and
# from cooldown history — the Reclaim reporter is the only consumer.
RECLAIM_SHADOW_SOURCES: frozenset[str] = frozenset({
    "reclaim_shadow_h5", "reclaim_shadow_h10", "reclaim_shadow_h20",
})

# h5/h20 are paired observations of the h10 anchor. Reporter de-duplication only.
RECLAIM_PAIRED_SOURCES: frozenset[str] = frozenset({
    "reclaim_shadow_h5", "reclaim_shadow_h20",
})

RECLAIM_EXIT_POLICY = "STRUCTURAL_STOP_FIXED_HORIZON"

# Every reclaim Outcome row carries one of these. Stats consumers across the
# repo filter `skip_reason IS NULL`, so tagging the rows keeps them out of the
# paper gate, decay check, Telegram summaries and drift monitor by construction
# rather than by a filter remembered at each call site.
RECLAIM_SKIP_REASON = "reclaim_shadow"
RECLAIM_NONTRADE_SKIP_REASON = "reclaim_nontrade"


def reclaim_source(horizon: int) -> str:
    return f"reclaim_shadow_h{horizon}"
