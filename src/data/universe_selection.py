"""Universe ticker selection helpers.

Keeps OHLCV fetch size bounded while reducing large-cap concentration bias.
"""

from __future__ import annotations

# Default tier thresholds (min dollar-volume) → fraction of max_tickers.
DEFAULT_TIER_ALLOCATIONS: dict[float, float] = {
    100e6: 0.60,   # $100M+ daily dollar volume
    50e6:  0.25,   # $50-100M
    10e6:  0.15,   # $10-50M
}


def _dollar_volume(entry: dict) -> float:
    price = float(entry.get("price") or entry.get("lastPrice") or 0.0)
    volume = float(entry.get("volume") or 0.0)
    return price * volume


def _score(entry: dict) -> tuple[float, float]:
    """Score an entry by market cap then dollar volume."""
    market_cap = float(entry.get("marketCap") or 0.0)
    return market_cap, _dollar_volume(entry)


def _round_robin_fill(candidates: list[dict], slots: int) -> list[dict]:
    """Pick *slots* entries from *candidates* using first-letter round-robin."""
    if slots <= 0 or not candidates:
        return []

    buckets: dict[str, list[dict]] = {}
    for e in candidates:
        sym = e.get("symbol", "")
        letter = sym[0].upper() if sym else "#"
        buckets.setdefault(letter, []).append(e)

    selected: list[dict] = []
    letters = sorted(buckets.keys())
    idx = 0
    while letters and len(selected) < slots:
        letter = letters[idx % len(letters)]
        bucket = buckets[letter]
        if bucket:
            selected.append(bucket.pop(0))
        if not bucket:
            letters.remove(letter)
            idx -= 1
        idx += 1

    return selected


def select_ohlcv_tickers(
    filtered_universe: list[dict],
    max_tickers: int,
    core_ratio: float = 0.5,
    tier_allocations: dict[float, float] | None = None,
) -> list[str]:
    """Select tickers for OHLCV fetch with dollar-volume stratification.

    Strategy:
    1. Classify candidates into dollar-volume tiers.
    2. Allocate slots per tier (default 60/25/15 for $100M+ / $50-100M / $10-50M).
    3. Overflow: if a tier has fewer candidates than its allocation, surplus
       slots flow to the adjacent tier above first, then below.
    4. Within each tier, rank by (market_cap, dollar_volume) descending and
       apply round-robin first-letter diversification.

    Parameters
    ----------
    filtered_universe : list[dict]
        Screened universe entries (must contain ``symbol``, ``price``,
        ``volume``, and ``marketCap`` keys).
    max_tickers : int
        Maximum number of tickers to return.
    core_ratio : float
        Legacy parameter kept for backward compatibility; ignored when
        *tier_allocations* is provided.
    tier_allocations : dict[float, float] | None
        Mapping of ``{min_dollar_volume: fraction}``.  Tiers are evaluated
        from highest threshold to lowest.  Fractions should sum to ~1.0.
    """
    if max_tickers <= 0 or not filtered_universe:
        return []

    # Fast path: universe fits entirely
    ranked = sorted(filtered_universe, key=_score, reverse=True)
    if len(ranked) <= max_tickers:
        return [e.get("symbol", "") for e in ranked if e.get("symbol")]

    tiers = tier_allocations if tier_allocations is not None else DEFAULT_TIER_ALLOCATIONS

    # Sort thresholds descending so highest tier is processed first.
    sorted_thresholds = sorted(tiers.keys(), reverse=True)

    # ── Classify candidates into tiers ──
    tier_candidates: dict[float, list[dict]] = {t: [] for t in sorted_thresholds}
    for entry in ranked:
        dv = _dollar_volume(entry)
        placed = False
        for threshold in sorted_thresholds:
            if dv >= threshold:
                tier_candidates[threshold].append(entry)
                placed = True
                break
        # Entries below the lowest tier threshold are discarded.
        if not placed:
            pass

    # ── Compute slot allocations ──
    raw_slots = {t: int(max_tickers * tiers[t]) for t in sorted_thresholds}

    # Ensure rounding doesn't lose slots — give remainder to highest tier.
    total_raw = sum(raw_slots.values())
    if total_raw < max_tickers:
        raw_slots[sorted_thresholds[0]] += max_tickers - total_raw

    # ── Overflow redistribution ──
    # Pass 1: cap each tier to its candidate count, accumulate surplus.
    final_slots: dict[float, int] = {}
    surplus = 0
    for t in sorted_thresholds:
        available = len(tier_candidates[t])
        alloc = raw_slots[t] + surplus
        if available >= alloc:
            final_slots[t] = alloc
            surplus = 0
        else:
            final_slots[t] = available
            surplus = alloc - available

    # Pass 2 (bottom-up): distribute any remaining surplus upward.
    if surplus > 0:
        for t in reversed(sorted_thresholds):
            available = len(tier_candidates[t])
            room = available - final_slots[t]
            if room > 0:
                give = min(room, surplus)
                final_slots[t] += give
                surplus -= give
            if surplus <= 0:
                break

    # ── Fill each tier with round-robin diversification ──
    all_selected: list[dict] = []
    for t in sorted_thresholds:
        candidates = tier_candidates[t]
        slots = final_slots[t]
        if slots <= 0:
            continue
        filled = _round_robin_fill(candidates, slots)
        all_selected.extend(filled)

    # ── Stable de-duplication ──
    selected_symbols: list[str] = []
    seen: set[str] = set()
    for e in all_selected:
        sym = e.get("symbol", "")
        if sym and sym not in seen:
            selected_symbols.append(sym)
            seen.add(sym)
        if len(selected_symbols) >= max_tickers:
            break
    return selected_symbols
