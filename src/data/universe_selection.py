"""Universe ticker selection helpers.

Keeps OHLCV fetch size bounded while reducing large-cap concentration bias.
"""

from __future__ import annotations


def _score(entry: dict) -> tuple[float, float]:
    """Score an entry by market cap then dollar volume."""
    market_cap = float(entry.get("marketCap") or 0.0)
    price = float(entry.get("price") or entry.get("lastPrice") or 0.0)
    volume = float(entry.get("volume") or 0.0)
    dollar_volume = price * volume
    return market_cap, dollar_volume


def select_ohlcv_tickers(
    filtered_universe: list[dict],
    max_tickers: int,
    core_ratio: float = 0.5,
) -> list[str]:
    """Select tickers for OHLCV fetch with deterministic diversification.

    Strategy:
    1. Rank by (market cap, dollar volume) descending.
    2. Take a core block from the top of the ranking.
    3. Fill remaining slots round-robin across ticker first-letter buckets.
    """
    if max_tickers <= 0 or not filtered_universe:
        return []

    ranked = sorted(filtered_universe, key=_score, reverse=True)
    if len(ranked) <= max_tickers:
        return [e.get("symbol", "") for e in ranked if e.get("symbol")]

    core_count = max(1, min(max_tickers, int(max_tickers * core_ratio)))
    core = [e for e in ranked[:core_count] if e.get("symbol")]
    remainder = [e for e in ranked[core_count:] if e.get("symbol")]

    buckets: dict[str, list[dict]] = {}
    for e in remainder:
        sym = e["symbol"]
        letter = sym[0].upper() if sym else "#"
        buckets.setdefault(letter, []).append(e)

    # Round-robin across first-letter buckets for breadth.
    diversified: list[dict] = []
    letters = sorted(buckets.keys())
    idx = 0
    target = max_tickers - len(core)
    while letters and len(diversified) < target:
        letter = letters[idx % len(letters)]
        bucket = buckets[letter]
        if bucket:
            diversified.append(bucket.pop(0))
        if not bucket:
            letters.remove(letter)
            idx -= 1
        idx += 1

    # Stable de-duplication while preserving order.
    selected_symbols: list[str] = []
    seen: set[str] = set()
    for e in core + diversified:
        sym = e.get("symbol", "")
        if sym and sym not in seen:
            selected_symbols.append(sym)
            seen.add(sym)
        if len(selected_symbols) >= max_tickers:
            break
    return selected_symbols

