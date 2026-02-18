from src.data.universe_selection import select_ohlcv_tickers


def _entry(symbol: str, mcap: float, price: float = 100.0, vol: float = 1_000_000) -> dict:
    return {
        "symbol": symbol,
        "marketCap": mcap,
        "price": price,
        "volume": vol,
    }


def test_select_ohlcv_tickers_respects_cap_and_uniqueness():
    universe = [_entry(f"A{i:03d}", 1_000_000 - i) for i in range(300)]
    selected = select_ohlcv_tickers(universe, max_tickers=120)

    assert len(selected) == 120
    assert len(set(selected)) == 120


def test_select_ohlcv_tickers_adds_breadth_beyond_top_cap_slice():
    # Strong large-cap concentration in A-prefix names; diversified tail in B..Z.
    top_heavy = [_entry(f"A{i:03d}", 10_000_000 - i) for i in range(250)]
    broad_tail = [_entry(f"{ch}{i:02d}", 1_000 - i) for ch in "BCDEFGHIJKLMNOPQRSTUVWXYZ" for i in range(8)]
    universe = top_heavy + broad_tail

    selected = select_ohlcv_tickers(universe, max_tickers=200)
    first_letters = {s[0] for s in selected}

    # Should not collapse to only A-prefix picks.
    assert len(first_letters) > 5


def test_tier_stratification_populates_all_buckets():
    """All three dollar-volume tiers should be represented when candidates exist."""
    # Tier 1: $100M+ daily dollar volume (price * volume >= 100M)
    tier1 = [_entry(f"H{i:03d}", mcap=50e9, price=200.0, vol=1_000_000) for i in range(100)]
    # Tier 2: $50-100M (price * volume in [50M, 100M))
    tier2 = [_entry(f"M{i:03d}", mcap=5e9, price=70.0, vol=1_000_000) for i in range(80)]
    # Tier 3: $10-50M (price * volume in [10M, 50M))
    tier3 = [_entry(f"S{i:03d}", mcap=1e9, price=25.0, vol=1_000_000) for i in range(60)]

    universe = tier1 + tier2 + tier3
    selected = select_ohlcv_tickers(universe, max_tickers=100)

    selected_set = set(selected)
    count_t1 = sum(1 for s in selected_set if s.startswith("H"))
    count_t2 = sum(1 for s in selected_set if s.startswith("M"))
    count_t3 = sum(1 for s in selected_set if s.startswith("S"))

    assert len(selected) == 100
    assert len(set(selected)) == 100
    # Each tier should have meaningful representation.
    assert count_t1 >= 50, f"Tier $100M+ got {count_t1}, expected >=50"
    assert count_t2 >= 20, f"Tier $50-100M got {count_t2}, expected >=20"
    assert count_t3 >= 10, f"Tier $10-50M got {count_t3}, expected >=10"


def test_tier_overflow_when_tier_underpopulated():
    """Surplus slots flow to other tiers when one tier lacks candidates."""
    # Only 5 candidates in tier 1 ($100M+), plenty in tiers 2 & 3
    tier1 = [_entry(f"H{i:03d}", mcap=50e9, price=200.0, vol=1_000_000) for i in range(5)]
    tier2 = [_entry(f"M{i:03d}", mcap=5e9, price=70.0, vol=1_000_000) for i in range(80)]
    tier3 = [_entry(f"S{i:03d}", mcap=1e9, price=25.0, vol=1_000_000) for i in range(60)]

    universe = tier1 + tier2 + tier3
    selected = select_ohlcv_tickers(universe, max_tickers=100)

    selected_set = set(selected)
    count_t1 = sum(1 for s in selected_set if s.startswith("H"))
    count_t2 = sum(1 for s in selected_set if s.startswith("M"))
    count_t3 = sum(1 for s in selected_set if s.startswith("S"))

    assert len(selected) == 100
    # All 5 tier-1 candidates should be picked
    assert count_t1 == 5
    # Overflow should fill into tier 2 and/or tier 3
    assert count_t2 + count_t3 == 95

