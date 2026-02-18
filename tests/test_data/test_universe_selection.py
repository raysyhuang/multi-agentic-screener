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

