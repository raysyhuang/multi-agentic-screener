from src.core.universe import normalize_ticker_for_yahoo


def test_normalize_ticker_for_yahoo_strips_dollar_and_uppercases():
    assert normalize_ticker_for_yahoo("$aapl") == "AAPL"
    assert normalize_ticker_for_yahoo(" brk.b ") == "BRK-B"
