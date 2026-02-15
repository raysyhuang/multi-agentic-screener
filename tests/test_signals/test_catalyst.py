"""Tests for catalyst signal model."""

from src.signals.catalyst import score_catalyst


def test_catalyst_fires_on_upcoming_earnings():
    features = {"close": 100.0, "atr_14": 3.0}
    fundamental_data = {
        "earnings_surprises": {
            "beat_streak": 4,
            "avg_surprise_pct": 15.0,
            "earnings_momentum": 5.0,
        },
        "insider_activity": {
            "insider_net_ratio": 0.6,
        },
    }
    sentiment = {"sentiment_score": 0.4}

    result = score_catalyst(
        "AAPL", features, fundamental_data,
        days_to_earnings=10, sentiment=sentiment,
    )

    assert result is not None
    assert result.direction == "LONG"
    assert result.catalyst_type == "earnings"
    assert result.stop_loss < result.entry_price
    assert result.target_1 > result.entry_price


def test_catalyst_skips_far_earnings():
    features = {"close": 100.0, "atr_14": 3.0}
    fundamental_data = {"earnings_surprises": {}, "insider_activity": {}}

    result = score_catalyst(
        "AAPL", features, fundamental_data,
        days_to_earnings=60,  # too far out
    )

    assert result is None


def test_catalyst_skips_no_earnings():
    features = {"close": 100.0, "atr_14": 3.0}
    fundamental_data = {"earnings_surprises": {}, "insider_activity": {}}

    result = score_catalyst(
        "AAPL", features, fundamental_data,
        days_to_earnings=None,
    )

    assert result is None


def test_catalyst_weak_fundamentals():
    features = {"close": 100.0, "atr_14": 3.0}
    fundamental_data = {
        "earnings_surprises": {
            "beat_streak": 0,
            "avg_surprise_pct": -5.0,
            "earnings_momentum": -3.0,
        },
        "insider_activity": {
            "insider_net_ratio": -0.5,
        },
    }
    sentiment = {"sentiment_score": -0.4}

    result = score_catalyst(
        "BAD", features, fundamental_data,
        days_to_earnings=10, sentiment=sentiment,
    )

    # Should be None or very low score due to weak fundamentals
    if result is not None:
        assert result.score < 50


def test_catalyst_holding_period_matches_earnings():
    features = {"close": 100.0, "atr_14": 3.0}
    fundamental_data = {
        "earnings_surprises": {"beat_streak": 3, "avg_surprise_pct": 10.0, "earnings_momentum": 2.0},
        "insider_activity": {"insider_net_ratio": 0.3},
    }

    result = score_catalyst(
        "AAPL", features, fundamental_data,
        days_to_earnings=7, sentiment={"sentiment_score": 0.2},
    )

    if result is not None:
        assert result.holding_period <= 15
        assert result.holding_period >= 7  # at least days_to_earnings
