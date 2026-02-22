"""Tests for fundamental feature engineering."""

from src.features.fundamental import (
    score_earnings_surprise,
    score_insider_activity,
    score_analyst_estimates,
    score_financial_ratios,
    days_to_next_earnings,
)


def test_earnings_surprise_beat_streak():
    earnings = [
        {"actualEarningResult": 1.5, "estimatedEarning": 1.2},
        {"actualEarningResult": 1.3, "estimatedEarning": 1.1},
        {"actualEarningResult": 1.0, "estimatedEarning": 0.9},
    ]
    result = score_earnings_surprise(earnings)
    assert result["beat_streak"] == 3
    assert result["last_surprise_pct"] > 0
    assert result["avg_surprise_pct"] > 0


def test_earnings_surprise_miss():
    earnings = [
        {"actualEarningResult": 0.8, "estimatedEarning": 1.0},
    ]
    result = score_earnings_surprise(earnings)
    assert result["beat_streak"] == 0
    assert result["last_surprise_pct"] < 0


def test_earnings_surprise_empty():
    result = score_earnings_surprise([])
    assert result["beat_streak"] == 0
    assert result["last_surprise_pct"] is None


def test_insider_activity_buying():
    txns = [
        {"transactionType": "P-Purchase", "securitiesTransacted": 1000, "price": 50, "transactionDate": "2026-02-01"},
        {"transactionType": "P-Purchase", "securitiesTransacted": 500, "price": 48, "transactionDate": "2026-01-15"},
    ]
    result = score_insider_activity(txns, lookback_days=60)
    assert result["insider_buy_count"] == 2
    assert result["insider_net_ratio"] > 0
    assert result["insider_buy_value"] > 0


def test_insider_activity_empty():
    result = score_insider_activity([])
    assert result["insider_buy_count"] == 0
    assert result["insider_net_ratio"] == 0.0


def test_days_to_next_earnings():
    from datetime import date, timedelta
    future = date.today() + timedelta(days=10)
    calendar = [{"symbol": "AAPL", "date": str(future)}]
    result = days_to_next_earnings(calendar, "AAPL")
    assert result == 10


def test_days_to_next_earnings_not_found():
    result = days_to_next_earnings([], "AAPL")
    assert result is None


def test_analyst_estimates_scores_revision_trend():
    estimates = [
        {"estimatedEpsAvg": 2.1, "estimatedRevenueAvg": 120_000_000},
        {"estimatedEpsAvg": 1.8, "estimatedRevenueAvg": 100_000_000},
    ]
    result = score_analyst_estimates(estimates)
    assert result["eps_estimate_next"] == 2.1
    assert result["eps_revision_pct"] > 0
    assert result["revenue_revision_pct"] > 0


def test_financial_ratios_flags_healthy_value_profile():
    ratios = {"priceEarningsRatio": 14.0, "priceToBookRatio": 1.8, "debtEquityRatio": 0.45}
    result = score_financial_ratios(ratios)
    assert result["mean_reversion_ok"] is True
    assert result["value_score"] >= 60


def test_financial_ratios_rejects_levered_expensive_profile():
    ratios = {"priceEarningsRatio": 48.0, "priceToBookRatio": 7.0, "debtEquityRatio": 3.2}
    result = score_financial_ratios(ratios)
    assert result["mean_reversion_ok"] is False
