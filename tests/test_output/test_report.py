"""Tests for HTML report generation."""

from datetime import date

from src.output.report import generate_daily_report, generate_performance_report


def test_generate_daily_report_with_picks():
    html = generate_daily_report(
        run_date=date(2025, 3, 15),
        regime="bull",
        regime_details={"vix": 14.0, "spy_trend": "above_sma20"},
        picks=[
            {
                "ticker": "AAPL",
                "direction": "LONG",
                "signal_model": "breakout",
                "entry_price": 195.50,
                "stop_loss": 190.00,
                "target_1": 210.00,
                "holding_period": 10,
                "confidence": 78,
                "thesis": "Strong momentum breakout.",
                "debate_summary": "Bull case wins on volume.",
                "risk_flags": ["earnings_imminent"],
            },
        ],
        vetoed=["MSFT", "NVDA"],
        pipeline_stats={"universe_size": 500, "candidates_scored": 10},
    )

    assert "AAPL" in html
    assert "LONG" in html
    assert "$195.50" in html
    assert "breakout" in html
    assert "78" in html
    assert "bull" in html.lower()
    assert "MSFT" in html
    assert "NVDA" in html


def test_generate_daily_report_no_picks():
    html = generate_daily_report(
        run_date=date(2025, 3, 15),
        regime="bear",
        regime_details={},
        picks=[],
        vetoed=[],
        pipeline_stats={"universe_size": 500, "candidates_scored": 10},
    )

    assert "No High-Conviction Picks" in html
    assert "bear" in html.lower()


def test_generate_performance_report_with_data():
    html = generate_performance_report(
        performance_data={
            "total_signals": 25,
            "overall": {"trades": 25, "win_rate": 0.64, "avg_pnl": 1.8},
            "by_model": {
                "breakout": {"trades": 15, "win_rate": 0.67, "avg_pnl": 2.1},
                "mean_reversion": {"trades": 10, "win_rate": 0.60, "avg_pnl": 1.3},
            },
            "by_regime": {
                "bull": {"trades": 18, "win_rate": 0.72, "avg_pnl": 2.5},
                "choppy": {"trades": 7, "win_rate": 0.43, "avg_pnl": -0.5},
            },
            "by_confidence": {
                "high": {"trades": 10, "win_rate": 0.80, "avg_pnl": 3.0},
                "medium": {"trades": 15, "win_rate": 0.53, "avg_pnl": 0.8},
            },
        },
        period_days=30,
    )

    assert "Performance Report" in html
    assert "breakout" in html
    assert "64.0%" in html


def test_generate_performance_report_no_data():
    html = generate_performance_report(
        performance_data={"total_signals": 0},
        period_days=30,
    )

    assert "No closed trades" in html
