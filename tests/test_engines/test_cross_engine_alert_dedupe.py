from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from src.main import (
    _cross_engine_alert_fingerprint_payload,
    _cross_engine_change_reasons,
    _stable_payload_hash,
)


def test_cross_engine_alert_hash_is_stable_for_order_only_changes():
    payload_a = _cross_engine_alert_fingerprint_payload(
        run_date=date(2026, 2, 21),
        regime_consensus="bull",
        engines_reporting=2,
        convergent_picks=[{"ticker": "COHR"}, {"ticker": "HAS"}],
        portfolio=[
            {"ticker": "COHR", "weight_pct": 6.36, "entry_price": 195.96, "stop_loss": 186.16, "target_price": 215.56, "holding_period_days": 14, "source": "unique"},
            {"ticker": "HAS", "weight_pct": 8.48, "entry_price": 86.89, "stop_loss": 82.55, "target_price": 95.58, "holding_period_days": 14, "source": "unique"},
        ],
        executive_summary="normal",
    )
    payload_b = _cross_engine_alert_fingerprint_payload(
        run_date=date(2026, 2, 21),
        regime_consensus="BULL",
        engines_reporting=2,
        convergent_picks=[{"ticker": "HAS"}, {"ticker": "COHR"}],
        portfolio=[
            {"ticker": "HAS", "weight_pct": 8.4801, "entry_price": 86.89001, "stop_loss": 82.55001, "target_price": 95.57999, "holding_period_days": 14, "source": "UNIQUE"},
            {"ticker": "COHR", "weight_pct": 6.3601, "entry_price": 195.96001, "stop_loss": 186.16001, "target_price": 215.56001, "holding_period_days": 14, "source": "UNIQUE"},
        ],
        executive_summary="normal",
    )

    assert _stable_payload_hash(payload_a) == _stable_payload_hash(payload_b)


def test_cross_engine_change_reasons_capture_material_updates():
    existing = SimpleNamespace(
        engines_reporting=1,
        executive_summary="No halt",
        portfolio_recommendation=[{"ticker": "HAS", "weight_pct": 21.21}],
        regime_consensus="bull",
        convergent_tickers=[{"ticker": "HAS"}],
    )
    new_synthesis = {
        "executive_summary": "No halt",
        "portfolio": [
            {"ticker": "HAS", "weight_pct": 8.48},
            {"ticker": "COHR", "weight_pct": 6.36},
        ],
        "regime_consensus": "choppy",
        "convergent_picks": [{"ticker": "HAS"}, {"ticker": "COHR"}],
    }

    reasons = _cross_engine_change_reasons(
        existing_synthesis=existing,
        new_synthesis=new_synthesis,
        new_engines_reporting=2,
    )

    assert any("engines reporting" in r for r in reasons)
    assert "portfolio tickers changed" in reasons
    assert "regime consensus changed" in reasons
    assert "convergent tickers changed" in reasons
