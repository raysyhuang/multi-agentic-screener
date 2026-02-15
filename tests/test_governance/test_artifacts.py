"""Tests for governance audit trail."""

from src.governance.artifacts import GovernanceRecord, GovernanceContext


def test_governance_record_defaults():
    record = GovernanceRecord(run_id="abc123", run_date="2025-03-15")
    assert record.run_id == "abc123"
    assert record.regime == ""
    assert record.trading_mode == "PAPER"
    assert record.decay_detected is False
    assert record.governance_flags == []
    assert record.eligibility_passed is True


def test_governance_record_roundtrip():
    record = GovernanceRecord(
        run_id="abc123",
        run_date="2025-03-15",
        regime="bull",
        trading_mode="PAPER",
        models_active=["breakout", "mean_reversion"],
        decay_detected=True,
        decay_reasons=["hit_rate_decay"],
        governance_flags=["low_breadth"],
    )
    d = record.to_dict()
    restored = GovernanceRecord.from_dict(d)
    assert restored.run_id == "abc123"
    assert restored.regime == "bull"
    assert restored.decay_detected is True
    assert "hit_rate_decay" in restored.decay_reasons
    assert "low_breadth" in restored.governance_flags


def test_governance_context_basic():
    ctx = GovernanceContext(run_id="test1", run_date="2025-03-15")
    ctx.__enter__()
    ctx.set_regime("bear")
    ctx.set_trading_mode("PAPER")
    ctx.set_models_active(["mean_reversion"])
    ctx.set_decay(False)
    ctx.set_pipeline_stats(universe_size=200, candidates_scored=10, picks_approved=2, duration_s=45.3)
    ctx.add_flag("test_flag")
    ctx.__exit__(None, None, None)

    record = ctx.record
    assert record.regime == "bear"
    assert record.trading_mode == "PAPER"
    assert record.models_active == ["mean_reversion"]
    assert record.universe_size == 200
    assert record.picks_approved == 2
    assert record.pipeline_duration_s == 45.3
    assert "test_flag" in record.governance_flags


def test_governance_context_captures_exception():
    ctx = GovernanceContext(run_id="test2", run_date="2025-03-15")
    ctx.__enter__()
    try:
        ctx.__exit__(ValueError, ValueError("test error"), None)
    except Exception:
        pass
    assert any("ValueError" in f for f in ctx.record.governance_flags)


def test_governance_config_hash():
    ctx = GovernanceContext(run_id="test3", run_date="2025-03-15")
    ctx.__enter__()
    ctx.set_config_hash({"min_price": 5.0, "top_n": 10})
    ctx.__exit__(None, None, None)
    assert len(ctx.record.config_hash) == 16

    # Same config → same hash
    ctx2 = GovernanceContext(run_id="test4", run_date="2025-03-15")
    ctx2.__enter__()
    ctx2.set_config_hash({"min_price": 5.0, "top_n": 10})
    ctx2.__exit__(None, None, None)
    assert ctx.record.config_hash == ctx2.record.config_hash
