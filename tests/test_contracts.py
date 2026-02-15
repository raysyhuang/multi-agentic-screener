"""Tests for data contracts — stage envelope and typed payloads."""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from src.contracts import (
    StageEnvelope,
    StageName,
    StageStatus,
    StageError,
    DataIngestPayload,
    TickerSnapshot,
    FeaturePayload,
    TickerFeatures,
    SignalPrefilterPayload,
    CandidateScores,
    RegimePayload,
    RegimeInfo,
    AgentReviewPayload,
    TickerReview,
    ValidationPayload,
    LeakageChecks,
    FragilityMetrics,
    FinalOutputPayload,
    FinalPick,
)


# ── StageEnvelope ──


def test_envelope_defaults():
    env = StageEnvelope(
        stage=StageName.DATA_INGEST,
        payload={"test": True},
    )
    assert len(env.run_id) == 12
    assert env.status == StageStatus.SUCCESS
    assert env.errors == []
    assert isinstance(env.created_at, datetime)


def test_envelope_errors_imply_failed():
    env = StageEnvelope(
        stage=StageName.FEATURE,
        payload=None,
        errors=[StageError(code="E001", message="Something broke")],
    )
    assert env.status == StageStatus.FAILED


def test_envelope_explicit_run_id():
    env = StageEnvelope(
        run_id="abc123",
        stage=StageName.REGIME,
        payload={},
    )
    assert env.run_id == "abc123"


# ── DataIngestPayload ──


def test_data_ingest_payload():
    payload = DataIngestPayload(
        asof_date=date(2025, 3, 15),
        universe=[
            TickerSnapshot(ticker="AAPL", last_price=195.0, volume=50_000_000),
            TickerSnapshot(ticker="MSFT", last_price=400.0, volume=30_000_000, market_cap=3e12),
        ],
    )
    assert len(payload.universe) == 2
    assert payload.universe[1].market_cap == 3e12
    assert payload.universe[0].source_provenance == "polygon"


# ── FeaturePayload ──


def test_feature_payload():
    payload = FeaturePayload(
        asof_date=date(2025, 3, 15),
        ticker_features=[
            TickerFeatures(ticker="AAPL", rsi_14=65.0, rvol_20d=2.1),
        ],
    )
    assert payload.ticker_features[0].rsi_14 == 65.0
    assert payload.ticker_features[0].returns_5d is None  # optional


# ── SignalPrefilterPayload ──


def test_signal_prefilter_payload():
    payload = SignalPrefilterPayload(
        asof_date=date(2025, 3, 15),
        candidates=[
            CandidateScores(
                ticker="AAPL",
                model_scores={"breakout": 72.0, "catalyst": 55.0},
                aggregate_score=68.0,
            ),
        ],
    )
    assert payload.candidates[0].model_scores["breakout"] == 72.0


# ── RegimePayload ──


def test_regime_payload():
    payload = RegimePayload(
        asof_date=date(2025, 3, 15),
        regime=RegimeInfo(
            label="bull",
            confidence=0.82,
            signals_allowed=["breakout", "catalyst"],
        ),
        gated_candidates=["AAPL", "MSFT"],
    )
    assert payload.regime.label == "bull"
    assert len(payload.gated_candidates) == 2


# ── AgentReviewPayload ──


def test_agent_review_payload():
    payload = AgentReviewPayload(
        ticker_reviews=[
            TickerReview(
                ticker="AAPL",
                signal_thesis="Strong momentum breakout",
                signal_confidence=78.0,
                counter_thesis="Resistance overhead",
                risk_decision="approve",
                risk_notes="Manageable risk",
            ),
            TickerReview(
                ticker="MSFT",
                signal_thesis="",
                signal_confidence=0,
                risk_decision="veto",
                risk_notes="Debate rejected",
            ),
        ],
    )
    assert len(payload.ticker_reviews) == 2
    assert payload.ticker_reviews[0].risk_decision == "approve"
    assert payload.ticker_reviews[1].risk_decision == "veto"


# ── ValidationPayload ──


def test_validation_payload_pass():
    payload = ValidationPayload(
        leakage_checks=LeakageChecks(),
        fragility_metrics=FragilityMetrics(slippage_sensitivity=0.1),
        validation_status="pass",
        checks={"timestamp_integrity_check": "pass", "next_bar_execution_check": "pass"},
        fragility_score=0.15,
    )
    assert payload.validation_status == "pass"
    assert payload.fragility_score == 0.15


def test_validation_payload_fail():
    payload = ValidationPayload(
        leakage_checks=LeakageChecks(
            asof_timestamp_present=False,
            future_data_columns_found=["forward_return"],
        ),
        fragility_metrics=FragilityMetrics(),
        validation_status="fail",
        checks={"timestamp_integrity_check": "fail"},
        key_risks=["Look-ahead bias detected"],
    )
    assert payload.validation_status == "fail"
    assert len(payload.key_risks) == 1


# ── FinalOutputPayload ──


def test_final_output_with_picks():
    payload = FinalOutputPayload(
        decision="Top1To2",
        picks=[
            FinalPick(
                ticker="AAPL",
                entry_zone=195.50,
                stop_loss=190.00,
                targets=[210.00, 215.00],
                confidence=78.0,
                regime_context="bull",
            ),
        ],
    )
    assert payload.decision == "Top1To2"
    assert len(payload.picks) == 1
    assert payload.picks[0].targets == [210.00, 215.00]


def test_final_output_no_trade():
    payload = FinalOutputPayload(
        decision="NoTrade",
        no_trade_reason="Validation gate failed",
    )
    assert payload.decision == "NoTrade"
    assert payload.no_trade_reason == "Validation gate failed"


def test_final_output_no_trade_requires_reason():
    with pytest.raises(ValidationError, match="no_trade_reason"):
        FinalOutputPayload(decision="NoTrade")


# ── Strict contracts — unknown fields rejected ──


def test_strict_model_rejects_unknown_fields():
    """Contract models must reject unknown fields (docs/data_contracts.md)."""
    with pytest.raises(ValidationError, match="extra"):
        TickerSnapshot(ticker="AAPL", last_price=195.0, volume=50_000_000, unknown_field="bad")

    with pytest.raises(ValidationError, match="extra"):
        RegimeInfo(label="bull", confidence=0.8, signals_allowed=[], bogus=True)

    with pytest.raises(ValidationError, match="extra"):
        LeakageChecks(asof_timestamp_present=True, extra_check=False)


def test_stage_envelope_rejects_extra_fields():
    """StageEnvelope itself should reject unknown fields."""
    with pytest.raises(ValidationError, match="extra"):
        StageEnvelope(
            stage=StageName.DATA_INGEST,
            payload={"test": True},
            mystery_field="should fail",
        )


# ── Full pipeline envelope chain ──


def test_envelope_chain_has_consistent_run_id():
    run_id = "test_run_123"
    envelopes = [
        StageEnvelope(run_id=run_id, stage=StageName.DATA_INGEST, payload={}),
        StageEnvelope(run_id=run_id, stage=StageName.FEATURE, payload={}),
        StageEnvelope(run_id=run_id, stage=StageName.SIGNAL_PREFILTER, payload={}),
        StageEnvelope(run_id=run_id, stage=StageName.REGIME, payload={}),
        StageEnvelope(run_id=run_id, stage=StageName.AGENT_REVIEW, payload={}),
        StageEnvelope(run_id=run_id, stage=StageName.VALIDATION, payload={}),
        StageEnvelope(run_id=run_id, stage=StageName.FINAL_OUTPUT, payload={}),
    ]
    assert all(e.run_id == run_id for e in envelopes)
    assert len(set(e.stage for e in envelopes)) == 7
