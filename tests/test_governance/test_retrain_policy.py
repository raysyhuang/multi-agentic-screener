"""Tests for retrain policy."""

from datetime import date, timedelta

from src.governance.retrain_policy import (
    should_retrain,
    generate_model_version,
    build_version_metadata,
)


def test_should_retrain_no_prior_training():
    result = should_retrain(last_train_date=None, new_samples=0)
    assert result.should_retrain is True
    assert "initial training" in result.reason.lower()


def test_should_retrain_insufficient_samples():
    result = should_retrain(
        last_train_date=date.today() - timedelta(days=30),
        new_samples=10,
        min_samples=50,
    )
    assert result.should_retrain is False
    assert "insufficient" in result.reason.lower()


def test_should_retrain_too_recent():
    result = should_retrain(
        last_train_date=date.today() - timedelta(days=3),
        new_samples=100,
        min_days=7,
    )
    assert result.should_retrain is False
    assert "recent" in result.reason.lower()


def test_should_retrain_ready():
    result = should_retrain(
        last_train_date=date.today() - timedelta(days=14),
        new_samples=60,
        min_samples=50,
        min_days=7,
    )
    assert result.should_retrain is True
    assert "ready" in result.reason.lower()


def test_generate_model_version():
    version = generate_model_version("bull", date(2025, 3, 15))
    assert version == "model_bull_2025-03-15_v1"

    version2 = generate_model_version("bear", date(2025, 3, 15), version_num=3)
    assert version2 == "model_bear_2025-03-15_v3"


def test_build_version_metadata():
    meta = build_version_metadata(
        regime="bull",
        train_start=date(2025, 1, 1),
        train_end=date(2025, 3, 15),
        n_samples=100,
        n_positive=65,
    )
    assert meta.regime == "bull"
    assert meta.n_samples == 100
    assert meta.hit_rate == 0.65
    assert "model_bull_2025-03-15" in meta.version_id
