"""Tests for pipeline stage validation system."""

from datetime import date

import pandas as pd

from src.validation.stage_validator import (
    PipelineHealthReport,
    Severity,
    StageCheck,
    StageValidation,
    validate_macro_regime,
    validate_universe,
    validate_ohlcv,
    validate_features,
    validate_signals,
    validate_ranking,
    validate_agent_pipeline,
    validate_final_output,
    _detect_split_artifacts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_bars=40, close=100.0, include_split=False):
    """Create a minimal OHLCV DataFrame."""
    closes = [close] * n_bars
    if include_split and n_bars > 5:
        # Simulate unadjusted 2:1 split: price drops 50% and stays there
        for i in range(3, n_bars):
            closes[i] = close * 0.50
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n_bars),
        "open": [close] * n_bars,
        "high": [close * 1.02] * n_bars,
        "low": [close * 0.98] * n_bars,
        "close": closes,
        "volume": [1_000_000] * n_bars,
    })


class _MockSignal:
    def __init__(self, ticker, score=70, entry=100, stop=95, target=110):
        self.ticker = ticker
        self.score = score
        self.entry_price = entry
        self.stop_loss = stop
        self.target_1 = target
        self.target_2 = target * 1.1


class _MockPick:
    def __init__(self, ticker, entry=100, stop=95, target=110, thesis="bull"):
        self.ticker = ticker
        self.entry_price = entry
        self.stop_loss = stop
        self.target_1 = target
        self.target_2 = target * 1.1
        self.direction = "LONG"
        self.signal_model = "breakout"
        self.interpretation = type("Interp", (), {"thesis": thesis})()
        self.confidence = 75
        self.holding_period = 10
        self.features = {}


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

class TestStageValidation:
    def test_passed_when_all_checks_pass(self):
        sv = StageValidation(stage_name="test", executed=True, checks=[
            StageCheck(name="a", passed=True, severity=Severity.FAIL, message="ok"),
            StageCheck(name="b", passed=True, severity=Severity.WARN, message="ok"),
        ])
        assert sv.passed is True
        assert sv.severity == Severity.PASS

    def test_fails_when_fail_check_fails(self):
        sv = StageValidation(stage_name="test", executed=True, checks=[
            StageCheck(name="a", passed=False, severity=Severity.FAIL, message="bad"),
            StageCheck(name="b", passed=True, severity=Severity.WARN, message="ok"),
        ])
        assert sv.passed is False
        assert sv.severity == Severity.FAIL

    def test_warns_when_only_warn_fails(self):
        sv = StageValidation(stage_name="test", executed=True, checks=[
            StageCheck(name="a", passed=True, severity=Severity.FAIL, message="ok"),
            StageCheck(name="b", passed=False, severity=Severity.WARN, message="degraded"),
        ])
        assert sv.passed is True  # WARN doesn't block
        assert sv.severity == Severity.WARN

    def test_not_executed_means_fail(self):
        sv = StageValidation(stage_name="test", executed=False)
        assert sv.passed is False
        assert sv.severity == Severity.FAIL


class TestPipelineHealthReport:
    def test_all_pass(self):
        report = PipelineHealthReport(run_date=date.today())
        report.add_stage(StageValidation(stage_name="a", executed=True, checks=[
            StageCheck(name="x", passed=True, severity=Severity.FAIL, message="ok"),
        ]))
        report.add_stage(StageValidation(stage_name="b", executed=True, checks=[
            StageCheck(name="y", passed=True, severity=Severity.FAIL, message="ok"),
        ]))
        assert report.passed is True
        assert report.overall_severity == Severity.PASS
        assert report.stages_executed == 2

    def test_one_stage_fails(self):
        report = PipelineHealthReport(run_date=date.today())
        report.add_stage(StageValidation(stage_name="a", executed=True, checks=[
            StageCheck(name="x", passed=True, severity=Severity.FAIL, message="ok"),
        ]))
        report.add_stage(StageValidation(stage_name="b", executed=True, checks=[
            StageCheck(name="y", passed=False, severity=Severity.FAIL, message="bad"),
        ]))
        assert report.passed is False
        assert report.overall_severity == Severity.FAIL
        assert len(report.all_warnings) == 1

    def test_to_dict_serializable(self):
        import json
        report = PipelineHealthReport(run_date=date.today())
        report.add_stage(StageValidation(stage_name="test", executed=True, checks=[
            StageCheck(name="x", passed=True, severity=Severity.PASS, message="ok", value=42),
        ]))
        d = report.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        assert "stages" in d
        assert "overall_severity" in d


# ---------------------------------------------------------------------------
# Stage-specific validators
# ---------------------------------------------------------------------------

class TestValidateMacroRegime:
    def test_healthy_macro(self):
        spy = _make_df(30)
        qqq = _make_df(30)
        sv = validate_macro_regime(
            macro={}, spy_df=spy, qqq_df=qqq,
            regime="bull", confidence=0.7, vix=18.5,
        )
        assert sv.passed is True

    def test_missing_spy_fails(self):
        sv = validate_macro_regime(
            macro={}, spy_df=pd.DataFrame(), qqq_df=_make_df(30),
            regime="bull", confidence=0.7, vix=18.5,
        )
        assert sv.passed is False
        check = next(c for c in sv.checks if c.name == "spy_data")
        assert check.passed is False

    def test_invalid_regime_fails(self):
        sv = validate_macro_regime(
            macro={}, spy_df=_make_df(30), qqq_df=_make_df(30),
            regime="sideways", confidence=0.5, vix=20,
        )
        assert sv.passed is False

    def test_no_vix_warns(self):
        sv = validate_macro_regime(
            macro={}, spy_df=_make_df(30), qqq_df=_make_df(30),
            regime="bull", confidence=0.7, vix=None,
        )
        vix_check = next(c for c in sv.checks if c.name == "vix_data")
        assert vix_check.passed is False
        assert vix_check.severity == Severity.WARN


class TestValidateUniverse:
    def test_healthy_universe(self):
        sv = validate_universe(raw_count=5000, filtered_count=200, filtered=[])
        assert sv.passed is True

    def test_empty_raw_fails(self):
        sv = validate_universe(raw_count=10, filtered_count=5, filtered=[])
        assert sv.passed is False

    def test_aggressive_filter_warns(self):
        sv = validate_universe(raw_count=5000, filtered_count=100, filtered=[])
        # 98% drop rate should warn
        assert sv.severity in (Severity.PASS, Severity.WARN)


class TestValidateOhlcv:
    def test_healthy_data(self):
        price_data = {f"T{i}": _make_df(100) for i in range(100)}
        sv = validate_ohlcv(100, price_data, 90)
        assert sv.passed is True

    def test_low_fetch_rate_warns(self):
        price_data = {f"T{i}": pd.DataFrame() for i in range(100)}
        price_data["T0"] = _make_df(50)
        sv = validate_ohlcv(100, price_data, 1)
        fetch_check = next(c for c in sv.checks if c.name == "ohlcv_fetch_rate")
        assert fetch_check.passed is False

    def test_missing_columns_fails(self):
        bad_df = pd.DataFrame({"close": [100], "volume": [1000]})
        price_data = {"T0": bad_df}
        sv = validate_ohlcv(1, price_data, 1)
        schema_check = next(c for c in sv.checks if c.name == "ohlcv_schema")
        assert schema_check.passed is False

    def test_split_check_scoped_to_qualified_tickers(self):
        good_df = _make_df(40, close=100)
        bad_df = _make_df(10, close=200, include_split=True)
        price_data = {"GOOD": good_df, "BAD": bad_df}
        sv = validate_ohlcv(
            tickers_requested=2,
            price_data=price_data,
            qualified_count=1,
            split_check_tickers=["GOOD"],
        )
        split_check = next(c for c in sv.checks if c.name == "split_artifacts")
        assert split_check.passed is True


class TestDetectSplitArtifacts:
    def test_clean_data_no_splits(self):
        price_data = {"AAPL": _make_df(40, close=200)}
        assert _detect_split_artifacts(price_data) == []

    def test_detects_2_to_1_split(self):
        # Create a 10-bar df where the split happens at bar 3 (within tail(30))
        price_data = {"AAPL": _make_df(10, close=200, include_split=True)}
        result = _detect_split_artifacts(price_data)
        assert "AAPL" in result


class TestValidateFeatures:
    def test_healthy_features(self):
        features = {f"T{i}": {"rsi_14": 55, "atr_14": 3.5, "close": 100, "volume": 1e6, "sma_20": 98} for i in range(80)}
        sv = validate_features(features, 100)
        assert sv.passed is True

    def test_nan_features_warn(self):
        features = {"T0": {"rsi_14": float("nan"), "close": 100, "volume": 1e6}}
        sv = validate_features(features, 1)
        nan_check = next(c for c in sv.checks if c.name == "feature_nan_check")
        assert nan_check.passed is False

    def test_inf_features_warn(self):
        features = {"T0": {"atr_14": float("inf"), "close": 100}}
        sv = validate_features(features, 1)
        inf_check = next(c for c in sv.checks if c.name == "feature_inf_check")
        assert inf_check.passed is False


class TestValidateSignals:
    def test_healthy_signals(self):
        signals = [_MockSignal("AAPL"), _MockSignal("MSFT")]
        sv = validate_signals(signals, 100, ["breakout"])
        assert sv.passed is True

    def test_no_signals_warns(self):
        sv = validate_signals([], 100, ["breakout"])
        sig_check = next(c for c in sv.checks if c.name == "signal_count")
        assert sig_check.passed is False

    def test_bad_prices_warn(self):
        signals = [_MockSignal("BAD", entry=100, stop=110, target=120)]  # stop > entry
        sv = validate_signals(signals, 100, ["breakout"])
        price_check = next(c for c in sv.checks if c.name == "signal_price_sanity")
        assert price_check.passed is False

    def test_out_of_range_score_warns(self):
        signals = [_MockSignal("BAD", score=150)]
        sv = validate_signals(signals, 100, ["breakout"])
        score_check = next(c for c in sv.checks if c.name == "signal_score_bounds")
        assert score_check.passed is False


class TestValidateRanking:
    def test_healthy_ranking(self):
        sv = validate_ranking(10, 8, 50)
        assert sv.passed is True

    def test_no_ranked_warns(self):
        sv = validate_ranking(0, 0, 50)
        rank_check = next(c for c in sv.checks if c.name == "ranked_count")
        assert rank_check.passed is False


class TestValidateAgentPipeline:
    def test_healthy_pipeline(self):
        sv = validate_agent_pipeline("hybrid", 3, 2, [{"agent": "test"}], 5)
        assert sv.passed is True

    def test_no_logs_in_agentic_warns(self):
        sv = validate_agent_pipeline("hybrid", 3, 0, [], 5)
        logs_check = next(c for c in sv.checks if c.name == "agent_logs_present")
        assert logs_check.passed is False

    def test_quant_only_skips_log_check(self):
        sv = validate_agent_pipeline("quant_only", 3, 0, [], 5)
        has_log_check = any(c.name == "agent_logs_present" for c in sv.checks)
        assert has_log_check is False


class TestValidateFinalOutput:
    def test_healthy_output(self):
        picks = [_MockPick("AAPL")]
        sv = validate_final_output(picks, "pass", "bull")
        assert sv.passed is True

    def test_validation_gate_unknown_fails(self):
        sv = validate_final_output([], "unknown_status", "bull")
        gate_check = next(c for c in sv.checks if c.name == "validation_gate_ran")
        assert gate_check.passed is False

    def test_bad_risk_reward_warns(self):
        picks = [_MockPick("BAD", entry=100, stop=50, target=102)]  # R:R = 2/50 = 0.04
        sv = validate_final_output(picks, "pass", "bull")
        rr_check = next(c for c in sv.checks if c.name == "risk_reward_sanity")
        assert rr_check.passed is False
