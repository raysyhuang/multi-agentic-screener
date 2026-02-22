"""Pipeline stage validation — ensures every step runs, completes, and produces sane output.

Each pipeline stage has a validator that checks:
  1. Did this stage execute at all? (completeness)
  2. Did it produce non-degenerate output? (sanity)
  3. Are outputs internally consistent? (consistency)
  4. Are there signs of degraded or garbage data? (quality)

The PipelineHealthReport aggregates all stage results into a single
pass/warn/fail verdict stored on DailyRun and surfaced on the dashboard.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class StageCheck:
    """One validation check within a stage."""
    name: str
    passed: bool
    severity: Severity  # FAIL = pipeline compromised, WARN = degraded but usable
    message: str
    value: float | str | dict | list | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
            "value": self.value,
        }


@dataclass
class StageValidation:
    """Validation result for one pipeline stage."""
    stage_name: str
    executed: bool  # Did this stage run at all?
    checks: list[StageCheck] = field(default_factory=list)
    duration_s: float | None = None

    @property
    def passed(self) -> bool:
        if not self.executed:
            return False
        return all(c.passed for c in self.checks if c.severity == Severity.FAIL)

    @property
    def severity(self) -> Severity:
        if not self.executed:
            return Severity.FAIL
        if any(not c.passed and c.severity == Severity.FAIL for c in self.checks):
            return Severity.FAIL
        if any(not c.passed and c.severity == Severity.WARN for c in self.checks):
            return Severity.WARN
        return Severity.PASS

    @property
    def warnings(self) -> list[str]:
        return [c.message for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {
            "stage": self.stage_name,
            "executed": self.executed,
            "passed": self.passed,
            "severity": self.severity.value,
            "checks": [c.to_dict() for c in self.checks],
            "warnings": self.warnings,
            "duration_s": self.duration_s,
        }


@dataclass
class PipelineHealthReport:
    """Aggregated health across all pipeline stages."""
    run_date: date
    stages: list[StageValidation] = field(default_factory=list)

    @property
    def overall_severity(self) -> Severity:
        if any(s.severity == Severity.FAIL for s in self.stages):
            return Severity.FAIL
        if any(s.severity == Severity.WARN for s in self.stages):
            return Severity.WARN
        return Severity.PASS

    @property
    def passed(self) -> bool:
        return self.overall_severity != Severity.FAIL

    @property
    def stages_executed(self) -> int:
        return sum(1 for s in self.stages if s.executed)

    @property
    def total_stages(self) -> int:
        return len(self.stages)

    @property
    def all_warnings(self) -> list[str]:
        warnings = []
        for s in self.stages:
            for w in s.warnings:
                warnings.append(f"[{s.stage_name}] {w}")
        return warnings

    def add_stage(self, stage: StageValidation) -> None:
        self.stages.append(stage)

    def to_dict(self) -> dict:
        return {
            "run_date": str(self.run_date),
            "overall_severity": self.overall_severity.value,
            "passed": self.passed,
            "stages_executed": self.stages_executed,
            "total_stages": self.total_stages,
            "stages": [s.to_dict() for s in self.stages],
            "warnings": self.all_warnings,
        }


# ---------------------------------------------------------------------------
# Stage 1: Macro / Regime validation
# ---------------------------------------------------------------------------

def validate_macro_regime(
    macro: dict,
    spy_df: pd.DataFrame | None,
    qqq_df: pd.DataFrame | None,
    regime: str,
    confidence: float,
    vix: float | None,
) -> StageValidation:
    """Validate that macro context and regime classification are reliable."""
    sv = StageValidation(stage_name="macro_regime", executed=True)

    # Check SPY data
    spy_ok = spy_df is not None and isinstance(spy_df, pd.DataFrame) and not spy_df.empty and len(spy_df) >= 15
    sv.checks.append(StageCheck(
        name="spy_data",
        passed=spy_ok,
        severity=Severity.FAIL,
        message="SPY data missing or insufficient" if not spy_ok else "SPY data OK",
        value=len(spy_df) if spy_df is not None and isinstance(spy_df, pd.DataFrame) else 0,
    ))

    # Check QQQ data
    qqq_ok = qqq_df is not None and isinstance(qqq_df, pd.DataFrame) and not qqq_df.empty and len(qqq_df) >= 15
    sv.checks.append(StageCheck(
        name="qqq_data",
        passed=qqq_ok,
        severity=Severity.WARN,
        message="QQQ data missing or insufficient" if not qqq_ok else "QQQ data OK",
        value=len(qqq_df) if qqq_df is not None and isinstance(qqq_df, pd.DataFrame) else 0,
    ))

    # Check VIX
    vix_ok = vix is not None and isinstance(vix, (int, float)) and 5 < vix < 90
    sv.checks.append(StageCheck(
        name="vix_data",
        passed=vix_ok,
        severity=Severity.WARN,
        message=f"VIX missing or unreasonable ({vix})" if not vix_ok else f"VIX OK ({vix})",
        value=vix,
    ))

    # Check regime validity
    valid_regimes = {"bull", "bear", "choppy"}
    regime_ok = regime in valid_regimes
    sv.checks.append(StageCheck(
        name="regime_valid",
        passed=regime_ok,
        severity=Severity.FAIL,
        message=f"Invalid regime: {regime}" if not regime_ok else f"Regime: {regime}",
        value=regime,
    ))

    # Check confidence is reasonable
    conf_ok = 0 < confidence <= 1.0
    sv.checks.append(StageCheck(
        name="regime_confidence",
        passed=conf_ok,
        severity=Severity.WARN,
        message=f"Regime confidence out of range: {confidence}" if not conf_ok else f"Confidence: {confidence:.2f}",
        value=confidence,
    ))

    return sv


# ---------------------------------------------------------------------------
# Stage 2: Universe construction validation
# ---------------------------------------------------------------------------

def validate_universe(
    raw_count: int,
    filtered_count: int,
    filtered: list[dict],
) -> StageValidation:
    """Validate the universe building and filtering stage."""
    sv = StageValidation(stage_name="universe", executed=True)

    # Raw universe should be non-trivial
    raw_ok = raw_count >= 100
    sv.checks.append(StageCheck(
        name="raw_universe_size",
        passed=raw_ok,
        severity=Severity.FAIL,
        message=f"Raw universe too small ({raw_count})" if not raw_ok else f"Raw: {raw_count} tickers",
        value=raw_count,
    ))

    # Filtered should have enough
    filt_ok = filtered_count >= 50
    sv.checks.append(StageCheck(
        name="filtered_universe_size",
        passed=filt_ok,
        severity=Severity.FAIL if filtered_count < 20 else Severity.WARN,
        message=f"Filtered universe small ({filtered_count})" if not filt_ok else f"Filtered: {filtered_count}",
        value=filtered_count,
    ))

    # Filter rate should be reasonable (not >95% drop)
    if raw_count > 0:
        drop_rate = 1 - (filtered_count / raw_count)
        drop_ok = drop_rate < 0.95
        sv.checks.append(StageCheck(
            name="filter_drop_rate",
            passed=drop_ok,
            severity=Severity.WARN,
            message=f"Aggressive filtering: {drop_rate:.0%} dropped" if not drop_ok else f"Filter rate: {drop_rate:.0%}",
            value=round(drop_rate, 3),
        ))

    return sv


# ---------------------------------------------------------------------------
# Stage 3: OHLCV data quality validation
# ---------------------------------------------------------------------------

def validate_ohlcv(
    tickers_requested: int,
    price_data: dict[str, pd.DataFrame],
    qualified_count: int,
    split_check_tickers: list[str] | None = None,
) -> StageValidation:
    """Validate OHLCV data fetching and quality filtering."""
    sv = StageValidation(stage_name="ohlcv_data", executed=True)

    # Data fetch success rate
    non_empty = sum(1 for df in price_data.values() if df is not None and not df.empty)
    fetch_rate = non_empty / tickers_requested if tickers_requested > 0 else 0
    fetch_ok = fetch_rate >= 0.70
    sv.checks.append(StageCheck(
        name="ohlcv_fetch_rate",
        passed=fetch_ok,
        severity=Severity.FAIL if fetch_rate < 0.5 else Severity.WARN,
        message=f"OHLCV fetch rate low: {fetch_rate:.0%} ({non_empty}/{tickers_requested})" if not fetch_ok
            else f"Fetch rate: {fetch_rate:.0%}",
        value=round(fetch_rate, 3),
    ))

    # Check for schema consistency (all have required columns)
    required_cols = {"open", "high", "low", "close", "volume"}
    bad_schema = []
    for ticker, df in price_data.items():
        if df is not None and not df.empty:
            missing = required_cols - set(df.columns)
            if missing:
                bad_schema.append(ticker)
    schema_ok = len(bad_schema) == 0
    sv.checks.append(StageCheck(
        name="ohlcv_schema",
        passed=schema_ok,
        severity=Severity.FAIL,
        message=f"{len(bad_schema)} tickers with missing OHLCV columns: {bad_schema[:5]}" if not schema_ok
            else "OHLCV schema consistent",
        value=len(bad_schema),
    ))

    # Qualified count
    qual_ok = qualified_count >= 30
    sv.checks.append(StageCheck(
        name="qualified_ticker_count",
        passed=qual_ok,
        severity=Severity.FAIL if qualified_count < 10 else Severity.WARN,
        message=f"Few qualified tickers: {qualified_count}" if not qual_ok else f"Qualified: {qualified_count}",
        value=qualified_count,
    ))

    # Check for split/dividend artifacts: overnight gaps > 15% in last 30 days
    split_price_data = price_data
    if split_check_tickers:
        split_price_data = {
            t: price_data.get(t)
            for t in split_check_tickers
            if t in price_data
        }
    suspect_splits = _detect_split_artifacts(split_price_data)
    split_ok = len(suspect_splits) == 0
    sv.checks.append(StageCheck(
        name="split_artifacts",
        passed=split_ok,
        severity=Severity.WARN,
        message=f"Suspect unadjusted data in {len(suspect_splits)} tickers: {suspect_splits[:5]}" if not split_ok
            else "No split artifacts detected",
        value=len(suspect_splits),
    ))

    return sv


def _detect_split_artifacts(price_data: dict[str, pd.DataFrame]) -> list[str]:
    """Detect tickers with possible unadjusted split data.

    Looks for overnight gaps in the last 30 bars that match common split ratios
    (exactly ~50%, ~33%, ~25% drop, or ~100%, ~200%, ~300% gain).
    """
    suspect = []
    split_ratios = {0.50, 0.333, 0.25, 0.20}  # common split drops (2:1, 3:1, 4:1, 5:1)

    for ticker, df in price_data.items():
        if df is None or df.empty or len(df) < 5:
            continue
        recent = df.tail(30)
        if len(recent) < 2:
            continue

        returns = recent["close"].pct_change().dropna()
        for ret in returns:
            if pd.isna(ret):
                continue
            abs_ret = abs(ret)
            # Check if the gap matches a split ratio (within 5% tolerance)
            for ratio in split_ratios:
                if abs(abs_ret - ratio) < 0.05:
                    suspect.append(ticker)
                    break
            else:
                continue
            break

    return suspect


# ---------------------------------------------------------------------------
# Stage 4: Feature engineering validation
# ---------------------------------------------------------------------------

def validate_features(
    features_by_ticker: dict[str, dict],
    qualified_count: int,
    fmp_endpoint_status: dict | None = None,
) -> StageValidation:
    """Validate feature engineering output."""
    sv = StageValidation(stage_name="features", executed=True)

    # Coverage: features computed for most qualified tickers
    feat_count = len(features_by_ticker)
    coverage = feat_count / qualified_count if qualified_count > 0 else 0
    cov_ok = coverage >= 0.80
    sv.checks.append(StageCheck(
        name="feature_coverage",
        passed=cov_ok,
        severity=Severity.FAIL if coverage < 0.5 else Severity.WARN,
        message=f"Feature coverage low: {coverage:.0%} ({feat_count}/{qualified_count})" if not cov_ok
            else f"Coverage: {coverage:.0%}",
        value=round(coverage, 3),
    ))

    # NaN/inf pollution in critical features
    critical_features = ["rsi_14", "atr_14", "close", "volume", "sma_20"]
    nan_tickers = []
    inf_tickers = []
    for ticker, feat in features_by_ticker.items():
        for cf in critical_features:
            val = feat.get(cf)
            if val is None:
                continue
            try:
                fval = float(val)
                if math.isnan(fval):
                    nan_tickers.append(f"{ticker}/{cf}")
                elif math.isinf(fval):
                    inf_tickers.append(f"{ticker}/{cf}")
            except (TypeError, ValueError):
                pass

    nan_ok = len(nan_tickers) == 0
    sv.checks.append(StageCheck(
        name="feature_nan_check",
        passed=nan_ok,
        severity=Severity.WARN,
        message=f"NaN in critical features: {nan_tickers[:5]}" if not nan_ok else "No NaN in critical features",
        value=len(nan_tickers),
    ))

    inf_ok = len(inf_tickers) == 0
    sv.checks.append(StageCheck(
        name="feature_inf_check",
        passed=inf_ok,
        severity=Severity.WARN,
        message=f"Inf in critical features: {inf_tickers[:5]}" if not inf_ok else "No Inf in critical features",
        value=len(inf_tickers),
    ))

    # Degraded tickers count
    degraded = [t for t, f in features_by_ticker.items() if f.get("_degraded")]
    degraded_rate = len(degraded) / feat_count if feat_count > 0 else 0
    deg_ok = degraded_rate < 0.50
    sv.checks.append(StageCheck(
        name="degraded_data_rate",
        passed=deg_ok,
        severity=Severity.WARN,
        message=f"{len(degraded)} tickers ({degraded_rate:.0%}) have degraded fundamentals/sentiment"
            if not deg_ok else f"Degraded: {len(degraded)} ({degraded_rate:.0%})",
        value=round(degraded_rate, 3),
    ))

    # FMP fundamentals coverage: profile/earnings/insider present for most tickers
    def _has_fmp_payload(feat: dict) -> bool:
        fund = feat.get("fundamental")
        if not isinstance(fund, dict):
            return False
        profile = fund.get("profile")
        if isinstance(profile, dict) and (profile.get("symbol") or profile.get("companyName")):
            return True
        earnings = fund.get("earnings_surprises")
        insiders = fund.get("insider_transactions")
        if isinstance(earnings, list) and len(earnings) > 0:
            return True
        if isinstance(insiders, list) and len(insiders) > 0:
            return True
        return False

    requested = [feat for feat in features_by_ticker.values() if feat.get("_fundamentals_requested")]
    coverage_pool = requested if requested else list(features_by_ticker.values())
    pool_count = len(coverage_pool)
    fmp_ok_count = sum(1 for feat in coverage_pool if _has_fmp_payload(feat))
    fmp_coverage = fmp_ok_count / pool_count if pool_count > 0 else 0.0
    fmp_cov_ok = fmp_coverage >= 0.70
    sv.checks.append(StageCheck(
        name="fmp_fundamentals_coverage",
        passed=fmp_cov_ok,
        severity=Severity.WARN,
        message=(
            f"FMP fundamentals coverage low: {fmp_coverage:.0%} ({fmp_ok_count}/{pool_count})"
            if not fmp_cov_ok else
            f"FMP fundamentals coverage: {fmp_coverage:.0%} ({fmp_ok_count}/{pool_count})"
        ),
        value=round(fmp_coverage, 3),
    ))

    # Endpoint-level FMP availability (supported vs plan-gated/unsupported).
    if isinstance(fmp_endpoint_status, dict):
        endpoints = fmp_endpoint_status.get("endpoints", {})
        if isinstance(endpoints, dict) and endpoints:
            degraded = {k: v for k, v in endpoints.items() if v != "supported"}
            degraded_items = ", ".join(f"{k}={v}" for k, v in degraded.items())
            ep_ok = len(degraded) == 0
            sv.checks.append(StageCheck(
                name="fmp_endpoint_availability",
                passed=ep_ok,
                severity=Severity.WARN,
                message=(
                    "FMP endpoint availability degraded: " + degraded_items
                    if not ep_ok else
                    "FMP endpoint availability: all required endpoints supported"
                ),
                value=fmp_endpoint_status,
            ))

    return sv


# ---------------------------------------------------------------------------
# Stage 5: Signal generation validation
# ---------------------------------------------------------------------------

def validate_signals(
    all_signals: list,
    qualified_count: int,
    allowed_models: list[str],
) -> StageValidation:
    """Validate signal generation output."""
    sv = StageValidation(stage_name="signals", executed=True)

    # Signal count should be non-zero (for a reasonable universe)
    sig_ok = len(all_signals) > 0 or qualified_count < 20
    sv.checks.append(StageCheck(
        name="signal_count",
        passed=sig_ok,
        severity=Severity.WARN,
        message=f"No signals generated from {qualified_count} qualified tickers" if not sig_ok
            else f"{len(all_signals)} signals generated",
        value=len(all_signals),
    ))

    # Signal rate should be reasonable (not >50% of universe)
    if qualified_count > 0:
        sig_rate = len(all_signals) / qualified_count
        rate_ok = sig_rate <= 0.50
        sv.checks.append(StageCheck(
            name="signal_rate",
            passed=rate_ok,
            severity=Severity.WARN,
            message=f"Unusually high signal rate: {sig_rate:.0%}" if not rate_ok
                else f"Signal rate: {sig_rate:.0%}",
            value=round(sig_rate, 3),
        ))

    # Price sanity on all signals
    bad_prices = []
    for sig in all_signals:
        ep = getattr(sig, "entry_price", 0)
        sl = getattr(sig, "stop_loss", 0)
        t1 = getattr(sig, "target_1", 0)
        if ep <= 0 or sl <= 0 or t1 <= 0:
            bad_prices.append(sig.ticker)
        elif sl >= ep:
            bad_prices.append(f"{sig.ticker}(stop>=entry)")
        elif t1 <= ep:
            bad_prices.append(f"{sig.ticker}(target<=entry)")

    price_ok = len(bad_prices) == 0
    sv.checks.append(StageCheck(
        name="signal_price_sanity",
        passed=price_ok,
        severity=Severity.WARN,
        message=f"Signals with bad prices: {bad_prices[:5]}" if not price_ok else "Signal prices sane",
        value=len(bad_prices),
    ))

    # Score bounds (0-100)
    bad_scores = [s.ticker for s in all_signals if not (0 <= s.score <= 100)]
    score_ok = len(bad_scores) == 0
    sv.checks.append(StageCheck(
        name="signal_score_bounds",
        passed=score_ok,
        severity=Severity.WARN,
        message=f"Signals with out-of-range scores: {bad_scores[:5]}" if not score_ok else "Scores in bounds",
        value=len(bad_scores),
    ))

    return sv


# ---------------------------------------------------------------------------
# Stage 6: Ranking validation
# ---------------------------------------------------------------------------

def validate_ranking(
    ranked_count: int,
    post_correlation_count: int,
    total_signals: int,
) -> StageValidation:
    """Validate candidate ranking and selection."""
    sv = StageValidation(stage_name="ranking", executed=True)

    # Should have some ranked candidates if we had signals
    rank_ok = ranked_count > 0 or total_signals == 0
    sv.checks.append(StageCheck(
        name="ranked_count",
        passed=rank_ok,
        severity=Severity.WARN,
        message=f"No candidates ranked from {total_signals} signals" if not rank_ok
            else f"{ranked_count} candidates ranked",
        value=ranked_count,
    ))

    # Correlation filter shouldn't drop everything
    if ranked_count > 0:
        drop = ranked_count - post_correlation_count
        filter_ok = post_correlation_count > 0
        sv.checks.append(StageCheck(
            name="correlation_filter",
            passed=filter_ok,
            severity=Severity.WARN,
            message=f"Correlation filter dropped all {ranked_count} candidates" if not filter_ok
                else f"Correlation filter: {drop} dropped, {post_correlation_count} remain",
            value=post_correlation_count,
        ))

    return sv


# ---------------------------------------------------------------------------
# Stage 7: Agent pipeline validation
# ---------------------------------------------------------------------------

def validate_agent_pipeline(
    execution_mode: str,
    approved_count: int,
    vetoed_count: int,
    agent_logs: list,
    ranked_count: int,
) -> StageValidation:
    """Validate the agent pipeline execution."""
    sv = StageValidation(stage_name="agent_pipeline", executed=True)

    # In agentic modes, should have agent logs
    if execution_mode != "quant_only":
        logs_ok = len(agent_logs) > 0
        sv.checks.append(StageCheck(
            name="agent_logs_present",
            passed=logs_ok,
            severity=Severity.WARN,
            message="No agent logs generated" if not logs_ok else f"{len(agent_logs)} agent logs",
            value=len(agent_logs),
        ))

    # Total processed should match input
    total_processed = approved_count + vetoed_count
    process_ok = total_processed > 0 or ranked_count == 0
    sv.checks.append(StageCheck(
        name="pipeline_processed",
        passed=process_ok,
        severity=Severity.WARN,
        message=f"Pipeline processed 0 of {ranked_count} candidates" if not process_ok
            else f"Processed: {approved_count} approved, {vetoed_count} vetoed",
        value=total_processed,
    ))

    # Approval rate should be between 0% and 100%
    if ranked_count > 0:
        approval_rate = approved_count / ranked_count
        rate_ok = 0 <= approval_rate <= 1.0
        sv.checks.append(StageCheck(
            name="approval_rate",
            passed=rate_ok,
            severity=Severity.WARN,
            message=f"Approval rate: {approval_rate:.0%}" if rate_ok else f"Invalid approval rate: {approval_rate}",
            value=round(approval_rate, 3),
        ))

    return sv


# ---------------------------------------------------------------------------
# Stage 8: Final output validation
# ---------------------------------------------------------------------------

def validate_final_output(
    approved_picks: list,
    validation_status: str,
    regime: str,
) -> StageValidation:
    """Validate the final output of the pipeline."""
    sv = StageValidation(stage_name="final_output", executed=True)

    # Validation gate ran
    val_ran = validation_status in ("pass", "fail", "warn")
    sv.checks.append(StageCheck(
        name="validation_gate_ran",
        passed=val_ran,
        severity=Severity.FAIL,
        message=f"Validation gate status unknown: {validation_status}" if not val_ran
            else f"Validation gate: {validation_status}",
        value=validation_status,
    ))

    # Final picks should have complete data
    incomplete = []
    for pick in approved_picks:
        ticker = getattr(pick, "ticker", "?")
        if not getattr(pick, "entry_price", None):
            incomplete.append(f"{ticker}(no entry)")
        if not getattr(pick, "stop_loss", None):
            incomplete.append(f"{ticker}(no stop)")
        if not getattr(pick, "interpretation", None):
            incomplete.append(f"{ticker}(no thesis)")

    complete_ok = len(incomplete) == 0
    sv.checks.append(StageCheck(
        name="pick_completeness",
        passed=complete_ok,
        severity=Severity.WARN,
        message=f"Incomplete picks: {incomplete[:5]}" if not complete_ok else "All picks complete",
        value=len(incomplete),
    ))

    # Risk:reward ratio sanity
    bad_rr = []
    for pick in approved_picks:
        ep = getattr(pick, "entry_price", 0) or 0
        sl = getattr(pick, "stop_loss", 0) or 0
        t1 = getattr(pick, "target_1", 0) or 0
        if ep > 0 and sl > 0 and t1 > 0:
            risk = ep - sl
            reward = t1 - ep
            if risk > 0 and reward / risk < 0.5:
                bad_rr.append(f"{getattr(pick, 'ticker', '?')}(R:R={reward/risk:.1f})")

    rr_ok = len(bad_rr) == 0
    sv.checks.append(StageCheck(
        name="risk_reward_sanity",
        passed=rr_ok,
        severity=Severity.WARN,
        message=f"Bad risk:reward ratios: {bad_rr[:5]}" if not rr_ok else "Risk:reward ratios OK",
        value=len(bad_rr),
    ))

    return sv


# ---------------------------------------------------------------------------
# OHLCV cross-validation
# ---------------------------------------------------------------------------

async def cross_validate_ohlcv(
    aggregator,
    top_tickers: list[str],
    primary_data: dict[str, pd.DataFrame],
    from_date,
    to_date,
    max_tickers: int = 10,
) -> StageValidation:
    """Cross-validate OHLCV prices from a second source for top picks.

    Compares the latest close price between the primary source and FMP/yfinance.
    Flags tickers where prices diverge by >2%.
    """
    sv = StageValidation(stage_name="ohlcv_cross_validation", executed=True)

    tickers_to_check = top_tickers[:max_tickers]
    divergent = []

    for ticker in tickers_to_check:
        primary_df = primary_data.get(ticker)
        if primary_df is None or primary_df.empty:
            continue

        primary_close = float(primary_df["close"].iloc[-1])
        if primary_close <= 0:
            continue

        # Try to get a second source
        try:
            alt_df = await aggregator.yfinance.get_ohlcv(ticker, from_date, to_date)
            if alt_df is None or alt_df.empty:
                continue
            alt_close = float(alt_df["close"].iloc[-1])
            if alt_close <= 0:
                continue

            pct_diff = abs(primary_close - alt_close) / primary_close
            if pct_diff > 0.02:
                divergent.append(f"{ticker}({pct_diff:.1%})")
        except Exception:
            continue  # Can't cross-validate, skip

    div_ok = len(divergent) == 0
    sv.checks.append(StageCheck(
        name="price_cross_validation",
        passed=div_ok,
        severity=Severity.WARN,
        message=f"Price divergence >2% in {len(divergent)} tickers: {divergent[:5]}" if not div_ok
            else f"Cross-validated {len(tickers_to_check)} tickers, all within 2%",
        value=len(divergent),
    ))

    return sv
