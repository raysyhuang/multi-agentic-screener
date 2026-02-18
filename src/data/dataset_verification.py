"""Dataset health verification — checks universe quality and cross-engine consistency."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# Reference mega-caps that should appear in any healthy US universe
MEGA_CAP_REFERENCE = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "UNH",
}

MIN_ALPHABET_DIVERSITY = 10  # Distinct first letters in top 200
MIN_MEGA_CAP_COVERAGE = 5   # At least 5 of 10 reference tickers present
MIN_UNIVERSE_SIZE = 50       # Minimum qualified tickers
MIN_OHLCV_BARS = 100         # Minimum bars per ticker for completeness
MAX_STALE_TRADING_DAYS = 3   # OHLCV freshness threshold


def _canonical_regime(regime: str | None) -> str:
    """Normalize regime labels across engines."""
    if not regime:
        return "unknown"
    r = regime.strip().lower()
    if any(k in r for k in ("unknown", "indeterminate", "ambiguous", "uncertain")):
        return "unknown"
    if "bull" in r:
        return "bull"
    if "bear" in r:
        return "bear"
    if "choppy" in r or "sideways" in r or "range" in r:
        return "choppy"
    return r


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    value: float | int | str | None = None


@dataclass
class DatasetHealthReport:
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "passed_count": self.passed_count,
            "total_checks": self.total_checks,
            "checks": [asdict(c) for c in self.checks],
            "warnings": self.warnings,
        }


@dataclass
class CrossEngineHealthReport:
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "passed_count": self.passed_count,
            "total_checks": self.total_checks,
            "checks": [asdict(c) for c in self.checks],
            "warnings": self.warnings,
        }


def verify_dataset(
    filtered: list[dict],
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    qualified_tickers: list[str],
    today: date,
) -> DatasetHealthReport:
    """Run health checks on the filtered universe and OHLCV data.

    Args:
        filtered: Raw filtered universe entries (dicts with symbol, price, volume, etc.)
        tickers: Top-N ticker symbols selected for OHLCV fetch
        price_data: Mapping of ticker → OHLCV DataFrame
        qualified_tickers: Tickers that passed OHLCV quality filter
        today: Current pipeline date
    """
    checks: list[CheckResult] = []
    warnings: list[str] = []

    # 1. Alphabet diversity
    first_letters = set()
    for t in tickers[:200]:
        if t:
            first_letters.add(t[0].upper())
    diversity = len(first_letters)
    alpha_ok = diversity >= MIN_ALPHABET_DIVERSITY
    checks.append(CheckResult(
        name="alphabet_diversity",
        passed=alpha_ok,
        detail=f"{diversity} distinct first letters in top {min(200, len(tickers))} tickers: {sorted(first_letters)}",
        value=diversity,
    ))
    if not alpha_ok:
        warnings.append(
            f"Alphabet bias: only {diversity} distinct first letters "
            f"({sorted(first_letters)}). Expected >= {MIN_ALPHABET_DIVERSITY}."
        )

    # 2. Dollar volume distribution
    dollar_volumes = []
    for entry in filtered[:200]:
        price = entry.get("price") or entry.get("lastPrice") or 0
        volume = entry.get("volume") or 0
        dv = price * volume
        dollar_volumes.append(dv)

    tier_100m = sum(1 for dv in dollar_volumes if dv >= 100_000_000)
    tier_50m = sum(1 for dv in dollar_volumes if 50_000_000 <= dv < 100_000_000)
    tier_10m = sum(1 for dv in dollar_volumes if 10_000_000 <= dv < 50_000_000)
    # Pass either a balanced spread OR an ultra-liquid universe.
    dv_ok = (tier_100m >= 5 and (tier_50m + tier_10m) >= 10) or (tier_100m >= 20)
    checks.append(CheckResult(
        name="dollar_volume_distribution",
        passed=dv_ok,
        detail=f"$100M+: {tier_100m}, $50-100M: {tier_50m}, $10-50M: {tier_10m}",
        value=f"{tier_100m}/{tier_50m}/{tier_10m}",
    ))
    if not dv_ok:
        warnings.append(
            f"Thin dollar volume spread: $100M+={tier_100m}, "
            f"$50-100M={tier_50m}, $10-50M={tier_10m}."
        )

    # 3. Known large-cap coverage
    ticker_set = set(tickers)
    present = MEGA_CAP_REFERENCE & ticker_set
    missing = MEGA_CAP_REFERENCE - ticker_set
    coverage = len(present)
    mega_ok = coverage >= MIN_MEGA_CAP_COVERAGE
    checks.append(CheckResult(
        name="mega_cap_coverage",
        passed=mega_ok,
        detail=f"{coverage}/10 reference mega-caps present. Missing: {sorted(missing) if missing else 'none'}",
        value=coverage,
    ))
    if not mega_ok:
        warnings.append(
            f"Mega-cap gap: only {coverage}/10 reference tickers present. "
            f"Missing: {sorted(missing)}."
        )

    # 4. Data freshness
    most_recent_date: date | None = None
    for ticker, df in price_data.items():
        if df is not None and not df.empty and "date" in df.columns:
            last = df["date"].max()
            if isinstance(last, pd.Timestamp):
                last = last.date()
            if most_recent_date is None or last > most_recent_date:
                most_recent_date = last

    if most_recent_date:
        # Count business days between most_recent_date and today
        stale_days = _business_days_between(most_recent_date, today)
        fresh_ok = stale_days <= MAX_STALE_TRADING_DAYS
        checks.append(CheckResult(
            name="data_freshness",
            passed=fresh_ok,
            detail=f"Most recent OHLCV: {most_recent_date} ({stale_days} trading days from {today})",
            value=stale_days,
        ))
        if not fresh_ok:
            warnings.append(
                f"Stale data: most recent OHLCV is {most_recent_date} "
                f"({stale_days} trading days old)."
            )
    else:
        checks.append(CheckResult(
            name="data_freshness",
            passed=False,
            detail="No OHLCV data found",
            value=None,
        ))
        warnings.append("No OHLCV data found at all.")

    # 5. OHLCV completeness
    total_with_data = 0
    sufficient_bars = 0
    for ticker in qualified_tickers:
        df = price_data.get(ticker)
        if df is not None and not df.empty:
            total_with_data += 1
            if len(df) >= MIN_OHLCV_BARS:
                sufficient_bars += 1

    if total_with_data > 0:
        completeness_pct = sufficient_bars / total_with_data * 100
    else:
        completeness_pct = 0.0
    complete_ok = completeness_pct >= 80.0
    checks.append(CheckResult(
        name="ohlcv_completeness",
        passed=complete_ok,
        detail=f"{sufficient_bars}/{total_with_data} tickers have >= {MIN_OHLCV_BARS} bars ({completeness_pct:.0f}%)",
        value=round(completeness_pct, 1),
    ))
    if not complete_ok:
        warnings.append(
            f"Low OHLCV completeness: {completeness_pct:.0f}% of tickers "
            f"have >= {MIN_OHLCV_BARS} bars."
        )

    # 6. Universe size
    n_qualified = len(qualified_tickers)
    size_ok = n_qualified >= MIN_UNIVERSE_SIZE
    checks.append(CheckResult(
        name="universe_size",
        passed=size_ok,
        detail=f"{n_qualified} qualified tickers (minimum: {MIN_UNIVERSE_SIZE})",
        value=n_qualified,
    ))
    if not size_ok:
        warnings.append(
            f"Small universe: only {n_qualified} qualified tickers "
            f"(expected >= {MIN_UNIVERSE_SIZE})."
        )

    all_passed = all(c.passed for c in checks)
    return DatasetHealthReport(passed=all_passed, checks=checks, warnings=warnings)


def verify_cross_engine(
    engine_results: list[dict],
    regime_context: dict,
) -> CrossEngineHealthReport:
    """Run cross-engine consistency checks after collecting external engine results.

    Args:
        engine_results: List of engine result dicts with keys:
            engine_name, regime, picks (list of dicts with ticker, entry_price),
            candidates_screened
        regime_context: Current regime context dict with "regime" key
    """
    checks: list[CheckResult] = []
    warnings: list[str] = []

    if not engine_results:
        return CrossEngineHealthReport(
            passed=True,
            checks=[CheckResult("no_engines", True, "No external engines reporting", None)],
        )

    # 0. Minimum reporting engines
    engines_reporting = len(engine_results)
    enough_engines = engines_reporting >= 2
    checks.append(CheckResult(
        name="engines_reporting",
        passed=enough_engines,
        detail=f"{engines_reporting} engines reporting (minimum 2 for synthesis confidence)",
        value=engines_reporting,
    ))
    if not enough_engines:
        warnings.append(
            f"Low engine coverage: only {engines_reporting} engine reporting."
        )

    # 1. Regime consensus
    our_regime = _canonical_regime(regime_context.get("regime", "unknown"))
    engine_regimes = {}
    for er in engine_results:
        name = er.get("engine_name", "unknown")
        regime = er.get("regime")
        if regime:
            engine_regimes[name] = _canonical_regime(regime)

    if engine_regimes:
        unique_regimes = set(engine_regimes.values())
        if our_regime != "unknown":
            unique_regimes.add(our_regime)

        consensus_ok = len(unique_regimes) <= 1
        regime_detail = ", ".join(f"{k}={v}" for k, v in sorted(engine_regimes.items()))
        if our_regime != "unknown":
            regime_detail = f"screener={our_regime}, " + regime_detail
        checks.append(CheckResult(
            name="regime_consensus",
            passed=consensus_ok,
            detail=f"Regimes: {regime_detail}",
            value=len(unique_regimes),
        ))
        if not consensus_ok:
            warnings.append(
                f"Regime divergence: {len(unique_regimes)} distinct regimes — {regime_detail}."
            )

    # 2. Pick overlap (convergence health)
    ticker_engines: dict[str, list[str]] = {}
    for er in engine_results:
        name = er.get("engine_name", "unknown")
        picks = er.get("picks", [])
        if isinstance(picks, list):
            for pick in picks:
                ticker = pick.get("ticker", "") if isinstance(pick, dict) else ""
                if ticker:
                    ticker_engines.setdefault(ticker, []).append(name)

    convergent = {t: engines for t, engines in ticker_engines.items() if len(engines) >= 2}
    total_unique = len(ticker_engines)
    overlap_count = len(convergent)
    overlap_ok = overlap_count >= 1 if total_unique > 0 else False
    checks.append(CheckResult(
        name="pick_overlap",
        passed=overlap_ok,
        detail=f"{overlap_count} tickers in 2+ engines out of {total_unique} unique",
        value=overlap_count,
    ))
    if not overlap_ok:
        if total_unique > 0:
            warnings.append(
                f"Zero pick convergence: {total_unique} unique tickers across engines, "
                "none appearing in 2+."
            )
        else:
            warnings.append("Zero pick convergence: no tickers produced by reporting engines.")

    # 3. Price consistency for overlapping tickers
    if convergent:
        price_mismatches = []
        for ticker, engines in convergent.items():
            prices = []
            for er in engine_results:
                if er.get("engine_name") in engines:
                    for pick in er.get("picks", []):
                        if isinstance(pick, dict) and pick.get("ticker") == ticker:
                            ep = pick.get("entry_price")
                            if ep and ep > 0:
                                prices.append((er["engine_name"], ep))
            if len(prices) >= 2:
                min_p = min(p for _, p in prices)
                max_p = max(p for _, p in prices)
                if min_p > 0 and (max_p - min_p) / min_p > 0.02:
                    price_mismatches.append(
                        f"{ticker}: {', '.join(f'{n}=${p:.2f}' for n, p in prices)}"
                    )

        price_ok = len(price_mismatches) == 0
        checks.append(CheckResult(
            name="price_consistency",
            passed=price_ok,
            detail=f"{len(price_mismatches)} mismatches (>2% spread)" if price_mismatches
                   else "All convergent tickers within 2% price spread",
            value=len(price_mismatches),
        ))
        if not price_ok:
            warnings.append(
                f"Price inconsistency in {len(price_mismatches)} convergent tickers: "
                + "; ".join(price_mismatches[:3])
            )

    # 4. Universe breadth per engine
    breadth_issues = []
    for er in engine_results:
        name = er.get("engine_name", "unknown")
        screened = er.get("candidates_screened", 0)
        picks_count = len(er.get("picks", []))
        if screened > 0 and screened < 20:
            breadth_issues.append(f"{name}: only {screened} screened")
        elif picks_count == 0 and screened == 0:
            breadth_issues.append(f"{name}: no picks or screening data")

    breadth_ok = len(breadth_issues) == 0
    engine_summary = ", ".join(
        f"{er.get('engine_name', '?')}={er.get('candidates_screened', '?')}"
        for er in engine_results
    )
    checks.append(CheckResult(
        name="universe_breadth",
        passed=breadth_ok,
        detail=f"Screened counts: {engine_summary}" if engine_summary else "No engine data",
        value=len(breadth_issues),
    ))
    if not breadth_ok:
        warnings.append(f"Universe breadth issues: {'; '.join(breadth_issues)}")

    all_passed = all(c.passed for c in checks)
    return CrossEngineHealthReport(passed=all_passed, checks=checks, warnings=warnings)


def _business_days_between(start: date, end: date) -> int:
    """Count business days (Mon-Fri) between two dates, exclusive of start."""
    if start >= end:
        return 0
    count = 0
    current = start + timedelta(days=1)
    while current <= end:
        if current.weekday() < 5:  # Mon=0..Fri=4
            count += 1
        current += timedelta(days=1)
    return count
