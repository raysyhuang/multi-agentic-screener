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

# Engine-specific breadth expectations. External engines intentionally screen
# at different scales; one global threshold creates false warnings.
MIN_SCREENED_BY_ENGINE: dict[str, int] = {
    "koocore_d": 5,
    "gemini_stst": 5,
    "top3_7d": 10,
}
DEFAULT_MIN_SCREENED = 20


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
