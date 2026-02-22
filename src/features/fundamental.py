"""Fundamental feature engineering — earnings, insider activity, institutional flow."""

from __future__ import annotations

from datetime import date, timedelta


def _to_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def score_earnings_surprise(earnings: list[dict]) -> dict:
    """Score recent earnings surprises.

    Returns features:
      - last_surprise_pct: most recent EPS surprise %
      - avg_surprise_pct: average of last 4 surprises
      - beat_streak: consecutive beats
      - earnings_momentum: improving vs deteriorating trend
    """
    if not earnings:
        return {
            "last_surprise_pct": None,
            "avg_surprise_pct": None,
            "beat_streak": 0,
            "earnings_momentum": 0.0,
        }

    surprises = []
    for e in earnings[:4]:
        actual = e.get("actualEarningResult") or e.get("actual")
        estimated = e.get("estimatedEarning") or e.get("estimate")
        if actual is not None and estimated is not None and estimated != 0:
            surprises.append((actual - estimated) / abs(estimated) * 100)

    if not surprises:
        return {
            "last_surprise_pct": None,
            "avg_surprise_pct": None,
            "beat_streak": 0,
            "earnings_momentum": 0.0,
        }

    # Beat streak
    beat_streak = 0
    for s in surprises:
        if s > 0:
            beat_streak += 1
        else:
            break

    # Momentum: are surprises getting bigger or smaller?
    momentum = 0.0
    if len(surprises) >= 2:
        momentum = surprises[0] - surprises[-1]  # positive = improving

    return {
        "last_surprise_pct": surprises[0],
        "avg_surprise_pct": sum(surprises) / len(surprises),
        "beat_streak": beat_streak,
        "earnings_momentum": momentum,
    }


def score_insider_activity(transactions: list[dict], lookback_days: int = 90) -> dict:
    """Score insider buying/selling activity.

    Returns:
      - insider_buy_count: number of buys in lookback period
      - insider_sell_count: number of sells
      - insider_net_ratio: (buys - sells) / total, range [-1, 1]
      - insider_buy_value: total dollar value of buys
    """
    if not transactions:
        return {
            "insider_buy_count": 0,
            "insider_sell_count": 0,
            "insider_net_ratio": 0.0,
            "insider_buy_value": 0.0,
        }

    cutoff = date.today() - timedelta(days=lookback_days)
    buys, sells = 0, 0
    buy_value = 0.0

    for txn in transactions:
        txn_date_str = txn.get("transactionDate") or txn.get("filingDate", "")
        if not txn_date_str:
            continue
        try:
            txn_date = date.fromisoformat(txn_date_str[:10])
        except ValueError:
            continue
        if txn_date < cutoff:
            continue

        txn_type = (txn.get("transactionType") or txn.get("acquistionOrDisposition", "")).upper()
        shares = abs(txn.get("securitiesTransacted") or txn.get("shares", 0))
        price = txn.get("price", 0) or 0

        if "P" in txn_type or "BUY" in txn_type or "PURCHASE" in txn_type:
            buys += 1
            buy_value += shares * price
        elif "S" in txn_type or "SELL" in txn_type or "SALE" in txn_type:
            sells += 1

    total = buys + sells
    net_ratio = (buys - sells) / total if total > 0 else 0.0

    return {
        "insider_buy_count": buys,
        "insider_sell_count": sells,
        "insider_net_ratio": net_ratio,
        "insider_buy_value": buy_value,
    }


def score_institutional_flow(holders: list[dict]) -> dict:
    """Score institutional ownership changes.

    Returns:
      - institutional_holders_count: number of holders
      - top10_ownership_pct: aggregate % held by top 10
    """
    if not holders:
        return {
            "institutional_holders_count": 0,
            "top10_ownership_pct": 0.0,
        }

    top10 = holders[:10]
    total_shares = sum(h.get("shares", 0) for h in top10)

    return {
        "institutional_holders_count": len(holders),
        "top10_ownership_pct": 0.0,  # Would need total shares outstanding to compute
    }


def score_analyst_estimates(estimates: list[dict]) -> dict:
    """Summarize analyst EPS/revenue estimate direction for thesis grounding."""
    if not estimates:
        return {
            "eps_estimate_next": None,
            "revenue_estimate_next": None,
            "eps_revision_pct": 0.0,
            "revenue_revision_pct": 0.0,
        }

    current = estimates[0] if isinstance(estimates[0], dict) else {}
    previous = estimates[1] if len(estimates) > 1 and isinstance(estimates[1], dict) else {}

    eps_now = _to_float(
        current.get("estimatedEpsAvg")
        or current.get("epsEstimated")
        or current.get("eps_estimate")
    )
    eps_prev = _to_float(
        previous.get("estimatedEpsAvg")
        or previous.get("epsEstimated")
        or previous.get("eps_estimate")
    )

    rev_now = _to_float(
        current.get("estimatedRevenueAvg")
        or current.get("revenueEstimated")
        or current.get("revenue_estimate")
    )
    rev_prev = _to_float(
        previous.get("estimatedRevenueAvg")
        or previous.get("revenueEstimated")
        or previous.get("revenue_estimate")
    )

    eps_revision_pct = 0.0
    if eps_now is not None and eps_prev and eps_prev != 0:
        eps_revision_pct = ((eps_now - eps_prev) / abs(eps_prev)) * 100

    rev_revision_pct = 0.0
    if rev_now is not None and rev_prev and rev_prev != 0:
        rev_revision_pct = ((rev_now - rev_prev) / abs(rev_prev)) * 100

    return {
        "eps_estimate_next": eps_now,
        "revenue_estimate_next": rev_now,
        "eps_revision_pct": round(eps_revision_pct, 3),
        "revenue_revision_pct": round(rev_revision_pct, 3),
    }


def score_financial_ratios(ratios: dict) -> dict:
    """Score basic value/quality profile from P/E, P/B, and debt-to-equity."""
    if not ratios:
        return {
            "pe_ratio": None,
            "pb_ratio": None,
            "debt_to_equity": None,
            "value_score": 50.0,
            "quality_score": 50.0,
            "mean_reversion_ok": False,
        }

    pe = _to_float(
        ratios.get("priceEarningsRatio")
        or ratios.get("peRatio")
        or ratios.get("pe")
    )
    pb = _to_float(
        ratios.get("priceToBookRatio")
        or ratios.get("pbRatio")
        or ratios.get("pb")
    )
    debt_to_equity = _to_float(
        ratios.get("debtEquityRatio")
        or ratios.get("debtToEquity")
    )

    value_score = 50.0
    if pe is not None:
        if pe <= 12:
            value_score += 30
        elif pe <= 18:
            value_score += 20
        elif pe >= 35:
            value_score -= 20
    if pb is not None:
        if pb <= 2.0:
            value_score += 20
        elif pb <= 3.0:
            value_score += 10
        elif pb >= 5.0:
            value_score -= 20
    value_score = max(0.0, min(100.0, value_score))

    quality_score = 50.0
    if debt_to_equity is not None:
        if debt_to_equity <= 0.8:
            quality_score += 35
        elif debt_to_equity <= 1.5:
            quality_score += 15
        elif debt_to_equity >= 2.5:
            quality_score -= 30
    quality_score = max(0.0, min(100.0, quality_score))

    mean_reversion_ok = value_score >= 55 and quality_score >= 50

    return {
        "pe_ratio": pe,
        "pb_ratio": pb,
        "debt_to_equity": debt_to_equity,
        "value_score": round(value_score, 2),
        "quality_score": round(quality_score, 2),
        "mean_reversion_ok": mean_reversion_ok,
    }


def days_to_next_earnings(earnings_calendar: list[dict], ticker: str) -> int | None:
    """How many days until this ticker's next earnings report."""
    today = date.today()
    for entry in earnings_calendar:
        if entry.get("symbol", "").upper() == ticker.upper():
            report_date_str = entry.get("date", "")
            try:
                report_date = date.fromisoformat(report_date_str[:10])
                if report_date >= today:
                    return (report_date - today).days
            except ValueError:
                continue
    return None
