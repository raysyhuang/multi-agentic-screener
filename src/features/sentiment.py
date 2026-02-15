"""Sentiment feature engineering — news headline scoring."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

# Simple keyword-based sentiment scoring (deterministic, no LLM dependency).
# This is intentionally basic — LLM agents in L4 handle nuanced interpretation.

POSITIVE_KEYWORDS = {
    "upgrade", "beat", "beats", "exceeds", "raises", "raised", "surges", "soars",
    "breakout", "rally", "bullish", "outperform", "buy", "strong", "record",
    "growth", "accelerat", "expands", "partnership", "acquisition", "approved",
    "innovative", "breakthrough", "revenue growth", "profit", "dividend",
    "buyback", "repurchase",
}

NEGATIVE_KEYWORDS = {
    "downgrade", "miss", "misses", "cuts", "lowered", "plunges", "crashes",
    "bearish", "underperform", "sell", "weak", "decline", "loss", "losses",
    "lawsuit", "investigation", "recall", "fraud", "bankruptcy", "default",
    "layoffs", "restructuring", "warning", "concern", "debt", "dilution",
}

STRONG_POSITIVE = {"fda approval", "acquisition", "buyback", "record revenue", "beats estimates"}
STRONG_NEGATIVE = {"bankruptcy", "fraud", "sec investigation", "delisting", "default"}


def score_headline(title: str) -> float:
    """Score a single headline. Returns [-1.0, 1.0]."""
    if not title:
        return 0.0

    title_lower = title.lower()
    score = 0.0

    # Strong signals first
    for phrase in STRONG_POSITIVE:
        if phrase in title_lower:
            score += 0.5

    for phrase in STRONG_NEGATIVE:
        if phrase in title_lower:
            score -= 0.5

    # Keyword matching
    words = set(re.findall(r'\w+', title_lower))
    for kw in POSITIVE_KEYWORDS:
        if kw in words or kw in title_lower:
            score += 0.15

    for kw in NEGATIVE_KEYWORDS:
        if kw in words or kw in title_lower:
            score -= 0.15

    return max(-1.0, min(1.0, score))


def score_news_batch(articles: list[dict], recency_hours: int = 72) -> dict:
    """Score a batch of news articles with recency weighting.

    Returns:
      - sentiment_score: weighted average [-1.0, 1.0]
      - article_count: number of articles scored
      - positive_count: articles with positive sentiment
      - negative_count: articles with negative sentiment
      - sentiment_momentum: recent vs older sentiment shift
    """
    if not articles:
        return {
            "sentiment_score": 0.0,
            "article_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "sentiment_momentum": 0.0,
        }

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=recency_hours)

    scores = []
    weights = []
    recent_scores = []
    older_scores = []

    for article in articles:
        title = article.get("title", "")
        s = score_headline(title)

        # Parse published date for recency weighting
        pub_str = article.get("published_utc") or article.get("publishedDate", "")
        try:
            if "T" in pub_str:
                pub_date = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            else:
                pub_date = datetime.fromisoformat(pub_str)
                pub_date = pub_date.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pub_date = now

        if pub_date < cutoff:
            continue

        hours_ago = max(1, (now - pub_date).total_seconds() / 3600)
        weight = 1.0 / (1.0 + hours_ago / 24)  # decay over days

        scores.append(s)
        weights.append(weight)

        # Split for momentum calculation
        if hours_ago <= 24:
            recent_scores.append(s)
        else:
            older_scores.append(s)

    if not scores:
        return {
            "sentiment_score": 0.0,
            "article_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "sentiment_momentum": 0.0,
        }

    # Weighted average
    total_weight = sum(weights)
    weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

    # Momentum: recent avg - older avg
    recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
    older_avg = sum(older_scores) / len(older_scores) if older_scores else 0.0
    momentum = recent_avg - older_avg

    positive_count = sum(1 for s in scores if s > 0.1)
    negative_count = sum(1 for s in scores if s < -0.1)

    return {
        "sentiment_score": round(weighted_score, 4),
        "article_count": len(scores),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "sentiment_momentum": round(momentum, 4),
    }
