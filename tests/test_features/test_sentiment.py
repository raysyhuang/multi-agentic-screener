"""Tests for sentiment scoring."""

from src.features.sentiment import score_headline, score_news_batch


def test_positive_headline():
    score = score_headline("Company beats earnings estimates, stock surges")
    assert score > 0


def test_negative_headline():
    score = score_headline("Company faces bankruptcy amid fraud investigation")
    assert score < 0


def test_neutral_headline():
    score = score_headline("Company to report quarterly results next week")
    # Neutral — no strong keywords
    assert -0.3 <= score <= 0.3


def test_empty_headline():
    assert score_headline("") == 0.0
    assert score_headline(None) == 0.0


def test_score_bounded():
    score = score_headline("UPGRADE BUY STRONG GROWTH SURGES BEATS RECORD BULLISH OUTPERFORM")
    assert -1.0 <= score <= 1.0


def test_score_news_batch_empty():
    result = score_news_batch([])
    assert result["sentiment_score"] == 0.0
    assert result["article_count"] == 0


def test_score_news_batch_positive(sample_news):
    result = score_news_batch(sample_news, recency_hours=720)  # wide window for test data
    assert result["article_count"] >= 1
    assert isinstance(result["sentiment_score"], float)
    assert isinstance(result["positive_count"], int)
    assert isinstance(result["negative_count"], int)
