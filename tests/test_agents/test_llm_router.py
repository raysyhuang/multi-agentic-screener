"""Tests for LLM router — JSON parsing logic (no actual API calls)."""

import pytest

from src.agents.llm_router import _try_parse_json, call_llm


def test_parse_direct_json():
    result = _try_parse_json('{"key": "value", "num": 42}')
    assert result == {"key": "value", "num": 42}


def test_parse_json_in_markdown():
    text = 'Here is the result:\n```json\n{"ticker": "AAPL", "score": 75}\n```\nDone.'
    result = _try_parse_json(text)
    assert result["ticker"] == "AAPL"
    assert result["score"] == 75


def test_parse_json_with_text_prefix():
    text = 'Based on my analysis:\n{"verdict": "PROCEED", "confidence": 80}'
    result = _try_parse_json(text)
    assert result["verdict"] == "PROCEED"


def test_parse_invalid_json():
    result = _try_parse_json("This is not JSON at all.")
    assert result is None


def test_parse_empty_string():
    result = _try_parse_json("")
    assert result is None


def test_parse_nested_json():
    text = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
    result = _try_parse_json(text)
    assert result["outer"]["inner"] == [1, 2, 3]
    assert result["flag"] is True


@pytest.mark.asyncio
async def test_gemini_model_raises_disabled():
    """Gemini provider is disabled per MVP scope."""
    with pytest.raises(ValueError, match="Gemini provider is disabled"):
        await call_llm(
            model="gemini-2.0-flash",
            system_prompt="test",
            user_prompt="test",
        )
