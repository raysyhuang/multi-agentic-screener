"""LLM router — dispatches to Claude (Anthropic) or GPT (OpenAI) based on model ID."""

from __future__ import annotations

import json
import logging
import time

import anthropic
import openai

from src.config import get_settings

logger = logging.getLogger(__name__)

# Cost per 1M tokens (USD) — updated as of 2025
_MODEL_COSTS: dict[str, tuple[float, float]] = {
    # (input_per_1M, output_per_1M)
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3-mini": (1.10, 4.40),
}


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate USD cost for an LLM call."""
    costs = _MODEL_COSTS.get(model)
    if not costs:
        # Fallback: match by prefix
        for key, val in _MODEL_COSTS.items():
            if model.startswith(key.split("-")[0]):
                costs = val
                break
    if not costs:
        return 0.0
    input_cost = (tokens_in / 1_000_000) * costs[0]
    output_cost = (tokens_out / 1_000_000) * costs[1]
    return round(input_cost + output_cost, 6)


async def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict | None = None,
    max_tokens: int = 2000,
    temperature: float = 0.3,
) -> dict:
    """Route an LLM call to the correct provider and return structured output.

    Args:
        model: Model identifier (e.g. "claude-sonnet-4-5-20250929", "gpt-4.1")
        system_prompt: System message
        user_prompt: User message with data
        response_schema: JSON schema for structured output (used as guidance)
        max_tokens: Max response tokens
        temperature: Sampling temperature

    Returns:
        dict with keys: content (str or dict), tokens_in, tokens_out, latency_ms, model
    """
    settings = get_settings()
    start = time.monotonic()

    if model.startswith("gemini"):
        raise ValueError(
            f"Gemini provider is disabled in MVP scope (model={model}). "
            "Gemini API is no longer available. Use Claude or GPT models instead."
        )
    elif model.startswith("claude"):
        result = await _call_anthropic(
            model, system_prompt, user_prompt, max_tokens, temperature, settings
        )
    elif model.startswith("gpt") or model.startswith("o"):
        result = await _call_openai(
            model, system_prompt, user_prompt, response_schema, max_tokens, temperature, settings
        )
    else:
        raise ValueError(f"Unknown model prefix: {model}")

    elapsed_ms = int((time.monotonic() - start) * 1000)
    result["latency_ms"] = elapsed_ms
    result["model"] = model

    tokens_in = result.get("tokens_in", 0)
    tokens_out = result.get("tokens_out", 0)
    cost = _estimate_cost(model, tokens_in, tokens_out)
    result["cost_usd"] = cost

    logger.info(
        "LLM call: model=%s, tokens_in=%d, tokens_out=%d, latency=%dms, cost=$%.4f",
        model, tokens_in, tokens_out, elapsed_ms, cost,
    )
    return result


async def _call_anthropic(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    settings,
) -> dict:
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    content = response.content[0].text
    # Try to parse as JSON
    parsed = _try_parse_json(content)

    return {
        "content": parsed if parsed else content,
        "raw": content,
        "tokens_in": response.usage.input_tokens,
        "tokens_out": response.usage.output_tokens,
    }


async def _call_openai(
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict | None,
    max_tokens: int,
    temperature: float,
    settings,
) -> dict:
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Use structured output if schema provided
    if response_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": response_schema,
                "strict": True,
            },
        }
    else:
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)

    content = response.choices[0].message.content
    parsed = _try_parse_json(content)

    return {
        "content": parsed if parsed else content,
        "raw": content,
        "tokens_in": response.usage.prompt_tokens,
        "tokens_out": response.usage.completion_tokens,
    }


def _try_parse_json(text: str) -> dict | None:
    """Attempt to extract JSON from LLM response."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        try:
            return json.loads(text[start:end].strip())
        except (json.JSONDecodeError, ValueError):
            pass

    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        try:
            return json.loads(text[start:end].strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Try finding JSON object boundaries
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None
