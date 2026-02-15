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
    "gpt-5.2": (2.0, 8.0),
    "gpt-5.2-nano": (0.10, 0.40),
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
        model: Model identifier (e.g. "claude-sonnet-4-5-20250929", "gpt-5.2")
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
        "max_completion_tokens": max_tokens,
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


async def call_llm_with_tools(
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: "ToolRegistry",
    max_tokens: int = 2000,
    temperature: float = 0.3,
    max_tool_rounds: int = 3,
) -> dict:
    """Multi-turn LLM call with tool use.

    Sends the prompt, and if the LLM returns tool_use blocks, executes
    the tools and sends results back. Repeats until the LLM returns a
    text response or max_tool_rounds is reached.

    Args:
        model: Model identifier
        system_prompt: System message
        user_prompt: User message
        tools: ToolRegistry with available tools
        max_tokens: Max response tokens per turn
        temperature: Sampling temperature
        max_tool_rounds: Max number of tool-use rounds (prevents runaway loops)

    Returns:
        dict with keys: content, raw, tokens_in, tokens_out, latency_ms, model, tool_calls
    """
    from src.agents.tools import ToolRegistry  # avoid circular import at module level

    settings = get_settings()
    start = time.monotonic()
    total_tokens_in = 0
    total_tokens_out = 0
    tool_call_log: list[dict] = []

    if model.startswith("claude"):
        result = await _call_anthropic_with_tools(
            model, system_prompt, user_prompt, tools,
            max_tokens, temperature, max_tool_rounds, settings,
            tool_call_log,
        )
    elif model.startswith("gpt") or model.startswith("o"):
        result = await _call_openai_with_tools(
            model, system_prompt, user_prompt, tools,
            max_tokens, temperature, max_tool_rounds, settings,
            tool_call_log,
        )
    else:
        raise ValueError(f"Unknown model prefix for tool-use: {model}")

    elapsed_ms = int((time.monotonic() - start) * 1000)
    tokens_in = result.get("tokens_in", 0)
    tokens_out = result.get("tokens_out", 0)
    cost = _estimate_cost(model, tokens_in, tokens_out)

    result["latency_ms"] = elapsed_ms
    result["model"] = model
    result["cost_usd"] = cost
    result["tool_calls"] = tool_call_log

    logger.info(
        "LLM tool-use call: model=%s, tokens_in=%d, tokens_out=%d, "
        "tool_rounds=%d, latency=%dms, cost=$%.4f",
        model, tokens_in, tokens_out, len(tool_call_log), elapsed_ms, cost,
    )
    return result


async def _call_anthropic_with_tools(
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: "ToolRegistry",
    max_tokens: int,
    temperature: float,
    max_tool_rounds: int,
    settings,
    tool_call_log: list[dict],
) -> dict:
    """Anthropic multi-turn tool-use loop."""
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    tool_schemas = tools.list_schemas_anthropic()

    messages = [{"role": "user", "content": user_prompt}]
    total_in = 0
    total_out = 0

    for round_num in range(max_tool_rounds + 1):
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
        )

        total_in += response.usage.input_tokens
        total_out += response.usage.output_tokens

        # Check if response contains tool use
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if not tool_use_blocks or round_num >= max_tool_rounds:
            # Final text response
            content_text = text_blocks[0].text if text_blocks else ""
            parsed = _try_parse_json(content_text)
            return {
                "content": parsed if parsed else content_text,
                "raw": content_text,
                "tokens_in": total_in,
                "tokens_out": total_out,
            }

        # Execute tools and build tool results
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tool_block in tool_use_blocks:
            tool_result = await tools.execute(tool_block.name, tool_block.input)
            tool_call_log.append({
                "tool": tool_block.name,
                "input": tool_block.input,
                "output": tool_result,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": json.dumps(tool_result, default=str),
            })
        messages.append({"role": "user", "content": tool_results})

    # Should not reach here, but return empty if it does
    return {"content": "", "raw": "", "tokens_in": total_in, "tokens_out": total_out}


async def _call_openai_with_tools(
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: "ToolRegistry",
    max_tokens: int,
    temperature: float,
    max_tool_rounds: int,
    settings,
    tool_call_log: list[dict],
) -> dict:
    """OpenAI multi-turn tool-use loop."""
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    tool_schemas = tools.list_schemas_openai()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    total_in = 0
    total_out = 0

    for round_num in range(max_tool_rounds + 1):
        kwargs = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        total_in += response.usage.prompt_tokens
        total_out += response.usage.completion_tokens

        if choice.finish_reason != "tool_calls" or round_num >= max_tool_rounds:
            content = choice.message.content or ""
            parsed = _try_parse_json(content)
            return {
                "content": parsed if parsed else content,
                "raw": content,
                "tokens_in": total_in,
                "tokens_out": total_out,
            }

        # Execute tool calls
        messages.append(choice.message)
        for tool_call in choice.message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            tool_result = await tools.execute(tool_call.function.name, args)
            tool_call_log.append({
                "tool": tool_call.function.name,
                "input": args,
                "output": tool_result,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, default=str),
            })

    return {"content": "", "raw": "", "tokens_in": total_in, "tokens_out": total_out}


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
