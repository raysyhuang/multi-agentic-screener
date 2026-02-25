"""MCP data connector client — enriches pipeline with institutional-grade data.

Connects to MCP servers configured in the project's .mcp.json via HTTP
streaming (SSE) transport. Each server exposes tools that this client
wraps as typed async methods for the DataAggregator to call.

Design principles:
  - Fail-open: MCP data is supplementary enrichment. If a connector is
    unavailable, the caller gets an empty result — never an exception.
  - Circuit-breaker aware: each MCP server is tracked separately.
  - Cache-friendly: results carry a suggested TTL for the DataCache.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx

from src.config import get_settings
from src.data.circuit_breaker import APICircuitBreaker

logger = logging.getLogger(__name__)

# SSE/HTTP request defaults
MCP_REQUEST_TIMEOUT = 30.0  # seconds per tool call
MCP_MAX_RETRIES = 2
MCP_BACKOFF_BASE = 1.0  # seconds


def _load_mcp_config() -> dict[str, dict]:
    """Load MCP server definitions from .mcp.json at project root."""
    config_path = Path(__file__).resolve().parent.parent.parent / ".mcp.json"
    if not config_path.exists():
        logger.info("No .mcp.json found — MCP connectors disabled")
        return {}
    try:
        data = json.loads(config_path.read_text())
        servers = data.get("mcpServers", {})
        logger.info("Loaded %d MCP server(s) from %s", len(servers), config_path)
        return servers
    except Exception as e:
        logger.warning("Failed to parse .mcp.json: %s", e)
        return {}


class MCPClient:
    """Async client for MCP financial data servers.

    Provides typed methods that map to high-value MCP tool calls.
    All methods return dicts/lists and never raise — they log warnings
    and return empty results on failure.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._servers = _load_mcp_config()
        self._enabled_servers: set[str] = set()
        self._circuit_breaker = APICircuitBreaker(
            failure_threshold=3, cooldown_seconds=300,
        )

        # Determine which servers are enabled based on config
        enabled_list = settings.mcp_enabled_providers
        if enabled_list:
            requested = {s.strip() for s in enabled_list.split(",") if s.strip()}
            self._enabled_servers = requested & set(self._servers)
            if requested - self._enabled_servers:
                logger.warning(
                    "MCP providers requested but not in .mcp.json: %s",
                    requested - self._enabled_servers,
                )
        else:
            # Default: enable all servers found in .mcp.json
            self._enabled_servers = set(self._servers)

        if self._enabled_servers:
            logger.info("MCP connectors active: %s", sorted(self._enabled_servers))

    @property
    def available(self) -> bool:
        """True if at least one MCP server is configured and enabled."""
        return bool(self._enabled_servers)

    @property
    def active_servers(self) -> list[str]:
        """Names of currently enabled MCP servers."""
        return sorted(self._enabled_servers)

    async def _call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Invoke an MCP tool via HTTP and return the parsed result.

        Returns None on any failure (timeout, server error, parse error).
        """
        if server_name not in self._enabled_servers:
            return None
        if self._circuit_breaker.is_open(f"mcp:{server_name}"):
            logger.debug("MCP circuit open for %s, skipping", server_name)
            return None

        server_cfg = self._servers[server_name]
        base_url = server_cfg["url"]

        # MCP HTTP transport: POST to /tools/call
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {},
            },
        }

        for attempt in range(MCP_MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=MCP_REQUEST_TIMEOUT) as client:
                    resp = await client.post(
                        base_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    if resp.status_code == 429:
                        delay = MCP_BACKOFF_BASE * (2 ** attempt)
                        logger.warning(
                            "MCP %s rate limited (attempt %d/%d) — retrying in %.1fs",
                            server_name, attempt + 1, MCP_MAX_RETRIES, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()

                result = resp.json()
                self._circuit_breaker.record_success(f"mcp:{server_name}")

                # MCP JSON-RPC response: result is in result.content
                if "result" in result:
                    content = result["result"].get("content", [])
                    # Extract text content from MCP response blocks
                    texts = [
                        block.get("text", "")
                        for block in content
                        if block.get("type") == "text"
                    ]
                    if texts:
                        # Try to parse as JSON, fall back to raw text
                        combined = "\n".join(texts)
                        try:
                            return json.loads(combined)
                        except json.JSONDecodeError:
                            return combined
                    return content
                elif "error" in result:
                    logger.warning(
                        "MCP %s tool %s returned error: %s",
                        server_name, tool_name, result["error"],
                    )
                    self._circuit_breaker.record_failure(f"mcp:{server_name}")
                    return None
                return result

            except httpx.TimeoutException:
                logger.warning(
                    "MCP %s timed out calling %s (attempt %d/%d)",
                    server_name, tool_name, attempt + 1, MCP_MAX_RETRIES,
                )
            except Exception as e:
                logger.warning(
                    "MCP %s error calling %s: %s (attempt %d/%d)",
                    server_name, tool_name, e, attempt + 1, MCP_MAX_RETRIES,
                )
                break  # Non-retryable

        self._circuit_breaker.record_failure(f"mcp:{server_name}")
        return None

    # ------------------------------------------------------------------
    # High-level enrichment methods (used by DataAggregator)
    # ------------------------------------------------------------------

    async def get_fundamentals(self, ticker: str) -> dict:
        """Fetch fundamental data from available MCP providers.

        Tries multiple providers in priority order and merges results.
        Returns a dict with keys: valuation, estimates, ownership, credit.
        """
        result: dict[str, Any] = {}

        # Morningstar — fair value, moat, star rating
        morningstar = await self._call_tool(
            "morningstar", "get_stock_analysis",
            {"ticker": ticker},
        )
        if morningstar:
            result["morningstar"] = morningstar

        # S&P Global / Capital IQ — comps, institutional ownership
        spglobal = await self._call_tool(
            "sp-global", "get_company_fundamentals",
            {"identifier": ticker},
        )
        if spglobal:
            result["sp_global"] = spglobal

        # FactSet — estimates, ownership chain, supply chain
        factset = await self._call_tool(
            "factset", "get_company_data",
            {"ticker": ticker},
        )
        if factset:
            result["factset"] = factset

        # Daloopa — granular KPI extraction
        daloopa = await self._call_tool(
            "daloopa", "get_company_kpis",
            {"ticker": ticker},
        )
        if daloopa:
            result["daloopa"] = daloopa

        return result

    async def get_earnings_context(self, ticker: str) -> dict:
        """Fetch earnings-related data — transcripts, sentiment, estimates.

        Enriches the catalyst signal model and earnings event exclusion.
        """
        result: dict[str, Any] = {}

        # Aiera — earnings call transcripts, event sentiment
        aiera = await self._call_tool(
            "aiera", "get_earnings_transcript",
            {"ticker": ticker},
        )
        if aiera:
            result["aiera_transcript"] = aiera

        # FactSet — consensus estimates, revisions
        estimates = await self._call_tool(
            "factset", "get_consensus_estimates",
            {"ticker": ticker},
        )
        if estimates:
            result["consensus_estimates"] = estimates

        return result

    async def get_news(self, ticker: str) -> list[dict]:
        """Fetch real-time news from MT Newswires.

        Supplements existing Polygon/FMP news with institutional wire feed.
        """
        news = await self._call_tool(
            "mtnewswires", "get_news",
            {"ticker": ticker, "limit": 20},
        )
        if isinstance(news, list):
            return news
        return []

    async def get_macro_enrichment(self) -> dict:
        """Fetch macro data from LSEG to enrich regime detection.

        Provides yield curves, credit spreads, FX carry, and macro dashboards
        beyond what FRED offers.
        """
        result: dict[str, Any] = {}

        # LSEG — fixed income, macro indicators
        macro = await self._call_tool(
            "lseg", "get_macro_dashboard", {},
        )
        if macro:
            result["lseg_macro"] = macro

        yields = await self._call_tool(
            "lseg", "get_yield_curve", {},
        )
        if yields:
            result["lseg_yields"] = yields

        credit = await self._call_tool(
            "lseg", "get_credit_spreads", {},
        )
        if credit:
            result["lseg_credit_spreads"] = credit

        return result

    async def get_credit_risk(self, ticker: str) -> dict:
        """Fetch credit analytics from Moody's for risk assessment."""
        moodys = await self._call_tool(
            "moodys", "get_credit_assessment",
            {"entity": ticker},
        )
        return moodys if isinstance(moodys, dict) else {}

    async def enrich_candidate(self, ticker: str) -> dict:
        """Run all per-ticker enrichments in parallel.

        Called by the DataAggregator for top candidates that pass
        the initial screen. Returns a merged enrichment payload.
        """
        fundamentals, earnings, news, credit = await asyncio.gather(
            self.get_fundamentals(ticker),
            self.get_earnings_context(ticker),
            self.get_news(ticker),
            self.get_credit_risk(ticker),
            return_exceptions=True,
        )

        enrichment: dict[str, Any] = {}
        if isinstance(fundamentals, dict) and fundamentals:
            enrichment["fundamentals"] = fundamentals
        if isinstance(earnings, dict) and earnings:
            enrichment["earnings_context"] = earnings
        if isinstance(news, list) and news:
            enrichment["mcp_news"] = news
        if isinstance(credit, dict) and credit:
            enrichment["credit_risk"] = credit

        return enrichment

    def get_stats(self) -> dict:
        """Return circuit breaker stats for all MCP providers."""
        return {
            "enabled_servers": sorted(self._enabled_servers),
            "circuit_breaker": self._circuit_breaker.get_stats(),
        }
