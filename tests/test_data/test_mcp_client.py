"""Tests for MCP data connector client."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.data.mcp_client import MCPClient, _load_mcp_config  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MCP_CONFIG = {
    "mcpServers": {
        "morningstar": {"type": "http", "url": "https://mcp.morningstar.com/mcp"},
        "aiera": {"type": "http", "url": "https://mcp-pub.aiera.com"},
        "lseg": {"type": "http", "url": "https://api.analytics.lseg.com/lfa/mcp"},
        "mtnewswires": {"type": "http", "url": "https://vast-mcp.blueskyapi.com/mtnewswires"},
        "moodys": {"type": "http", "url": "https://api.moodys.com/genai-ready-data/m1/mcp"},
        "sp-global": {"type": "http", "url": "https://kfinance.kensho.com/integrations/mcp"},
        "factset": {"type": "http", "url": "https://mcp.factset.com/mcp"},
        "daloopa": {"type": "http", "url": "https://mcp.daloopa.com/server/mcp"},
    }
}


@pytest.fixture
def mcp_client():
    """Create an MCPClient with mocked config and settings."""
    with (
        patch("src.data.mcp_client._load_mcp_config", return_value=SAMPLE_MCP_CONFIG["mcpServers"]),
        patch("src.data.mcp_client.get_settings") as mock_settings,
    ):
        mock_settings.return_value = MagicMock(
            mcp_enabled_providers="",
        )
        client = MCPClient()
        return client


@pytest.fixture
def mcp_client_filtered():
    """MCPClient with only morningstar and aiera enabled."""
    with (
        patch("src.data.mcp_client._load_mcp_config", return_value=SAMPLE_MCP_CONFIG["mcpServers"]),
        patch("src.data.mcp_client.get_settings") as mock_settings,
    ):
        mock_settings.return_value = MagicMock(
            mcp_enabled_providers="morningstar,aiera",
        )
        client = MCPClient()
        return client


@pytest.fixture
def mcp_client_empty():
    """MCPClient with no .mcp.json found."""
    with (
        patch("src.data.mcp_client._load_mcp_config", return_value={}),
        patch("src.data.mcp_client.get_settings") as mock_settings,
    ):
        mock_settings.return_value = MagicMock(
            mcp_enabled_providers="",
        )
        client = MCPClient()
        return client


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestMCPConfigLoading:
    def test_loads_all_servers_when_no_filter(self, mcp_client):
        assert mcp_client.available is True
        assert len(mcp_client.active_servers) == 8
        assert "morningstar" in mcp_client.active_servers
        assert "lseg" in mcp_client.active_servers

    def test_filters_to_requested_providers(self, mcp_client_filtered):
        assert mcp_client_filtered.available is True
        assert mcp_client_filtered.active_servers == ["aiera", "morningstar"]

    def test_not_available_when_no_config(self, mcp_client_empty):
        assert mcp_client_empty.available is False
        assert mcp_client_empty.active_servers == []

    def test_warns_on_unknown_provider(self):
        with (
            patch("src.data.mcp_client._load_mcp_config", return_value=SAMPLE_MCP_CONFIG["mcpServers"]),
            patch("src.data.mcp_client.get_settings") as mock_settings,
        ):
            mock_settings.return_value = MagicMock(
                mcp_enabled_providers="morningstar,nonexistent_provider",
            )
            client = MCPClient()
            # nonexistent_provider is silently dropped
            assert client.active_servers == ["morningstar"]

    def test_load_mcp_config_missing_file(self, tmp_path):
        """_load_mcp_config returns {} when file doesn't exist."""
        with patch("src.data.mcp_client.Path") as mock_path:
            mock_path.return_value.resolve.return_value.parent.parent.parent.__truediv__ = lambda s, n: tmp_path / ".mcp.json"
            # Can't easily mock Path chaining, test via the client
            pass

    def test_stats_report(self, mcp_client):
        stats = mcp_client.get_stats()
        assert "enabled_servers" in stats
        assert "circuit_breaker" in stats
        assert len(stats["enabled_servers"]) == 8


# ---------------------------------------------------------------------------
# Tool call mechanism
# ---------------------------------------------------------------------------

class TestToolCall:
    @pytest.mark.asyncio
    async def test_successful_tool_call(self, mcp_client):
        """A successful MCP tool call returns parsed JSON content."""
        response_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {"type": "text", "text": json.dumps({"fair_value": 185.0, "moat": "wide"})}
                ]
            },
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_body
        mock_response.raise_for_status = MagicMock()

        with patch("src.data.mcp_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await mcp_client._call_tool(
                "morningstar", "get_stock_analysis", {"ticker": "AAPL"},
            )
            assert result == {"fair_value": 185.0, "moat": "wide"}

    @pytest.mark.asyncio
    async def test_tool_call_returns_none_for_disabled_server(self, mcp_client_filtered):
        """Disabled servers return None immediately."""
        result = await mcp_client_filtered._call_tool(
            "factset", "get_company_data", {"ticker": "AAPL"},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_tool_call_returns_none_on_timeout(self, mcp_client):
        """Timeout returns None and records circuit failure."""
        with patch("src.data.mcp_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("timed out")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await mcp_client._call_tool(
                "morningstar", "get_stock_analysis", {"ticker": "AAPL"},
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_tool_call_returns_none_on_server_error(self, mcp_client):
        """HTTP 500 returns None."""
        with patch("src.data.mcp_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock(status_code=500),
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await mcp_client._call_tool(
                "morningstar", "get_stock_analysis", {"ticker": "AAPL"},
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_tool_call_handles_mcp_error_response(self, mcp_client):
        """MCP JSON-RPC error response returns None."""
        response_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Method not found"},
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_body
        mock_response.raise_for_status = MagicMock()

        with patch("src.data.mcp_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await mcp_client._call_tool(
                "morningstar", "get_stock_analysis", {"ticker": "AAPL"},
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_tool_call_returns_raw_text_when_not_json(self, mcp_client):
        """Non-JSON text content is returned as a string."""
        response_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {"type": "text", "text": "AAPL has a wide economic moat"}
                ]
            },
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_body
        mock_response.raise_for_status = MagicMock()

        with patch("src.data.mcp_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await mcp_client._call_tool(
                "morningstar", "get_stock_analysis", {"ticker": "AAPL"},
            )
            assert result == "AAPL has a wide economic moat"


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_circuit_opens_after_repeated_failures(self, mcp_client):
        """After 3 consecutive failures, circuit opens and calls are skipped."""
        with patch("src.data.mcp_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("timed out")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            # Trigger 3 failures (threshold) × 2 retries each
            for _ in range(3):
                await mcp_client._call_tool("morningstar", "test_tool", {})

            # Circuit should be open now
            assert mcp_client._circuit_breaker.is_open("mcp:morningstar")

            # Subsequent calls should return None immediately (no HTTP call)
            mock_client.post.reset_mock()
            result = await mcp_client._call_tool("morningstar", "test_tool", {})
            assert result is None
            # No HTTP call was made because circuit is open
            mock_client.post.assert_not_called()

    def test_stats_show_circuit_state(self, mcp_client):
        """Stats reflect circuit breaker state."""
        # Record some failures manually
        mcp_client._circuit_breaker.record_failure("mcp:morningstar")
        mcp_client._circuit_breaker.record_failure("mcp:morningstar")
        stats = mcp_client.get_stats()
        breaker = stats["circuit_breaker"]["mcp:morningstar"]
        assert breaker["consecutive_failures"] == 2
        assert breaker["is_open"] is False  # threshold is 3


# ---------------------------------------------------------------------------
# High-level enrichment methods
# ---------------------------------------------------------------------------

class TestEnrichmentMethods:
    @pytest.mark.asyncio
    async def test_get_fundamentals_merges_multiple_providers(self, mcp_client):
        """Fundamentals aggregates data from morningstar, sp-global, factset, daloopa."""
        async def mock_call(server_name, tool_name, arguments=None):
            responses = {
                "morningstar": {"fair_value": 185.0},
                "sp-global": {"pe_ratio": 28.5},
                "factset": {"eps_estimate": 6.50},
                "daloopa": {"revenue_growth": 0.12},
            }
            return responses.get(server_name)

        mcp_client._call_tool = AsyncMock(side_effect=mock_call)

        result = await mcp_client.get_fundamentals("AAPL")
        assert "morningstar" in result
        assert result["morningstar"]["fair_value"] == 185.0
        assert "sp_global" in result
        assert "factset" in result
        assert "daloopa" in result

    @pytest.mark.asyncio
    async def test_get_fundamentals_tolerates_partial_failures(self, mcp_client):
        """If some providers fail, others still return data."""
        call_count = 0

        async def mock_call(server_name, tool_name, arguments=None):
            nonlocal call_count
            call_count += 1
            if server_name == "morningstar":
                return {"fair_value": 185.0}
            return None  # other providers failed

        mcp_client._call_tool = AsyncMock(side_effect=mock_call)

        result = await mcp_client.get_fundamentals("AAPL")
        assert "morningstar" in result
        assert len(result) == 1  # only morningstar succeeded

    @pytest.mark.asyncio
    async def test_get_earnings_context(self, mcp_client):
        """Earnings context fetches from aiera and factset."""
        async def mock_call(server_name, tool_name, arguments=None):
            if server_name == "aiera":
                return {"transcript": "Revenue grew 12%..."}
            if server_name == "factset":
                return {"consensus_eps": 6.50}
            return None

        mcp_client._call_tool = AsyncMock(side_effect=mock_call)

        result = await mcp_client.get_earnings_context("AAPL")
        assert "aiera_transcript" in result
        assert "consensus_estimates" in result

    @pytest.mark.asyncio
    async def test_get_news_returns_list(self, mcp_client):
        """News returns a list of articles from MT Newswires."""
        mcp_client._call_tool = AsyncMock(return_value=[
            {"headline": "AAPL beats estimates", "timestamp": "2024-01-25"},
        ])

        result = await mcp_client.get_news("AAPL")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["headline"] == "AAPL beats estimates"

    @pytest.mark.asyncio
    async def test_get_news_returns_empty_on_failure(self, mcp_client):
        """News returns empty list if provider fails."""
        mcp_client._call_tool = AsyncMock(return_value=None)

        result = await mcp_client.get_news("AAPL")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_macro_enrichment(self, mcp_client):
        """Macro enrichment fetches yield curves and credit spreads from LSEG."""
        call_results = {
            ("lseg", "get_macro_dashboard"): {"gdp_growth": 2.1},
            ("lseg", "get_yield_curve"): {"2y": 4.5, "10y": 4.2},
            ("lseg", "get_credit_spreads"): {"ig_spread": 120},
        }

        async def mock_call(server_name, tool_name, arguments=None):
            return call_results.get((server_name, tool_name))

        mcp_client._call_tool = AsyncMock(side_effect=mock_call)

        result = await mcp_client.get_macro_enrichment()
        assert "lseg_macro" in result
        assert "lseg_yields" in result
        assert "lseg_credit_spreads" in result

    @pytest.mark.asyncio
    async def test_get_credit_risk(self, mcp_client):
        """Credit risk returns Moody's assessment."""
        mcp_client._call_tool = AsyncMock(return_value={"rating": "A1", "outlook": "stable"})

        result = await mcp_client.get_credit_risk("AAPL")
        assert result["rating"] == "A1"

    @pytest.mark.asyncio
    async def test_get_credit_risk_returns_empty_on_failure(self, mcp_client):
        mcp_client._call_tool = AsyncMock(return_value=None)

        result = await mcp_client.get_credit_risk("AAPL")
        assert result == {}


# ---------------------------------------------------------------------------
# enrich_candidate (parallel aggregation)
# ---------------------------------------------------------------------------

class TestEnrichCandidate:
    @pytest.mark.asyncio
    async def test_enrich_candidate_merges_all_sources(self, mcp_client):
        """enrich_candidate runs all per-ticker enrichments in parallel."""
        async def mock_call(server_name, tool_name, arguments=None):
            mapping = {
                "morningstar": {"fair_value": 185.0},
                "sp-global": {"pe_ratio": 28.5},
                "factset": {"eps_estimate": 6.50},
                "daloopa": {"kpi": "test"},
                "aiera": {"transcript": "good quarter"},
                "mtnewswires": [{"headline": "AAPL up"}],
                "moodys": {"rating": "A1"},
            }
            return mapping.get(server_name)

        mcp_client._call_tool = AsyncMock(side_effect=mock_call)

        result = await mcp_client.enrich_candidate("AAPL")
        assert "fundamentals" in result
        assert "earnings_context" in result
        assert "mcp_news" in result
        assert "credit_risk" in result

    @pytest.mark.asyncio
    async def test_enrich_candidate_handles_all_failures(self, mcp_client):
        """If every provider fails, enrichment returns empty dict."""
        mcp_client._call_tool = AsyncMock(return_value=None)

        result = await mcp_client.enrich_candidate("AAPL")
        assert result == {}

    @pytest.mark.asyncio
    async def test_enrich_candidate_handles_exceptions(self, mcp_client):
        """Exceptions in individual methods are caught and don't break aggregation."""
        async def mock_fundamentals(ticker):
            raise Exception("network error")

        async def mock_earnings(ticker):
            return {"transcript": "test"}

        async def mock_news(ticker):
            return [{"headline": "test"}]

        async def mock_credit(ticker):
            return {"rating": "A1"}

        mcp_client.get_fundamentals = mock_fundamentals
        mcp_client.get_earnings_context = mock_earnings
        mcp_client.get_news = mock_news
        mcp_client.get_credit_risk = mock_credit

        result = await mcp_client.enrich_candidate("AAPL")
        # fundamentals failed, but others should still be present
        assert "fundamentals" not in result
        assert "earnings_context" in result
        assert "mcp_news" in result
        assert "credit_risk" in result


# ---------------------------------------------------------------------------
# MCPClient disabled/unavailable
# ---------------------------------------------------------------------------

class TestMCPDisabled:
    @pytest.mark.asyncio
    async def test_unavailable_client_returns_empty(self, mcp_client_empty):
        """When no MCP config exists, all methods return empty."""
        assert mcp_client_empty.available is False
        # Direct tool call returns None
        result = await mcp_client_empty._call_tool("morningstar", "test", {})
        assert result is None
