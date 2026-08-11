"""The screener request must actually ask FMP to omit funds.

Defence in depth for the universe gate: `filter_universe` drops ETFs from the
response, and this asks the API not to send them in the first place. The
parameters have to go out as the lowercase strings the API expects — Python's
`False` serializes to "False", which the endpoint would not match.

`filter_universe` does not depend on this working; if FMP ignores the
parameters, the client-side gate still catches them.
"""

from __future__ import annotations

import pytest

from src.data.fmp_client import FMPClient


class _FakeResponse:
    @staticmethod
    def json() -> list[dict]:
        return [{"symbol": "AAPL"}]


@pytest.mark.asyncio
async def test_screener_requests_no_etfs_or_funds(monkeypatch) -> None:
    captured: dict = {}

    async def fake_request(url, params):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse()

    client = FMPClient()
    monkeypatch.setattr(client, "_request", fake_request)

    await client.get_stock_screener()

    assert captured["params"]["isEtf"] == "false"
    assert captured["params"]["isFund"] == "false"


@pytest.mark.asyncio
async def test_flags_are_strings_not_python_bools(monkeypatch) -> None:
    """`False` would serialize to "False" and silently not match."""
    captured: dict = {}

    async def fake_request(url, params):
        captured.update(params)
        return _FakeResponse()

    client = FMPClient()
    monkeypatch.setattr(client, "_request", fake_request)

    await client.get_stock_screener()

    for flag in ("isEtf", "isFund"):
        assert isinstance(captured[flag], str), f"{flag} must be a string"
        assert captured[flag].islower()
