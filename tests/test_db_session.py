from __future__ import annotations

import src.db.session as db_session


def test_get_engine_no_ssl_for_local(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_async_engine(url: str, **kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_session_factory", None)
    monkeypatch.setattr(db_session, "_get_async_url", lambda: "postgresql+asyncpg://localhost/test")
    monkeypatch.setattr(db_session, "create_async_engine", fake_create_async_engine)

    db_session.get_engine()

    assert captured["kwargs"]["pool_pre_ping"] is True
    assert captured["kwargs"]["connect_args"] == {}


def test_get_engine_forces_ssl_for_heroku_postgres(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_async_engine(url: str, **kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_session_factory", None)
    monkeypatch.setattr(
        db_session,
        "_get_async_url",
        lambda: "postgresql+asyncpg://u:p@c55vaqijj0vpoi.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com/d",
    )
    monkeypatch.setattr(db_session, "create_async_engine", fake_create_async_engine)

    db_session.get_engine()

    assert captured["kwargs"]["connect_args"] == {"ssl": "require"}
