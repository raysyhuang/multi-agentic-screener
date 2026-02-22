import sys
import types
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import OperationalError

from src.engines import gemini_runner


def _op_error(msg: str) -> OperationalError:
    return OperationalError("SELECT 1", {}, Exception(msg))


@pytest.fixture(autouse=True)
def _reset_db_initialized():
    gemini_runner._db_initialized = False
    yield
    gemini_runner._db_initialized = False


def _install_fake_app_database(monkeypatch, side_effects):
    app_mod = types.ModuleType("app")
    app_mod.__path__ = []
    db_mod = types.ModuleType("app.database")
    calls = {"count": 0}

    def _init_db():
        idx = calls["count"]
        calls["count"] += 1
        effect = side_effects[min(idx, len(side_effects) - 1)]
        if isinstance(effect, Exception):
            raise effect

    db_mod.init_db = _init_db
    monkeypatch.setitem(sys.modules, "app", app_mod)
    monkeypatch.setitem(sys.modules, "app.database", db_mod)
    return calls


def test_ensure_gemini_db_retries_then_succeeds(monkeypatch):
    calls = _install_fake_app_database(
        monkeypatch,
        [_op_error("tmp-1"), _op_error("tmp-2"), None],
    )
    sleeps: list[int] = []
    monkeypatch.setattr(gemini_runner.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(gemini_runner, "get_settings", lambda: SimpleNamespace(database_url=""))

    gemini_runner._ensure_gemini_db()

    assert calls["count"] == 3
    assert gemini_runner._db_initialized is True
    assert sleeps == [2, 4]


def test_ensure_gemini_db_raises_after_retry_exhaustion(monkeypatch):
    calls = _install_fake_app_database(
        monkeypatch,
        [_op_error("tmp-1"), _op_error("tmp-2"), _op_error("tmp-3")],
    )
    sleeps: list[int] = []
    monkeypatch.setattr(gemini_runner.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(gemini_runner, "get_settings", lambda: SimpleNamespace(database_url=""))

    with pytest.raises(OperationalError):
        gemini_runner._ensure_gemini_db()

    assert calls["count"] == 3
    assert gemini_runner._db_initialized is False
    assert sleeps == [2, 4]

