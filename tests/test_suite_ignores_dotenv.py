"""The test suite must not inherit the developer's credentials.

`Settings` loads `PROJECT_ROOT/.env` (src/config.py), so on a machine holding
real credentials the suite sees populated API keys and a live DATABASE_URL,
while CI sees neither. Anything touching settings then passes locally and fails
on the runner.

That happened three times in a single session, each costing a CI round trip:
`validate_keys_for_mode()` in `src.worker`, the same call again in `main()`, and
an aggregator cache test falling through to the real SQLite file. The direction
is what makes it expensive — local is the permissive environment, so it can
never catch what CI will reject.

`pytest_configure` in conftest cuts the connection. These tests keep it cut.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings, get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_settings_do_not_read_the_dotenv_file() -> None:
    assert Settings.model_config.get("env_file") is None, (
        "the suite is loading .env again; tests will pass locally and fail in CI"
    )


def test_api_keys_are_empty_regardless_of_the_machine() -> None:
    """The concrete symptom: a populated .env made key validation succeed."""
    if os.environ.get("POLYGON_API_KEY"):
        # An explicitly exported variable is a deliberate choice and still wins;
        # only the implicit dotenv inheritance is severed.
        return

    settings = Settings()
    assert settings.polygon_api_key == ""
    assert settings.fmp_api_key == ""
    assert settings.telegram_bot_token == ""


def test_this_matters_only_because_the_dotenv_exists_locally() -> None:
    """Documents why the guard is not a no-op on a developer machine.

    On CI there is no `.env` and this test proves nothing; locally there is one,
    and it proves the suite is ignoring it. Recorded so a future reader does not
    delete the guard as pointless after seeing it pass on a bare runner.
    """
    dotenv = PROJECT_ROOT / ".env"
    if not dotenv.exists():
        return  # CI: nothing to be isolated from

    assert get_settings().polygon_api_key == "", (
        "a .env exists and its keys are reaching Settings — the isolation in "
        "conftest.pytest_configure is not working"
    )
