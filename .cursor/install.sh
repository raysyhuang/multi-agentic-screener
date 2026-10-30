#!/usr/bin/env bash
# Cloud Agent install — idempotent dependency refresh, run after checkout.
#
# Builds the full development toolchain from a base image: PostgreSQL 16
# (the app + tests use JSONB, so SQLite is not a substitute), the Python
# toolchain, and the project with its dev + research extras. Safe to re-run.
set -euo pipefail

# Resolve the repo root (parent of .cursor/) regardless of caller cwd.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Installing system packages (PostgreSQL, build tools, Python)"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  postgresql postgresql-contrib \
  build-essential python3.12-venv python3-dev libpq-dev \
  python-is-python3 python3-pip

# Install into the system interpreter (mirrors CI's plain `pip install -e`),
# so `python`/`pytest`/`alembic`/`uvicorn` resolve on PATH in every fresh shell
# without needing a venv to be activated.
echo "==> Installing Python dependencies (dev + research extras)"
sudo pip install --break-system-packages -e ".[dev,research]"

# CI lints against a pinned ruff (0.7.4); newer ruff releases flag the repo with
# rules it does not adopt. Pin locally so `ruff check .` matches CI exactly.
echo "==> Pinning ruff to the CI version"
sudo pip install --break-system-packages "ruff==0.7.4"

# Local dev .env (gitignored). The app and Alembic both read DATABASE_URL from
# here via pydantic-settings. Only created if absent, so local edits survive.
if [ ! -f .env ]; then
  echo "==> Writing local .env"
  cat > .env <<'EOF'
# Local Cloud Agent development environment
DATABASE_URL=postgresql://mas:mas@localhost:5432/mas
TRADING_MODE=PAPER
EXECUTION_MODE=quant_only
LOG_FORMAT=text
EOF
fi

# Expose DATABASE_URL to every shell (login/interactive/non-login) so the
# integration migration-probe tests, which read os.environ directly, work
# without a manual export — matching how CI provides it.
if ! grep -q '^DATABASE_URL=' /etc/environment 2>/dev/null; then
  echo "==> Registering DATABASE_URL in /etc/environment"
  echo 'DATABASE_URL=postgresql://mas:mas@localhost:5432/mas' | sudo tee -a /etc/environment >/dev/null
fi

echo "==> install.sh complete"
