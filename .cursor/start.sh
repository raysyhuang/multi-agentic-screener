#!/usr/bin/env bash
# Cloud Agent start — per-boot service reconciliation. Idempotent.
#
# Brings PostgreSQL up, ensures the dev role/database exist, and applies the
# Alembic migration chain so the schema is present before the dashboard or any
# test runs. Returns once the database is ready.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Starting PostgreSQL 16"
sudo pg_ctlcluster 16 main start 2>/dev/null || true

echo "==> Waiting for PostgreSQL to accept connections"
for _ in $(seq 1 30); do
  if sudo -u postgres pg_isready -q; then
    break
  fi
  sleep 1
done
sudo -u postgres pg_isready -q

# Dev role + database (idempotent). CREATEDB is required because the migration
# reversibility test provisions its own throwaway probe database.
echo "==> Ensuring 'mas' role and database exist"
sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='mas'" | grep -q 1 \
  || sudo -u postgres psql -c "CREATE ROLE mas LOGIN PASSWORD 'mas' CREATEDB;"
sudo -u postgres psql -c "ALTER ROLE mas CREATEDB;" >/dev/null
sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='mas'" | grep -q 1 \
  || sudo -u postgres createdb -O mas mas

echo "==> Applying database migrations (alembic upgrade head)"
alembic upgrade head

echo "==> start.sh complete — database ready at postgresql://mas:mas@localhost:5432/mas"
