"""SQLite-based response cache with TTL for data layer deduplication.

Sits in front of the DataAggregator fallback chains to avoid redundant API calls.
Uses WAL journal mode for concurrent read performance with the async aggregator.
On Heroku the SQLite file is ephemeral (filesystem resets on deploy) — fine for a cache.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sqlite3
import time
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data_cache.sqlite"

# ---------------------------------------------------------------------------
# TTL constants (seconds)
# ---------------------------------------------------------------------------
TTL_HISTORICAL_OHLCV = 30 * 24 * 3600   # 30 days — data >3 days old is immutable
TTL_RECENT_OHLCV = 1 * 3600              # 1 hour — live/recent price data
TTL_FUNDAMENTALS = 4 * 3600              # 4 hours
TTL_NEWS = 30 * 60                       # 30 minutes
TTL_UNIVERSE = 12 * 3600                 # 12 hours
TTL_MACRO = 1 * 3600                     # 1 hour
TTL_EARNINGS_CALENDAR = 6 * 3600         # 6 hours


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def df_to_json(df: pd.DataFrame) -> str:
    """Serialize a DataFrame to JSON, handling date columns."""
    data = df.copy()
    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = data[col].astype(str)
        elif data[col].apply(lambda x: isinstance(x, date)).any():
            data[col] = data[col].astype(str)
    return data.to_json(orient="split", date_format="iso")


def json_to_df(json_str: str) -> pd.DataFrame:
    """Deserialize a DataFrame from JSON, restoring date columns."""
    df = pd.read_json(io.StringIO(json_str), orient="split")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def classify_ohlcv_ttl(to_date: date | None) -> int:
    """Return appropriate TTL for OHLCV data based on how recent the end date is."""
    if to_date is None:
        return TTL_RECENT_OHLCV
    days_ago = (date.today() - to_date).days
    if days_ago > 3:
        return TTL_HISTORICAL_OHLCV
    return TTL_RECENT_OHLCV


class DataCache:
    """SQLite response cache with TTL expiry and hit/miss tracking."""

    def __init__(self, db_path: Path | str | None = None):
        self._db_path = str(db_path or DEFAULT_DB_PATH)
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            timeout=5,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_table()

        # Stats tracking
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                key        TEXT PRIMARY KEY,
                data_json  TEXT    NOT NULL,
                source     TEXT,
                ticker     TEXT,
                endpoint   TEXT,
                ttl_seconds INTEGER NOT NULL,
                created_at REAL    NOT NULL,
                expires_at REAL    NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_expires
            ON cache_entries(expires_at)
        """)
        self._conn.commit()

    @staticmethod
    def build_key(source: str, ticker: str, endpoint: str, **params) -> str:
        """Build a deterministic cache key from source, ticker, endpoint, and params."""
        parts = {
            "source": source,
            "ticker": ticker,
            "endpoint": endpoint,
            **{k: str(v) for k, v in sorted(params.items())},
        }
        raw = json.dumps(parts, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        """Return cached JSON if within TTL, else None (deletes expired)."""
        now = time.time()
        row = self._conn.execute(
            "SELECT data_json, expires_at FROM cache_entries WHERE key = ?",
            (key,),
        ).fetchone()

        if row is None:
            self._misses += 1
            return None

        data_json, expires_at = row
        if now > expires_at:
            self._conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            self._conn.commit()
            self._misses += 1
            self._evictions += 1
            return None

        self._hits += 1
        return data_json

    def put(
        self,
        key: str,
        data_json: str,
        ttl_seconds: int,
        source: str = "",
        ticker: str = "",
        endpoint: str = "",
    ) -> None:
        """Store JSON string with metadata and TTL."""
        now = time.time()
        self._conn.execute(
            """INSERT OR REPLACE INTO cache_entries
               (key, data_json, source, ticker, endpoint, ttl_seconds, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (key, data_json, source, ticker, endpoint, ttl_seconds, now, now + ttl_seconds),
        )
        self._conn.commit()
        self._stores += 1

    def get_stats(self) -> dict:
        """Return cache performance statistics."""
        total_requests = self._hits + self._misses
        row = self._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
        total_entries = row[0] if row else 0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "stores": self._stores,
            "evictions": self._evictions,
            "hit_rate": round(self._hits / total_requests, 4) if total_requests > 0 else 0.0,
            "total_entries": total_entries,
        }

    def clear_expired(self) -> int:
        """Bulk eviction of stale entries. Returns number of rows deleted."""
        now = time.time()
        cursor = self._conn.execute(
            "DELETE FROM cache_entries WHERE expires_at < ?", (now,)
        )
        self._conn.commit()
        deleted = cursor.rowcount
        self._evictions += deleted
        return deleted

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
