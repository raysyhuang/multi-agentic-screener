"""Temporary script to delete stale daily_runs row."""
import asyncio
from src.db.session import get_session, init_db
from sqlalchemy import text

async def cleanup():
    await init_db()
    async with get_session() as session:
        result = await session.execute(
            text("DELETE FROM daily_runs WHERE run_date = '2026-02-15'")
        )
        print(f"Deleted {result.rowcount} stale daily_runs rows")

asyncio.run(cleanup())
