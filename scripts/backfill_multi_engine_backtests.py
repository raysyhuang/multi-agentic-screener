#!/usr/bin/env python3
"""Backfill local multi-engine backtest JSON files into the DB-backed dashboard store."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


async def _run(backtest_dir: Path) -> int:
    from src.backtest.multi_engine.persistence import persist_multi_engine_backtest_report

    files = sorted(backtest_dir.glob("multi_engine_*.json"))
    if not files:
        print(f"No backtest files found in {backtest_dir}")
        return 1

    ok = 0
    for path in files:
        try:
            report = json.loads(path.read_text())
        except Exception as exc:
            print(f"SKIP {path.name}: {exc}")
            continue

        persisted = await persist_multi_engine_backtest_report(report, path.name)
        if persisted:
            ok += 1
            print(f"OK   {path.name}")
        else:
            print(f"FAIL {path.name} (DB unavailable or persistence error)")

    print(f"\nPersisted {ok}/{len(files)} backtest files")
    return 0 if ok > 0 else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default="backtest_results/multi_engine",
        help="Directory containing multi_engine_*.json files",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(Path(args.dir))))


if __name__ == "__main__":
    main()
