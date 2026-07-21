"""PEAD live-wiring dry run — validates signal emission with real data, NO DB.

Exercises the exact live path the pipeline uses (earnings calendar -> EPS surprise
-> score_post_earnings_drift) against real FMP + Polygon data, and prints the PEAD
signals it WOULD emit. Needs only FMP_API_KEY + POLYGON_API_KEY — no Neon, no
writes — so the wiring can be verified before enabling pead_enabled on a paper box.

Usage:
  python scripts/pead_dry_run.py [--days 5] [--min-surprise 10] [--limit 0]
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import date, timedelta

from src.config import get_settings
from src.data.fmp_client import FMPClient
from src.data.polygon_client import PolygonClient
from src.features.technical import compute_all_technical_features, latest_features
from src.signals.post_earnings_drift import eps_surprise_pct, score_post_earnings_drift


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5, help="look back this many days for reporters")
    ap.add_argument("--min-surprise", type=float, default=None, help="override pead_min_surprise")
    ap.add_argument("--limit", type=int, default=0, help="cap reporters scored (0 = all beats)")
    args = ap.parse_args()

    s = get_settings()
    min_surprise = args.min_surprise if args.min_surprise is not None else s.pead_min_surprise
    today = date.today()
    frm = today - timedelta(days=args.days)

    fmp = FMPClient()
    poly = PolygonClient()

    print(f"PEAD dry run — reporters {frm}..{today}, min_surprise={min_surprise}% "
          f"(pead_enabled={s.pead_enabled}, this run does NOT persist)\n")

    cal = await fmp.get_earnings_calendar(frm, today)
    beats = []
    for row in cal or []:
        sym = str(row.get("symbol", "")).upper()
        sp = eps_surprise_pct(row.get("epsActual"), row.get("epsEstimated"))
        if sym and sp is not None and sp >= min_surprise:
            beats.append((sym, sp, row.get("date")))
    beats.sort(key=lambda x: x[1], reverse=True)
    if args.limit:
        beats = beats[: args.limit]
    print(f"{len(beats)} beats >= {min_surprise}% in window. Scoring against live OHLCV...\n")

    hdr = f"{'ticker':<8}{'surprise%':>10}{'entry':>10}{'stop':>10}{'target1':>10}{'score':>8}  report_date"
    print(hdr); print("-" * len(hdr))
    emitted = 0
    for sym, sp, rdate in beats:
        try:
            df = await poly.get_ohlcv(sym, today - timedelta(days=120), today)
        except Exception:
            continue
        if df is None or df.empty or len(df) < 60:
            continue
        enr = compute_all_technical_features(df)
        feat = latest_features(enr)
        feat["close"] = float(df["close"].iloc[-1])
        sig = score_post_earnings_drift(
            sym, df, feat, earnings_surprise_pct=sp,
            regime="unknown", min_surprise=min_surprise,
            stop_atr_mult=s.pead_stop_atr_mult, target_atr_mult=s.pead_target_atr_mult,
            holding_period=s.pead_holding_period,
        )
        if sig:
            emitted += 1
            print(f"{sig.ticker:<8}{sp:>10.1f}{sig.entry_price:>10.2f}{sig.stop_loss:>10.2f}"
                  f"{sig.target_1:>10.2f}{sig.score:>8.1f}  {rdate}")

    print(f"\nWould emit {emitted} PEAD signal(s). (Live pipeline caps by ranking + "
          f"max_final_picks; this dry run scores every beat.)")


if __name__ == "__main__":
    asyncio.run(main())
