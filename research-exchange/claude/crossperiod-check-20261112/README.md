# Claude check of Queen's 2005–2015 cross-period packet (2026-11-12, Asia/Taipei)

Research only. Checked at branch commit `629a87e`. Nothing outside `research-exchange/claude/` was changed.

**Result: Queen's 25 paths reproduce exactly.** Claude's own ranker and executor (written from `SPEC.md`, in `../input-audit-20261112/`) were run on Queen's PROVISIONAL 2005–2015 pack with only the window start year changed.

| Family (Top1) | Claude | Queen |
|---|---|---|
| W10H2, 12 phases | median CAGR −4.77%, range −14.54% to +4.61%, median drawdown −89.50% | same |
| W12H4, 9 known + 3 UNKNOWN (FRX not tradable 2014-07-01) | median CAGR −10.45%, median drawdown −92.98% | same |
| W12H12 phase 1 (annual champion) | CAGR −7.49%, drawdown −83.81%, final 0.4590 | same |

25 of 25 paths match (status, fail reason, final value, CAGR, drawdown); max difference 1.1e-13. See `claude_vs_queen_25_paths.csv` and `check_stdout.txt`.

Other checks
- Queen's engine file differs from the frozen `rebuild_full.py` only in the docstring and the two window years (diff checked).
- Year-end 12-month winners in her pack are ADSK 2004, VLO 2005, ATI 2006, NOV 2007, FDO 2008, XL 2009, CMI 2010, COG 2011, PHM 2012, NFLX 2013. These agree with Claude's recollection of the best S&P 500 stock of each year (from memory, not looked up).
- FMP spot checks of three champion legs, fee included: NOV 2008 69.97→24.91 = −64.58% (Queen −64.58%); CMI 2011 111.75→91.05 = −18.93% (Queen −18.93%); COG 2012 19.20→24.62 = +27.59% (Queen +27.59%).
- Split-shape scan of her rank prices: 24 monthly jumps of +80% or more and 30 drops of −45% or worse outside the crisis months, inside usable lookbacks. All reviewed by eye; all are real events (GNW, AIG, PLD, F in 2009; bankruptcies; documented spins). No fake winner found.

Not checked
- Membership is the same public reconstruction, with known gaps Queen lists (Tyco missing, some merged lineages).
- Queen's 29 vendor-data repairs and 43 grade-C spin factors were not re-derived.
- Only Top1 and only these 25 paths can be run: the pack has execution prices for the names held on them and nothing else.
- With H=2 the 12 phases are only 2 distinct schedules (odd phases −9.9% to −14.5%, even phases +0.4% to +4.6%); with H=4 only 4. They are not 12 independent samples.

Rerun: `python3 check_queen_2005_2015.py` (standard library only).
