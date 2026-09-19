# H3 — drift after dividend news and forward splits: REJECTED at G1 (2026-09-19)

**Verdict: both parts REJECTED.**

| Primary (20 sessions, moving-block CI decisional) | n | Adjusted-price screen | **Raw-price screen** |
|---|---|---|---|
| **H3a** dividend initiations + ≥25% increases, pooled | 692 / 693 | +0.56%, block [−0.73, +1.81] | **+0.21%**, block [−1.08, +1.46] |
| **H3b** forward stock splits | 132 / 131 | −0.46%, block [−2.75, +1.85] | −0.11%, block [−2.53, +2.34] |

Under the registered screen, H3a narrowly clears condition (a): +0.56% against a +0.50% line. It fails condition (b). Once the price screen stops admitting future reverse-splitters (§ 3), H3a falls to +0.21%. H3b is negative, or close to zero, under both screens.

## Pre-registration

The criteria are in the docstring of `scripts/h3_dividends_splits.py`. It was committed in `c0a10a0` and pushed at 2026-09-19T09:16:42Z, before any dividend or split data had been fetched. Three pre-data amendments followed; each was pushed before any fetch succeeded, and none changed a definition or a G1 criterion:

1. **Never follow Polygon's `next_url`.** The cursor drops the date filter, so a second page returned rows with no declaration date.
2. **Key prior-dividend history on declaration date or ex-date.** Some rows have no declaration date. Missing an earlier dividend would turn a regular payer into a false "initiation".
3. **Fetch dividends per universe ticker, full history, one page each.** Whole-market ex-date days are flooded by mutual-fund share classes; one day alone had more than 1,000 F-tickers.

The data was fetched at 09:57:04Z (`data/cache/corp_actions/manifest.json`). It covers 4,721 tickers with no failures: 124,375 dividends and 6,524 splits. The first study run was at 09:57:10Z. After the review fixes it was re-run; the final JSONs were generated at 2026-09-19T11:25:35Z (adjusted screen) and 11:25:55Z (raw screen).

- **H3a.** A cash dividend (`CD`, USD) counts as an event if it is either:
  - an initiation: no CD in the previous 400 days; or
  - an increase: at least 1.25× the previous CD of the same frequency.

  Entry is the second session on or after the declaration date. The declaration time is unknown, and this choice is look-ahead-free for any time.
- **H3b.** Forward splits. Entry is the first session on or after the execution date, which is known in advance. Drift after the announcement cannot be tested, because the data has no announcement date.
- **G1.** Three conditions must all hold:
  - (a) excess_20 ≥ +0.50%;
  - (b) the moving-block CI lower bound is > 0;
  - (c) the mean is positive in at least 2 years with n ≥ 30.

## Descriptive cells

These are variants. None of them can change the verdict.

| Cell | Adjusted screen | Raw screen |
|---|---|---|
| Initiations, 20 sessions | +0.63% (n=197), block [−0.81, +2.14] | +0.21% (n=198), block [−1.24, +1.72] |
| Increases, 20 sessions | +0.54% (n=495), block [−0.95, +1.99] | +0.21% (n=495), block [−1.27, +1.65] |
| H3a, 60 sessions | +1.48%, block [−0.71, +3.68] | +0.73%, block [−1.42, +2.85] |
| Forward splits, 60 sessions | +3.69%, block [−1.54, +11.24] | +4.27%, block [−1.62, +13.06] |
| Reverse splits, 20 sessions | **−16.4%** (n=227), median −27% | −10.9% (n=16) |

## 3. What the reverse splits exposed: a look-ahead in the universe price screen

The reverse-split cell looked like an artifact at first, so it was checked. It is real: MULN, HUBC, AREB and SMX are death-spiral microcaps with serial reverse splits, and their prices collapsed within weeks.

The check also found a defect that affects every study in this series. The **$5 price floor was applied to split-adjusted prices.** A stock trading at $0.50 that later does a 1-for-100 reverse split shows up as $50 in its own adjusted history. The screen therefore admitted it using information about a split that had not yet happened.

`src/research/event_study.split_price_multiplier` now rebuilds the raw traded price from the split history. `build_panel(..., splits=...)` applies the floor to that raw price. Dollar volume needs no correction, because the price and volume adjustments cancel.

Scale of the problem: **2.1% of eligible name-days across 436 tickers** were sub-$5 penny stocks that later reverse-split. Only 16 of the 227 reverse-split events survive the raw screen. The rest were never tradable at $5 or more.

These names collapse, and they sat inside the **base rates**. That pulled the same-day base rates down, which pushed most events' excess returns up. This is why most positive numbers in H1, H3 and H5 shrink under the raw screen. The correction is not uniformly downward: forward splits at 60 sessions rise from +3.69% to +4.27%. `h1_pead_wide_FINDINGS.md` § 7 shows the effect on PEAD.

## Amendment after the Codex review of PR #123

The definitions are unchanged; the implementation was corrected. Every number above comes from the corrected code.

- **Share basis.** Payments are now compared on a single share basis. Each cash amount is divided by the factor of any splits executed after its ex-date. Before this fix, NVDA's post-split $0.01 dividend (a 150% raise NVIDIA itself announced) read as a cut. In the other direction, CIM's $0.11 → $0.35 across a 1:3 reverse split read as a +218% increase, when it was really +6%.
- **Row order.** Rows sharing a ticker, a declaration or ex-date and a frequency are now one payment, with their components summed. Ties sort deterministically on date, ex-date, amount and frequency. Previously, reversing the input list changed around 100 classifications.
- **Split rows** for the share basis go through the same `event_study.clean_splits` as the price screen. Identical duplicates are applied once, and conflicting same-day rows are dropped. None of these occur in this universe, so the results are unchanged.

## Caveats

- **Dividend definitions.** 400-day lookback; history starts 2022-01-01, and events whose lookback that start does not cover are skipped. An "increase" is measured only against the previous dividend of the same frequency.
- **Split timing.** Split execution dates are known in advance. The announcement, which is where the literature finds its drift, is not in the data.
- **Scope.** Stage 0 only: no stops, no costs.

## Provenance

- **Prices.** `ohlcv_polygon_wide_3y.parquet`, sha256 `2d04af2d…`.
- **Corporate actions.** `data/cache/corp_actions/`:
  - `dividends.json`, sha256 `b4f83edc…`
  - `splits.json`, sha256 `3319d67b…`

## Reproduce

```bash
python scripts/h3_dividends_splits.py --stage fetch
python scripts/h3_dividends_splits.py --stage study --json-out outputs/research/h3_dividends_splits.json
python scripts/h3_dividends_splits.py --stage study --raw-price-screen --json-out outputs/research/h3_dividends_splits_rawscreen.json
```
