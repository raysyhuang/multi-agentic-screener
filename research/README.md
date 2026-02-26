# Research Scripts

Offline research tools for factor analysis using Qlib. These are **not** part of the production pipeline.

## Setup

```bash
pip install -r research/requirements-research.txt
```

## Scripts

### 1. Convert OHLCV to Qlib format

Fetches 2 years of daily OHLCV for the MAS ticker universe via yfinance, then converts to Qlib binary format.

```bash
python research/scripts/convert_ohlcv_to_qlib.py
```

Output: `research/data/qlib_data/` (git-ignored)

### 2. Run Alpha158 IC analysis

Loads Qlib data, runs the Alpha158 feature handler, and computes per-factor IC.

```bash
python research/scripts/run_alpha158_analysis.py
```

Output: top 20 factors by IC printed to stdout + saved to `research/outputs/`.

## Notes

- These scripts require `qlib` and `lightgbm` which are NOT in production `requirements.txt`.
- Run locally on a weekend — not in CI, not on Heroku.
- The `data/` and `outputs/` directories are git-ignored.
