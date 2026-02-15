# Data Contracts

## Contract Goals

- Typed payloads between every stage.
- Strict stage boundaries to prevent hidden coupling.
- Stable schemas that are provider/model agnostic.

## Core Envelope

Every stage output must include:

- `run_id`: unique run identifier.
- `stage`: canonical stage name.
- `created_at`: ISO timestamp.
- `status`: `success` | `failed`.
- `payload`: stage-specific object.
- `errors`: list of structured errors.

## Stage Contracts

### 1) `DataIngestPayload`

- `asof_date`
- `universe`: list of ticker snapshots
  - `ticker`
  - `last_price`
  - `volume`
  - `avg_volume_20d`
  - `market_cap` (optional)
  - `source_provenance`

### 2) `FeaturePayload`

- `asof_date`
- `ticker_features`: list
  - `ticker`
  - `returns_5d`
  - `returns_10d`
  - `rsi_14`
  - `atr_pct`
  - `rvol_20d`
  - `distance_from_sma20`
  - `distance_from_sma50`
  - `feature_quality_flags`

### 3) `SignalPrefilterPayload`

- `asof_date`
- `candidates`: list
  - `ticker`
  - `model_scores` (`breakout`, `pullback`, `squeeze`, `catalyst`)
  - `aggregate_score`
  - `prefilter_flags`

### 4) `RegimePayload`

- `asof_date`
- `regime`
  - `label`: `trending_up` | `trending_down` | `choppy` | `high_volatility`
  - `confidence`
  - `signals_allowed`
- `gated_candidates`

### 5) `AgentReviewPayload`

- `ticker_reviews`: list
  - `ticker`
  - `signal_thesis`
  - `signal_confidence`
  - `counter_thesis`
  - `confidence_adjustment`
  - `risk_decision`: `approve` | `veto` | `resize`
  - `risk_notes`

### 6) `ValidationPayload`

- `leakage_checks`
  - `asof_timestamp_present`
  - `next_bar_execution_enforced`
  - `future_data_columns_found`
- `fragility_metrics`
  - `slippage_sensitivity`
  - `threshold_sensitivity`
  - `confidence_calibration_bucket`
- `validation_status`: `pass` | `fail`

### 7) `FinalOutputPayload`

- `decision`: `Top1To2` | `NoTrade`
- `picks`: list (0..2)
  - `ticker`
  - `entry_zone`
  - `stop_loss`
  - `targets`
  - `confidence`
  - `regime_context`
  - `validation_card`
- `no_trade_reason` (required if no picks)

## Versioning Rules

- Contract versions are semantic (`v1`, `v1.1`).
- Breaking schema changes require:
  - version bump,
  - migration note,
  - validator update.

## Validation Rules

- Pydantic models enforce all contracts at stage boundaries.
- Unknown fields are forbidden by default.
- Missing required fields fail the run and force `NoTrade`.
