# Validation Contract

## Purpose

Prevent false confidence from leakage, unrealistic execution assumptions, and fragile parameter tuning.

## Hard Rules

1. `NoLookAhead`
   - Signals can only use data available at `asof_timestamp`.
   - Trade simulation uses next-bar (or later) execution assumptions.

2. `ExecutionRealism`
   - Slippage and costs are included in every validation report.
   - No same-bar fill assumptions for close-derived signals.

3. `LeakageDefense`
   - Reject payloads missing `asof_timestamp`.
   - Reject columns/fields tagged as future-known.
   - Reject datasets with publication timestamp after as-of.

4. `FragilityDisclosure`
   - Required sensitivity tests:
     - slippage +25% and +50% impact.
     - threshold +/- 10% impact.
   - Validation card reports whether thesis survives perturbations.

5. `NoSilentPass`
   - Any failed validation check blocks picks.
   - Pipeline emits `NoTrade` with explicit failed checks.

## Checks Per Run

- `timestamp_integrity_check`
- `next_bar_execution_check`
- `future_data_guard_check`
- `slippage_sensitivity_check`
- `threshold_sensitivity_check`
- `confidence_calibration_check`

## Validation Card Schema

- `status`: `pass` | `fail`
- `checks`: map of check name -> `pass` | `fail`
- `fragility_score`: `0..1` (lower is safer)
- `key_risks`: list of strings
- `notes`: human-readable diagnostics

## Release Gate

A run can publish picks only if:

- all hard checks pass,
- at least one candidate remains risk-approved,
- final decision card is complete and persisted.
