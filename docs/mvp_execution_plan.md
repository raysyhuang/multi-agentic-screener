# MVP Execution Plan

## Milestone 1: Deterministic Core

Deliverables:

- Config + environment loading.
- Typed stage contracts.
- Deterministic ingest/features/signal/regime pipeline.
- CLI entrypoint that runs end-to-end without LLMs.

Definition of done:

- One run writes full stage artifacts.
- Candidate list reproducible under same inputs.

## Milestone 2: Agent Overlay (Claude + OpenAI)

Deliverables:

- `ClaudeSignalInterpreter` integration.
- `OpenAIAdversarialValidator` integration.
- `RiskGatekeeper` rule set and final decision shaping.

Definition of done:

- Each candidate has thesis and counter-thesis.
- Pipeline can return `Top1To2` or explicit `NoTrade`.

## Milestone 3: Validation + Tracking

Deliverables:

- Validation gate implementation per validation contract.
- Run persistence for stage artifacts and final outcomes.
- Queryable history endpoints.

Definition of done:

- Validation card generated per run.
- Historical outcomes retrievable by date/regime/confidence.

## Milestone 4: Automation + Product Surface

Deliverables:

- Daily scheduler.
- Telegram summary alerts.
- Clean frontend page for daily picks, evidence, and recent history.

Definition of done:

- Unattended run completes and updates UI + Telegram.
- Failure states are visible in API/UI logs.

## MVP Operating Mode

- Initial mode is paper-trading recommendation only.
- Real trade usage should wait for a sufficient observation period.
- Keep Gemini provider interface present but disabled until API is restored.
