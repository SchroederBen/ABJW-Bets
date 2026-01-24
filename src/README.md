# Source Code

This directory contains the implementation of the ABJW Bets system.

## Planned Structure
- `data/` – Data ingestion and preprocessing scripts
- `features/` – Feature engineering logic
- `models/` – Model training, calibration, and evaluation
- `llm/` – OpenAI integration and contextual feature generation
- `profitability/` – Expected value and betting logic
- `utils/` – Shared helper functions

## Notes
- All code should assume PostgreSQL as the primary data store
- Training and inference use identical feature schemas
- LLM outputs are constrained to structured numeric formats