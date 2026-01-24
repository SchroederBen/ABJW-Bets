# ABJW Bets – Documentation

This directory contains documentation describing the design and architecture of the ABJW Bets system.

## System Overview
ABJW Bets is built around a unified data pipeline that combines historical statistics, real-time game data, sportsbook market inputs, and LLM-generated contextual features into a single feature schema stored in PostgreSQL.

## Data Sources
- Cleaning the Glass (historical training data)
- Basketball-Reference (day-of-game statistics)
- Sportsbook odds and market movement
- LLM-generated contextual signals (injury uncertainty, market bias)

## High-Level Pipeline
1. Historical data ingestion and cleaning
2. Feature engineering and standardization
3. Model training and validation
4. Day-of-game feature alignment
5. Probability estimation and calibration
6. Expected value and profitability analysis

## Design Philosophy
- Focus on probability quality and expected value
- Prefer interpretability and calibration over raw accuracy
- Treat LLMs as structured feature generators, not black-box predictors
- Avoid overfitting and ensure reproducibility