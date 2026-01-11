# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IHUTE (Nashville Transportation Incentive Simulation) is an agent-based simulation framework for evaluating incentive mechanisms to reduce urban congestion on the I-24 corridor. It combines event-driven simulation with behavioral economics, trained on 369,831 historical Hytch rideshare trips.

## Commands

### Testing
```bash
pytest tests/ -v                      # Run all tests
pytest tests/ --cov=src               # With coverage
pytest tests/test_agents_base.py -v   # Single test module
pytest tests/ -k "test_carpool"       # Tests matching pattern
```

### dbt (Data Transformations)
```bash
cd dbt && dbt build                   # Build all models and run tests
cd dbt && dbt test                    # Run schema and data tests only
cd dbt && dbt run --select staging    # Run specific layer
```

### Dashboard
```bash
cd app && python app.py               # Run Gradio dashboard locally
```

## Architecture

### Three-Layer Design
1. **Simulation Engine** (`src/`) - Event-driven agent-based simulation with spatial indexing
2. **Data Pipeline** (`dbt/`) - dbt Core + DuckDB transformations (staging → intermediate → marts)
3. **Dashboard** (`app/`) - Gradio + Plotly interactive visualization

### Core Simulation Modules (`src/`)
- `agents/` - Agent types: commuter, pacer, behavioral models. Travel modes (drive, carpool, transit, etc.) and decision rules (utility max, softmax, satisficing)
- `incentives/` - Mechanism implementations: carpool rewards, pacer driving, temporal shifts, transit promotion
- `simulation/` - Event-driven engine with priority queue scheduling and R-tree spatial indexing
- `optimization/` - Budget-constrained allocation: greedy (0.63 approx), DP (optimal), genetic, online secretary

### Data Pipeline (`dbt/models/`)
- `staging/` - Raw data cleaning (LADDMS trajectories, Hytch trips, simulation outputs)
- `intermediate/` - Feature engineering and business logic joins
- `marts/` - Analytics-ready dimensions and facts (traffic, incentives, behavioral, simulation)

### Dashboard Tabs (`app/components/`)
- Traffic flow heatmaps, incentive analytics funnels, behavioral calibration curves, scenario comparison, live KPIs, I-24 corridor map

## Key Configuration

Main config: `incentives.yml` - Controls simulation parameters, agent populations, incentive mechanisms, and optimization settings.

Database: `warehouse.duckdb` - File-based analytical database, no external dependencies.

## Data Sources

- **Hytch trips** (`data/raw/hytch_trips.parquet`) - 369K rideshare trips for behavioral calibration
- **LADDMS** - I-24 GPS trajectory data
- **OSM Network** - Nashville road network with corridor focus

## Behavioral Model

Agents use calibrated utility functions with softmax choice probabilities. Key parameters:
- Temperature (τ): Controls randomness in mode/route choice
- Value of time, schedule flexibility, incentive sensitivity derived from Hytch data