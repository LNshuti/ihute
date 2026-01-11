# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IHUTE (Nashville Transportation Incentive Simulation) is an agent-based simulation framework for evaluating incentive mechanisms to reduce urban congestion on the I-24 corridor. It combines event-driven simulation with behavioral economics, trained on 369,831 historical Hytch rideshare trips.

## Commands

### Testing
```bash
pytest tests/ -v                      # Run all tests (229 tests)
pytest tests/ --cov=src               # With coverage (requires pytest-cov)
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

### Gradio Deployment
```bash
# Local development with auto-reload
cd app && gradio app.py

# Deploy to Hugging Face Spaces
# 1. Create a new Space at huggingface.co/new-space (select Gradio SDK)
# 2. Push the app/ directory contents to the Space repo
# 3. Ensure requirements.txt includes: gradio, plotly, duckdb, pandas

# Required files for HF Spaces deployment:
# - app.py (main application)
# - requirements.txt (dependencies)
# - warehouse.duckdb (database file, or configure remote connection)
```

## Architecture

### Three-Layer Design
1. **Simulation Engine** (`src/`) - Event-driven agent-based simulation with spatial indexing
2. **Data Pipeline** (`dbt/`) - dbt Core + DuckDB transformations (staging → intermediate → marts)
3. **Dashboard** (`app/`) - Gradio + Plotly interactive visualization

## Source Code (`src/`)

### Agents Module (`src/agents/`)

#### Base Classes (`base.py`)
- **TravelMode** - Enum: DRIVE_ALONE, CARPOOL_DRIVER, CARPOOL_PASSENGER, TRANSIT, WALK, BIKE, RIDESHARE
- **DecisionRule** - Enum: UTILITY_MAX, SOFTMAX, EPSILON_GREEDY, SATISFICING
- **AgentState** - Dataclass tracking position, velocity, mode, route, incentives earned
- **AgentPreferences** - Dataclass with VOT, beta coefficients (time, cost, incentive, comfort, reliability), ASCs, decision parameters
- **TripAttributes** - Dataclass for mode, travel_time, cost, incentive, comfort, reliability
- **LinearUtilityModel** - Implements `U = ASC + β_time·time + β_cost·cost + β_incentive·incentive + β_comfort·comfort`
- **BaseAgent** - Abstract class with decide_mode(), decide_departure_time(), decide_route(), respond_to_incentive()
- **PopulationParameters** - Configuration for heterogeneous agent generation
- **generate_heterogeneous_preferences()** - Creates varied preferences from population distributions

#### Commuter Agent (`commuter.py`)
- **CommuterProfile** - Home/work locations, arrival time, flexibility, car/transit access, carpool eligibility
- **CommuterAgent** - Daily commuter with mode choice, departure timing, route selection, incentive response
- **create_commuter_population()** - Generates heterogeneous commuter populations

#### Pacer Agent (`pacer.py`)
- **PacerProfile** - Enrolled corridors, max speed reduction tolerance, driving skill
- **PacerPerformance** - Tracks trips, miles, earnings, smoothness scores
- **PacerAgent** - Flow stabilization driver with pacing sessions, speed tracking, performance metrics
- **create_pacer_population()** - Generates pacer driver populations with boosted incentive responsiveness

#### Behavioral Models (`behavioral.py`)
- **LogitModel** - Multinomial logit with scale parameter and ASCs
- **MixedLogitModel** - Random taste variation with normal/lognormal/triangular coefficient distributions
- **ProspectTheoryModel** - Reference-dependent choice with loss aversion (λ=2.25), diminishing sensitivity (α=0.88), probability weighting
- **RegretMinimizationModel** - Minimizes anticipated regret across attributes (Chorus 2010)
- **estimate_incentive_elasticity()** - Simulates participation rate changes across incentive levels

### Incentives Module (`src/incentives/`)

#### Base Classes (`base.py`)
- **IncentiveType** - Enum: CARPOOL, PACER, DEPARTURE_SHIFT, TRANSIT, CONGESTION_PRICING, PARKING
- **IncentiveConfig** - Budget constraints, active hours/days, corridor/zone IDs, reward parameters
- **IncentiveAllocation** - Tracks individual allocations with status, conditions, outcomes
- **IncentiveResult** - Success/failure with allocation and metrics
- **BaseIncentive** - Abstract class with check_eligibility(), compute_reward(), verify_completion(), offer/accept/complete workflow

#### Carpool Incentive (`carpool.py`)
- **CarpoolMatch** - Driver, passengers, origin/destination, departure time, status
- **CarpoolIncentive** - Rewards per passenger ($2.50 default), driver bonus, distance bonus, peak multipliers
- Match finding with spatial (haversine) and temporal compatibility
- Match completion with reward distribution

#### Pacer Incentive (`pacer.py`)
- **PacerSession** - Speed/position samples, target speed, smoothness tracking
- **PacerIncentive** - Reward per mile ($0.15), smoothness threshold (0.7), minimum distance (2 mi)
- Session management with real-time feedback
- Corridor-level performance aggregation

### Other Modules
- `simulation/` - Event-driven engine with priority queue scheduling and R-tree spatial indexing
- `optimization/` - Budget-constrained allocation: greedy (0.63 approx), DP (optimal), genetic, online secretary
- `ml/` - Machine learning models for behavioral calibration
- `data/` - Data loading and preprocessing
- `utils/` - Utility functions

## Data Pipeline (`dbt/models/`)

- `staging/` - Raw data cleaning (LADDMS trajectories, Hytch trips, simulation outputs)
- `intermediate/` - Feature engineering and business logic joins
- `marts/` - Analytics-ready dimensions and facts (traffic, incentives, behavioral, simulation)

## Dashboard (`app/`)

**Live Demo**: https://huggingface.co/spaces/LeonceNsh/nashville-incentive-simulation

- `app.py` - Main Gradio application
- `database.py` - DuckDB connection and query execution
- `components/` - UI components for traffic flow heatmaps, incentive analytics, behavioral calibration, scenario comparison

## Test Coverage (`tests/`)

229 tests covering:
- `test_agents_base.py` - Enums, dataclasses, LinearUtilityModel, BaseAgent (31 tests)
- `test_agents_commuter.py` - CommuterProfile, CommuterAgent, population generation (28 tests)
- `test_agents_pacer.py` - PacerProfile, PacerAgent, pacing sessions (30 tests)
- `test_agents_behavioral.py` - All 4 behavioral models, elasticity estimation (36 tests)
- `test_incentives_base.py` - IncentiveConfig, BaseIncentive workflow (33 tests)
- `test_incentives_carpool.py` - CarpoolMatch, eligibility, rewards, matching (26 tests)
- `test_incentives_pacer.py` - PacerSession, sessions, corridor performance (45 tests)

## Key Configuration

**Main config:** `incentives.yml` - Controls simulation parameters, agent populations (10K agents), incentive mechanisms, optimization settings, experiment sweeps.

**Database:** `warehouse.duckdb` - File-based analytical database, no external dependencies.

## Data Sources

- **Hytch trips** (`data/raw/hytch_trips.parquet`) - 369K rideshare trips for behavioral calibration
- **LADDMS** - I-24 GPS trajectory data
- **OSM Network** - Nashville road network with I-24 corridor focus

## Behavioral Model Details

Agents use calibrated utility functions with configurable choice rules:

| Decision Rule | Description |
|--------------|-------------|
| UTILITY_MAX | Pure utility maximization (deterministic) |
| SOFTMAX | Probabilistic choice ∝ exp(U/τ), temperature τ controls randomness |
| EPSILON_GREEDY | Random exploration with probability ε |
| SATISFICING | Accept first option above threshold |

Key parameters derived from Hytch data:
- Value of time (VOT): lognormal, mean $25/hr
- Incentive sensitivity (β_incentive): ~0.15
- Schedule flexibility: 30-90 minutes typical
