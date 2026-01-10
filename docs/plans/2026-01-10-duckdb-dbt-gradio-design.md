# DuckDB + dbt Core + Gradio Data Platform Design

**Date:** 2026-01-10
**Status:** Approved
**Author:** Claude (with user collaboration)

## Overview

A scalable data platform combining DuckDB as the analytical database, dbt Core for data transformations, and Gradio for interactive visualization. Deployed to Hugging Face Spaces.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Data Sources                                     │
├─────────────────┬─────────────────┬─────────────────────────────────────┤
│  LADDMS (I-24)  │  Hytch Trips    │  Simulation Outputs                 │
│  - raw_JSON/    │  - 369K trips   │  - Agent decisions                  │
│  - PET/         │  - parquet      │  - Incentive allocations            │
│  - zones.pkl    │                 │  - Metrics timeseries               │
└────────┬────────┴────────┬────────┴──────────────┬──────────────────────┘
         │                 │                       │
         ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      DuckDB + dbt Core                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  staging/         │  intermediate/      │  marts/                       │
│  - stg_laddms_*   │  - int_trajectories │  - fct_trips                  │
│  - stg_hytch_*    │  - int_incentives   │  - fct_incentive_events       │
│  - stg_sim_*      │  - int_agents       │  - dim_corridors              │
│                   │                     │  - dim_agents                 │
│                   │                     │  - metrics_*                  │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Gradio Dashboard (HuggingFace)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Traffic  │ │Incentive │ │Behavioral│ │Simulation│ │   Map    │       │
│  │  Flow    │ │Analytics │ │  Calib   │ │ Compare  │ │  View    │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
nashville-incentive-sim/
├── dbt/                        # dbt project
│   ├── dbt_project.yml
│   ├── profiles.yml
│   ├── models/
│   │   ├── staging/
│   │   │   ├── laddms/
│   │   │   │   ├── _laddms__sources.yml
│   │   │   │   ├── stg_laddms__trajectories.sql
│   │   │   │   ├── stg_laddms__trajectory_counts.sql
│   │   │   │   ├── stg_laddms__zones.sql
│   │   │   │   └── stg_laddms__pet.sql
│   │   │   ├── hytch/
│   │   │   │   ├── _hytch__sources.yml
│   │   │   │   ├── stg_hytch__trips.sql
│   │   │   │   └── stg_hytch__participants.sql
│   │   │   └── simulation/
│   │   │       ├── _simulation__sources.yml
│   │   │       ├── stg_sim__agent_decisions.sql
│   │   │       ├── stg_sim__incentive_events.sql
│   │   │       └── stg_sim__metrics_timeseries.sql
│   │   ├── intermediate/
│   │   │   ├── int_trajectory_speeds.sql
│   │   │   ├── int_corridor_congestion.sql
│   │   │   ├── int_trip_features.sql
│   │   │   ├── int_incentive_outcomes.sql
│   │   │   └── int_simulation_scenarios.sql
│   │   └── marts/
│   │       ├── core/
│   │       │   ├── dim_corridors.sql
│   │       │   ├── dim_time_periods.sql
│   │       │   ├── dim_agents.sql
│   │       │   └── dim_incentive_types.sql
│   │       ├── traffic/
│   │       │   ├── fct_corridor_flows.sql
│   │       │   └── metrics_congestion.sql
│   │       ├── incentives/
│   │       │   ├── fct_incentive_events.sql
│   │       │   └── metrics_incentive_effectiveness.sql
│   │       ├── behavioral/
│   │       │   ├── fct_mode_choices.sql
│   │       │   └── metrics_elasticity.sql
│   │       └── simulation/
│   │           ├── fct_simulation_runs.sql
│   │           └── metrics_scenario_comparison.sql
│   ├── tests/
│   │   ├── generic/
│   │   │   ├── test_positive_values.sql
│   │   │   ├── test_valid_coordinates.sql
│   │   │   └── test_timestamp_order.sql
│   │   └── singular/
│   │       ├── assert_incentive_budget_not_exceeded.sql
│   │       ├── assert_trajectory_continuity.sql
│   │       └── assert_simulation_metrics_bounds.sql
│   ├── macros/
│   └── seeds/
├── app/                        # Gradio application
│   ├── app.py
│   ├── database.py
│   ├── components/
│   │   ├── traffic_flow.py
│   │   ├── incentive_analytics.py
│   │   ├── behavioral_calib.py
│   │   ├── simulation_compare.py
│   │   ├── realtime_metrics.py
│   │   └── geo_map.py
│   ├── queries/
│   ├── tests/
│   │   ├── test_database.py
│   │   ├── test_queries.py
│   │   ├── test_components.py
│   │   └── test_integration.py
│   └── requirements.txt
├── .github/workflows/
│   ├── test.yml
│   ├── dbt-build.yml
│   ├── deploy-hf.yml
│   └── publish.yml
└── warehouse.duckdb            # Built database file
```

## dbt Model Details

### Staging Layer
- Raw data cleaning, type casting, renaming
- One model per source table
- Minimal transformations

### Intermediate Layer
- Business logic, joins, calculations
- Reusable building blocks
- Not exposed to end users

### Marts Layer
- Analytics-ready tables
- Dimension and fact tables
- Metrics aggregations

## Gradio Dashboard Tabs

| Tab | Key Visuals | Data Source |
|-----|-------------|-------------|
| Traffic Flow | Speed heatmap, hourly volume, congestion timeline | `metrics_congestion`, `fct_corridor_flows` |
| Incentive Analytics | Uptake funnel, cost per VMT, budget burn | `fct_incentive_events`, `metrics_incentive_effectiveness` |
| Behavioral Calibration | Elasticity curve, model AUC/RMSE, feature importance | `metrics_elasticity`, `fct_mode_choices` |
| Simulation Comparison | Before/after charts, scenario dropdown | `fct_simulation_runs`, `metrics_scenario_comparison` |
| Live Metrics | KPI cards, trend sparklines | Aggregated from marts |
| Corridor Map | Interactive I-24 map with segment coloring | `dim_corridors`, `fct_corridor_flows` |

### Interactivity
- Date range picker (global filter)
- Scenario selector dropdown
- Corridor/segment filter
- Time-of-day slider (AM peak, PM peak, off-peak)

## Testing Strategy

### dbt Tests
- **Schema tests:** not_null, unique, accepted_range, relationships
- **Generic tests:** positive_values, valid_coordinates, timestamp_order
- **Singular tests:** budget constraints, data continuity, metric bounds
- **Freshness tests:** source data recency

### Gradio App Tests (pytest)
- `test_database.py` - Connection, query execution
- `test_queries.py` - Each SQL returns expected schema
- `test_components.py` - Each viz component renders
- `test_integration.py` - Full app smoke test

## GitHub Workflows

### test.yml
- Triggers: push, pull_request
- Matrix: Python 3.10, 3.11, 3.12, 3.13, 3.14
- Steps: Install deps, run pytest with coverage

### dbt-build.yml
- Triggers: push (changes to dbt/ or data/)
- Steps: Install dbt-duckdb, run dbt build + test

### deploy-hf.yml
- Triggers: push to main (app/, dbt/, warehouse.duckdb)
- Steps: Push to Hugging Face Space

### publish.yml
- Triggers: release created
- Steps: Test matrix, build, publish to PyPI

## Dependencies

```
dbt-duckdb>=1.7.0
duckdb>=0.10.0
gradio>=4.0.0
plotly>=5.18.0
folium>=0.15.0
pandas>=2.0.0
pyarrow>=14.0.0
pytest>=8.0.0
pytest-cov>=4.0.0
```

## Implementation Phases

1. **Phase 1: dbt Foundation** - Project setup, staging models, schema tests
2. **Phase 2: Intermediate & Marts** - Transformations, dimensions, facts
3. **Phase 3: Gradio App** - Database layer, 6 dashboard components
4. **Phase 4: Testing & CI/CD** - pytest suite, GitHub workflows
5. **Phase 5: Deploy** - Build DuckDB, push to Hugging Face

## Deployment

- DuckDB file committed to repo (file-based approach)
- Gradio app deployed to Hugging Face Spaces
- Auto-deploy on push to main branch
