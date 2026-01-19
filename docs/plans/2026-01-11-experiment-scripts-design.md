# Experiment Scripts Design

## Overview

Create a `scripts/run_experiments.py` CLI tool that runs three types of transportation incentive experiments with Monte Carlo replications, outputting both raw data and aggregated statistics.

## Experiments

### 1. Pacer Threshold (`pacer_threshold`)
Tests how pacer driver participation rates affect traffic flow stability.

```bash
python -m scripts.run_experiments --experiment pacer_threshold \
    --participation-rates 0.01 0.02 0.05 0.10 0.15 0.20 \
    --replications 30
```

**Parameters:** participation_rate (fraction of drivers as pacers)

**Metrics:** speed_variance, avg_travel_time, throughput, total_pacer_cost, smoothness_score

### 2. Carpool Elasticity (`carpool_elasticity`)
Tests reward sensitivity and targeting effectiveness for carpooling.

```bash
python -m scripts.run_experiments --experiment carpool_elasticity \
    --reward-levels 1.0 2.0 5.0 10.0 \
    --targeting-precision low medium high \
    --replications 20
```

**Parameters:** reward_per_passenger ($), targeting_precision (low/medium/high)

**Metrics:** carpool_rate, vmt_reduction, cost_per_vmt_reduced, match_rate

### 3. Event Egress (`event_egress`)
Simulates stadium/concert departure with staggered incentives.

```bash
python -m scripts.run_experiments --experiment event_egress \
    --delay-distribution uniform concentrated \
    --total-delay-budget 1000 \
    --replications 25
```

**Parameters:** delay_distribution (uniform/concentrated), total_delay_budget (person-minutes)

**Metrics:** peak_demand, avg_wait_time, congestion_duration, incentive_cost

## Project Structure

```
scripts/
├── __init__.py
├── run_experiments.py          # Main CLI entry point
├── experiments/
│   ├── __init__.py
│   ├── base.py                 # BaseExperiment class
│   ├── pacer_threshold.py
│   ├── carpool_elasticity.py
│   └── event_egress.py

src/simulation/
├── __init__.py
├── engine.py                   # Event-driven simulation engine
├── events.py                   # Event types
├── network.py                  # Simple road network
└── metrics.py                  # Metrics collection

results/
├── pacer_threshold/
├── carpool_elasticity/
└── event_egress/
```

## Simulation Engine

Lightweight event-driven engine:

- `Event`: time, event_type, agent_id, data
- `SimulationEngine`: event queue, agents, metrics collector
- Event types: departure, arrival, mode_choice, pacing_update
- Travel times: free-flow + congestion factor (no full microsimulation)

## Experiment Framework

```python
@dataclass
class ExperimentResult:
    experiment_name: str
    parameters: dict
    replication: int
    metrics: dict
    raw_data: pd.DataFrame

class BaseExperiment(ABC):
    def get_parameter_grid(self, **cli_args) -> list[dict]
    def run_single(self, params: dict, replication: int) -> ExperimentResult
    def run_all(self, replications: int, parallel: bool = True) -> pd.DataFrame
    def generate_report(self, results: pd.DataFrame) -> None
```

## Output Structure

```
results/pacer_threshold/
├── raw/
│   ├── rate_0.01_rep_01.parquet
│   └── ...
├── summary.csv          # Aggregated stats with CI
└── report.json          # Metadata
```

## CLI Options

```
--experiment NAME        Required: pacer_threshold, carpool_elasticity, event_egress
--output-dir PATH        Override default results directory
--seed INT               Random seed (default: 42)
--parallel/--no-parallel Enable/disable parallel replications
--verbose                Show progress and logging
```
