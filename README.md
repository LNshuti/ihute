# Nashville Transportation Incentive Simulation

An agent-based simulation framework for evaluating incentive mechanisms to reduce urban congestion. 

##  About

This project builds algorithms for incentive-based congestion mitigation, treating traffic participants as strategic **agents whose behavior can be influenced through carefully designed reward mechanisms**.

### Features

- **Simulation Algorithm Design**: Event-driven simulation with spatial indexing for large-scale corridor simulations (10,000+ agents)
- **Incentive Optimization**: Approximation algorithms for optimal reward allocation under budget constraints
<<<<<<< HEAD
- **Behavioral Model Learning**: ML techniques to extract response functions from a growing dataset of 369,831+ historical rideshare trips
=======
- **Behavioral Model Learning**: ML techniques to extract response functions from 369,831 historical rideshare trips
- **Equilibrium Computation**: Algorithms for computing Nash/Stackelberg equilibria in incentive-mediated systems
- **Demographic Integration**: Agent profiles enriched with ZCTA-level demographics (income, poverty) from population-dyna platform for realistic behavioral heterogeneity

### Demographic-Aware Agent Modeling

Agents are enriched with ZCTA-level demographics from the **[population-dyna](https://github.com/LNshuti/population-dyna)** platform, enabling income-stratified analysis and realistic behavioral heterogeneity.

**Data Coverage:**
- **376 Tennessee ZCTAs** (Nashville and surrounding areas)
- Income range: $14k-$70k (avg $62k)
- Poverty rates: 0-100% (avg 14%)
- Even distribution across 5 income quintiles

**Behavioral Calibration:**

Demographics influence agent preferences through empirically-grounded relationships:

| Demographic | Impact | Formula/Relationship |
|-------------|--------|---------------------|
| **Household Income** | Value of Time (VOT) | 50% of hourly wage (DOT guidance) |
| **Income Quintile** | Incentive Sensitivity | Q1 (low) → 2.66x more responsive than Q5 (high) |
| **Income Quintile** | Cost Sensitivity | Q1 → 1.5x baseline, Q5 → 0.5x baseline |
| **Income** | Car Ownership | Logistic function: P(car) = 1/(1+exp(-(income-40k)/10k)) |

**Key Results:**
- **VOT Variation:** $13/hr (low income) → $16/hr (high income)
- **Incentive Response:** 2.66x higher for low-income agents
- **Spatial Heterogeneity:** Agents distributed across 376 ZCTAs with realistic income patterns

**Usage Example:**

```python
from src.agents.commuter import create_demographic_commuter_population

# Create agents with Nashville demographics
agents = create_demographic_commuter_population(
    n_agents=1000,
    home_region=((36.0, -87.0), (36.4, -86.5)),  # Nashville bbox
    work_region=((36.1, -86.9), (36.3, -86.6)),
    warehouse_path="warehouse.duckdb",
)

# Each agent has:
# - profile.home_zcta (e.g., "37203")
# - profile.household_income ($14k-$70k)
# - profile.income_quintile (1-5)
# - preferences.vot (calibrated to income)
# - preferences.beta_incentive (income-dependent)
```

**Research Questions Enabled:**
- "Do low-income neighborhoods benefit more from carpool incentives?"
- "Which ZCTAs have highest pacer participation potential?"
- "Are current incentives equitable across income groups?"
- "How does income affect mode choice and schedule flexibility?"
>>>>>>> f08c9f0 (add demog)

### Incentive Use Cases

| Use Case | Objective | Key Mechanism |
|----------|-----------|---------------|
| **Pacer Driving** | Reduce stop-and-go waves | Rewards for smooth speed profiles |
| **Carpooling** | Increase vehicle occupancy | Time/corridor-specific shared ride incentives |
| **Event Egress** | Flatten post-Titans game peaks | Departure delay & mode-shift rewards |
| **Transit Promotion** | Encourage mode shift | Geofenced peak-period transit incentives |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Simulation Controller                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Agents     │  │  Incentive   │  │    Road Network      │   │
│  │  (Strategic) │◄─┤   Engine     │◄─┤  (Spatial Index)     │   │
│  └──────┬───────┘  └──────────────┘  └──────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Event-Driven Simulation Engine              │   │
│  │  • Priority Queue Scheduling                             │   │
│  │  • Spatial Hashing for Proximity Queries                 │   │
│  │  • Incremental State Updates                             │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Optimization │  │     ML       │  │    Analytics &       │   │
│  │  Algorithms  │  │  Calibration │  │    Validation        │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
ihute/
├── src/
│   ├── agents/              # Agent models and behavior
│   │   ├── base.py          # Base agent class
│   │   ├── commuter.py      # Commuter agent with mode/route choice
│   │   ├── pacer.py         # Flow-stabilizing pacer driver
│   │   └── behavioral.py    # Behavioral response functions
│   ├── incentives/          # Incentive mechanism implementations
│   │   ├── base.py          # Base incentive class
│   │   ├── carpool.py       # Carpooling incentives
│   │   ├── pacer.py         # Pacer driving rewards
│   │   ├── temporal.py      # Departure time shift incentives
│   │   └── transit.py       # Transit promotion incentives
│   ├── simulation/          # Core simulation engine
│   │   ├── engine.py        # Event-driven simulation controller
│   │   ├── events.py        # Event types and scheduling
│   │   ├── network.py       # Road network with spatial indexing
│   │   └── metrics.py       # Performance measurement
│   ├── optimization/        # Incentive optimization algorithms
│   │   ├── greedy.py        # Greedy allocation with approximation bounds
│   │   ├── dynamic.py       # Dynamic programming approaches
│   │   ├── metaheuristic.py # GA/simulated annealing
│   │   └── online.py        # Online allocation algorithms
│   ├── ml/                  # Machine learning for calibration
│   │   ├── features.py      # Feature engineering from GPS data
│   │   ├── models.py        # Classification/regression models
│   │   └── validation.py    # Cross-validation and testing
│   ├── data/                # Data loading and processing
│   │   ├── hytch.py         # Hytch rideshare data loader
│   │   ├── demographics.py  # ZCTA demographics loader (population-dyna)
│   │   ├── network.py       # Road network data (OSM)
│   │   └── events.py        # Event schedules (Titans games)
│   └── utils/               # Utilities
│       ├── config.py        # Configuration management
│       ├── logging.py       # Structured logging
│       └── visualization.py # Plotting and dashboards
├── tests/                   # Unit and integration tests
├── notebooks/               # Jupyter notebooks for analysis
├── configs/                 # YAML configuration files
├── data/                    # Data directory (gitignored)
│   ├── raw/                 # Raw input data
│   ├── processed/           # Processed datasets
│   └── models/              # Trained ML models
├── docs/                    # Documentation
├── scripts/                 # CLI scripts
├── pyproject.toml           # Project configuration
└── README.md
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/LNshuti/ihute.git
cd ihute

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Command Line Interface

```bash
# Run pacer simulation on I-24
python -m scripts.run_simulation --config configs/pacer_i24.yaml --agents 100000

# Train behavioral model from Hytch data
python -m scripts.train_model --data data/raw/hytch_trips.parquet --output data/models/

# Optimize incentive allocation
python -m scripts.optimize --scenario titans_game --budget 500000 --algorithm greedy

# Run full experiment suite
python -m scripts.run_experiments --suite all --output results/
```

## Data Sources

### Hytch Rideshare Data (Primary)

| Metric | Value |
|--------|-------|
| Total trips | 369,831 |
| Date range | 2018-2024 |
| Avg participants/trip | 2.5 |
| Training set | 352,354 trips (2018-2019) |
| Test set | 17,477 trips (2022-2024) |

Features extracted:
- Trip distance, duration, time-of-day
- Origin/destination zones
- Carpool formation rates
- Incentive response elasticity

### Nashville Road Network

- OpenStreetMap extract for Davidson County
- Focus corridors: I-24, I-40, I-65
- Spatial indexing with R-tree for proximity queries

### Event Data

- Titans game schedules (Nissan Stadium)
- Concert and event calendar
- Historical traffic patterns from TDOT

### Demographics Data (population-dyna)

**Source:** [population-dyna platform](https://github.com/LNshuti/population-dyna)

| Metric | Value |
|--------|-------|
| Total ZCTAs | 376 (Tennessee) |
| Coverage | Nashville + surrounding areas |
| Income Range | $14k - $70k |
| Avg Poverty Rate | 14.1% |
| Data Vintage | 2020-2022 |

**Datasets Used:**
- `zcta_poverty.parquet` - ZCTA-level poverty rates (2011-2022)
- `county_unemployment.parquet` - County unemployment time series (1990-2024)
- Derived: Median household income estimates from poverty rates

**Integration:**
- Loaded into DuckDB via dbt pipeline
- 376 ZCTAs with income quintiles, poverty rates
- Used to calibrate agent VOT and behavioral parameters
- Enables income-stratified simulation analysis

## Methodology

### Agent Behavioral Models

Agents make decisions using bounded-rational utility maximization:

```
U(action) = β₀ + β₁·travel_time + β₂·cost + β₃·incentive + β₄·comfort + ε
```

Decision rules:
- **Softmax**: P(action) ∝ exp(U(action)/τ)
- **Epsilon-greedy**: Explore with probability ε
- **Best response**: Pure utility maximization

### Incentive Optimization

Budget-constrained allocation problem:

```
maximize    Σᵢ congestion_reduction(incentiveᵢ)
subject to  Σᵢ costᵢ ≤ B (budget)
            incentiveᵢ ≥ 0
```

Algorithms implemented:
| Algorithm | Approximation Ratio | Time Complexity |
|-----------|---------------------|-----------------|
| Greedy | 1 - 1/e ≈ 0.63 | O(n log n) |
| Dynamic Programming | Optimal (pseudo-poly) | O(nB) |
| Genetic Algorithm | Empirical | O(g·p·n) |
| Online (Secretary) | 1/e ≈ 0.37 | O(n) |

### Equilibrium Computation

For multi-agent strategic interactions:
- **Best Response Dynamics**: Iterate until convergence
- **Fictitious Play**: Learn from historical play
- **Potential Games**: Exploit structure for faster convergence

## Evaluation Metrics

### Computational Metrics
- Runtime complexity (wall-clock, big-O)
- Memory footprint and scalability
- Convergence rate to equilibrium
- Approximation quality

### Transportation Metrics
- Peak demand reduction (%)
- Travel time reliability (95th percentile)
- Vehicle-miles traveled (VMT) reduction
- Average vehicle occupancy
- Incentive efficiency ($/VMT reduced)

### Validation Metrics
- Prediction accuracy (AUC, RMSE)
- Behavioral model fit (χ², KS test)
- Simulation-to-reality gap

## Experiments

### Experiment 1: Pacer Participation Threshold

**Question**: What minimum pacer participation rate yields measurable congestion reduction?

```bash
python -m scripts.run_experiments --experiment pacer_threshold \
    --participation-rates 0.01 0.02 0.05 0.10 0.15 0.20 \
    --replications 3000
```

### Experiment 2: Carpool Incentive Elasticity

**Question**: Is it more effective to increase reward magnitude or targeting precision?

```bash
python -m scripts.run_experiments --experiment carpool_elasticity \
    --reward-levels 1.0 2.0 5.0 10.0 \
    --targeting-precision low medium high
```

<<<<<<< HEAD
=======
### Experiment 3: Event Egress Optimization

**Question**: Are small delays across many participants more effective than large delays for few?

```bash
python -m scripts.run_experiments --experiment event_egress \
    --delay-distribution uniform concentrated \
    --total-delay-budget 100000
```


>>>>>>> f08c9f0 (add demog)
## Development

### Building Documentation

```bash
cd docs/
make html
```

### Gradio Dashboard Deployment

**Local Development:**
```bash
# Run locally
cd app && python app.py

# Run with auto-reload for development
cd app && gradio app.py
```

**Live Demo:** https://huggingface.co/spaces/LeonceNsh/ihute

