# Nashville Transportation Incentive Simulation

An agent-based simulation framework for evaluating incentive mechanisms to reduce urban congestion. 

##  About

This project builds algorithms for incentive-based congestion mitigation, treating traffic participants as strategic agents whose behavior can be influenced through carefully designed reward mechanisms.

### Features

- **Simulation Algorithm Design**: Event-driven agent-based simulation with spatial indexing for large-scale corridor simulations (10,000+ agents)
- **Incentive Optimization**: Approximation algorithms for optimal reward allocation under budget constraints
- **Behavioral Model Learning**: ML techniques to extract response functions from 369,831 historical rideshare trips
- **Equilibrium Computation**: Algorithms for computing Nash/Stackelberg equilibria in incentive-mediated systems

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
git clone https://github.com/yourusername/ihute.git
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
python -m scripts.run_simulation --config configs/pacer_i24.yaml --agents 10000

# Train behavioral model from Hytch data
python -m scripts.train_model --data data/raw/hytch_trips.parquet --output data/models/

# Optimize incentive allocation
python -m scripts.optimize --scenario titans_game --budget 50000 --algorithm greedy

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
    --replications 30
```

### Experiment 2: Carpool Incentive Elasticity

**Question**: Is it more effective to increase reward magnitude or targeting precision?

```bash
python -m scripts.run_experiments --experiment carpool_elasticity \
    --reward-levels 1.0 2.0 5.0 10.0 \
    --targeting-precision low medium high
```

### Experiment 3: Event Egress Optimization

**Question**: Are small delays across many participants more effective than large delays for few?

```bash
python -m scripts.run_experiments --experiment event_egress \
    --delay-distribution uniform concentrated \
    --total-delay-budget 1000
```

## Configuration

Example configuration file (`configs/i24_corridor.yaml`):

```yaml
simulation:
  name: "I-24 Carpooling Study"
  duration_hours: 4
  warmup_hours: 0.5
  time_step_seconds: 1.0
  random_seed: 42

network:
  source: "osm"
  bounds:
    north: 36.20
    south: 36.05
    east: -86.60
    west: -86.90
  corridors:
    - name: "I-24"
      osm_relation_id: 123456

agents:
  total: 10000
  types:
    commuter:
      fraction: 0.85
      behavioral_model: "softmax"
      temperature: 0.5
    pacer_eligible:
      fraction: 0.15
      compliance_rate: 0.7

incentives:
  - type: "carpool"
    enabled: true
    reward_per_passenger: 2.50
    peak_hours: [7, 8, 9, 17, 18, 19]
    peak_multiplier: 1.5
    budget_daily: 10000

  - type: "pacer"
    enabled: true
    reward_per_mile: 0.10
    smoothness_threshold: 0.8
    corridor_segments: ["I-24-seg-1", "I-24-seg-2"]

optimization:
  algorithm: "greedy"
  budget_constraint: 50000
  objective: "vmt_reduction"

output:
  directory: "results/i24_carpool"
  save_trajectories: false
  save_metrics_interval: 300
  plots: ["demand_profile", "speed_heatmap", "incentive_uptake"]
```

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test module
pytest tests/test_simulation.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

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

**Deploy to Hugging Face Spaces:**
1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select "Gradio" as the SDK
3. Clone your Space repository:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   ```
4. Copy the app files:
   ```bash
   cp -r app/* YOUR_SPACE_NAME/
   cp warehouse.duckdb YOUR_SPACE_NAME/  # Include database
   ```
5. Create `requirements.txt` in the Space:
   ```
   gradio>=4.0.0
   plotly>=5.0.0
   duckdb>=0.9.0
   pandas>=2.0.0
   ```
6. Push to deploy:
   ```bash
   cd YOUR_SPACE_NAME
   git add . && git commit -m "Deploy dashboard" && git push
   ```

**Live Demo:** https://huggingface.co/spaces/LeonceNsh/ihute

