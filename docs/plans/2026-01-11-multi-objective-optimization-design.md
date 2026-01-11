# Multi-Objective Simulation Optimization Design

**Date**: 2026-01-11
**Status**: Approved

## Overview

Jupyter notebook implementing constrained multi-objective optimization for the IHUTE traffic simulation using PyTorch with MLX acceleration on Apple Silicon.

## Objectives (4)

1. **VMT Reduction** - Minimize total vehicle-miles-traveled
2. **Travel Time Savings** - Minimize aggregate commuter travel time
3. **Budget Efficiency** - Minimize cost per unit improvement ($/VMT)
4. **Equity** - Maximize fair distribution of benefits across zones

## Approach: Hybrid Neural Surrogate + Differentiable Pareto

### Decision Variables (12 dimensions)

| Variable | Range | Description |
|----------|-------|-------------|
| carpool_reward | $1-10 | Reward per passenger |
| pacer_reward | $0.05-0.30 | Reward per mile |
| peak_multiplier | 1.0-2.5 | Peak hour bonus |
| budget_split | 0-1 | Carpool vs pacer allocation |
| morning_peak_start | 6-8 | AM peak start hour |
| morning_peak_end | 8-10 | AM peak end hour |
| evening_peak_start | 16-18 | PM peak start hour |
| evening_peak_end | 18-20 | PM peak end hour |
| zone_weights[4] | 0-1 each | Priority for 4 corridor segments |

### Neural Surrogate Architecture

```
Input (12) → LayerNorm
    → Linear(12, 128) → SiLU → Dropout(0.1)
    → Linear(128, 256) → SiLU → Dropout(0.1)
    → Linear(256, 128) → SiLU
    → 4 separate heads:
        → VMT Reduction head (128 → 1)
        → Travel Time head (128 → 1)
        → Budget Efficiency head (128 → 1)
        → Equity head (128 → 1)
```

Training: ~5000 synthetic samples, multi-task loss with uncertainty weighting.

### Constraints

- **Budget**: total_budget ≤ $50,000/day
- **Equity floor**: min_zone_allocation ≥ 5% per zone
- **Box constraints**: All decision variables within bounds

### Optimization Method

**EPO (Exact Pareto Optimal)** with Augmented Lagrangian for constraints:

```
L(x, λ, μ) = f(x) + Σ λ_i * g_i(x) + (μ/2) * Σ max(0, g_i(x))²
```

50 starting points → 100 iterations each → filter dominated → ~20-30 Pareto solutions

## Notebook Structure

1. Setup & Configuration
2. Data Preparation (DuckDB + behavioral model sampling)
3. Neural Surrogate Model (PyTorch + MLX)
4. Constrained Pareto Optimization (EPO solver)
5. Pareto Front Visualization (Plotly)
6. Solution Selection & Validation

## Visualizations

- Parallel coordinates plot
- Pairwise trade-off matrix (4×4)
- 3D interactive Pareto front
- Solution comparison table

## Validation

- Surrogate R² > 0.85 per objective
- Hypervolume and spacing metrics for Pareto quality
- Back-validation against LogitModel/ProspectTheoryModel

## Dependencies

- torch>=2.0
- mlx>=0.4.0
- plotly>=5.0
- pandas, numpy, duckdb, scipy

## Outputs

- `results/pareto_front.csv`
- `results/surrogate_model.pt`
- `results/optimization_report.html`
