---
title: Nashville Transportation Incentive Simulation
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Simulation dashboard for I-24 corridor incentive evaluation
tags:
  - transportation
  - simulation
  - incentives
  - nashville
  - traffic
  - demographics
  - agent-based-modeling
---

# Nashville Transportation Incentive Simulation Dashboard

Interactive dashboard for evaluating incentive mechanisms to reduce urban congestion on the I-24 corridor in Nashville, TN.

## Features

- **Traffic Flow Analysis**: Real-time and historical speed heatmaps, volume patterns, and congestion timelines
- **Incentive Analytics**: Conversion funnels, spending analysis, and cost-effectiveness metrics
- **Behavioral Calibration**: ML model performance, elasticity curves, and feature importance
- **Demographics Analysis** ✨ NEW: Income distribution, poverty rates, and behavioral impact across 376 Nashville ZCTAs
- **Simulation Comparison**: Scenario analysis comparing treatment effects against baseline
- **Live Metrics**: Key performance indicators with trend sparklines
- **Corridor Map**: Interactive map with traffic conditions and zone statistics

## Data Sources

- **Hytch Rideshare Trips**: 369,831 historical trips for behavioral calibration
- **LADDMS**: I-24 GPS trajectory data
- **Population-Dyna Demographics**: 376 Tennessee ZCTAs with income and poverty data
- **Simulation Outputs**: Agent-based model results

## Technology Stack

- **Frontend**: Gradio + Plotly
- **Database**: DuckDB (in-memory for demo)
- **Simulation**: Custom agent-based model with behavioral economics

## About

This dashboard is part of the IHUTE (Nashville Transportation Incentive Simulation) project, which combines event-driven simulation with behavioral economics to evaluate carpooling, pacer driving, and other incentive mechanisms for congestion mitigation.

Built with DuckDB, dbt, and Gradio.
