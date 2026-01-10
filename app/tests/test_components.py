"""Tests for dashboard visualization components."""

import pytest
import pandas as pd
import plotly.graph_objects as go

from ..components import (
    # Traffic flow
    create_speed_heatmap,
    create_hourly_volume_chart,
    create_congestion_timeline,
    # Incentive analytics
    create_funnel_chart,
    create_spend_chart,
    create_effectiveness_chart,
    # Behavioral
    create_elasticity_curve,
    create_feature_importance_chart,
    create_model_metrics_display,
    # Simulation
    create_scenario_comparison_chart,
    create_cost_effectiveness_chart,
    # Metrics
    create_kpi_gauge,
    create_metric_card,
    create_sparkline,
    # Map
    create_corridor_map,
    create_zone_comparison,
)


class TestTrafficComponents:
    """Test traffic flow visualization components."""

    def test_speed_heatmap_empty_data(self):
        """Test heatmap handles empty data gracefully."""
        df = pd.DataFrame()
        fig = create_speed_heatmap(df)
        assert isinstance(fig, go.Figure)

    def test_speed_heatmap_with_data(self):
        """Test heatmap with valid data."""
        df = pd.DataFrame({
            'hour': list(range(24)),
            'day_of_week': [0] * 24,
            'avg_speed': [50] * 24
        })
        fig = create_speed_heatmap(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_hourly_volume_chart(self):
        """Test hourly volume chart creation."""
        df = pd.DataFrame({
            'hour': [7, 8, 9, 17, 18, 19],
            'time_period': ['AM_PEAK'] * 3 + ['PM_PEAK'] * 3,
            'total_vehicles': [100, 150, 120, 130, 160, 140],
            'avg_speed': [35, 30, 32, 28, 25, 30]
        })
        fig = create_hourly_volume_chart(df)
        assert isinstance(fig, go.Figure)

    def test_congestion_timeline(self):
        """Test congestion timeline chart."""
        df = pd.DataFrame({
            'hour_bucket': pd.date_range('2024-01-01', periods=24, freq='h'),
            'corridor_id': ['I-24'] * 24,
            'avg_speed_mph': [45 + i % 10 for i in range(24)],
            'level_of_service': ['B'] * 24
        })
        fig = create_congestion_timeline(df)
        assert isinstance(fig, go.Figure)


class TestIncentiveComponents:
    """Test incentive analytics components."""

    def test_funnel_chart(self):
        """Test funnel chart creation."""
        df = pd.DataFrame({
            'incentive_type': ['CARPOOL', 'PACER'],
            'total_offers': [100, 80],
            'accepts': [60, 50],
            'completions': [40, 35]
        })
        fig = create_funnel_chart(df)
        assert isinstance(fig, go.Figure)

    def test_spend_chart(self):
        """Test spending pie chart."""
        df = pd.DataFrame({
            'incentive_type': ['CARPOOL', 'PACER', 'TRANSIT'],
            'total_spend': [5000, 3000, 2000],
            'n_events': [200, 150, 100],
            'avg_payout': [25, 20, 20]
        })
        fig = create_spend_chart(df)
        assert isinstance(fig, go.Figure)

    def test_effectiveness_chart(self):
        """Test cost effectiveness bar chart."""
        df = pd.DataFrame({
            'incentive_type': ['CARPOOL', 'PACER'],
            'n_completed': [40, 35],
            'total_cost': [1000, 700],
            'avg_cost': [25, 20]
        })
        fig = create_effectiveness_chart(df)
        assert isinstance(fig, go.Figure)


class TestBehavioralComponents:
    """Test behavioral calibration components."""

    def test_elasticity_curve(self):
        """Test elasticity curve visualization."""
        df = pd.DataFrame({
            'incentive_bucket': ['NONE', 'LOW', 'MEDIUM', 'HIGH'],
            'n_trips': [100, 150, 120, 80],
            'carpool_rate': [0.1, 0.18, 0.28, 0.38],
            'avg_incentive': [0, 1.5, 3.5, 7.0]
        })
        fig = create_elasticity_curve(df)
        assert isinstance(fig, go.Figure)

    def test_feature_importance_chart(self):
        """Test feature importance bar chart."""
        df = pd.DataFrame({
            'feature': ['incentive', 'distance', 'time'],
            'importance': [0.4, 0.35, 0.25]
        })
        fig = create_feature_importance_chart(df)
        assert isinstance(fig, go.Figure)

    def test_model_metrics_display(self):
        """Test model metrics radar chart."""
        metrics = {'auc': 0.78, 'rmse': 0.15, 'accuracy': 0.82, 'n_samples': 1000}
        fig = create_model_metrics_display(metrics)
        assert isinstance(fig, go.Figure)


class TestSimulationComponents:
    """Test simulation comparison components."""

    def test_scenario_comparison_chart(self):
        """Test scenario comparison bar chart."""
        df = pd.DataFrame({
            'scenario_name': ['Carpool', 'Pacer', 'Transit'],
            'n_agents': [10000] * 3,
            'treatment_avg_speed': [48, 46, 44],
            'baseline_avg_speed': [42] * 3,
            'speed_improvement_pct': [14.3, 9.5, 4.8],
            'vmt_reduction_pct': [12, 8, 5],
            'peak_reduction_pct': [10, 6, 4],
            'treatment_spend': [5000, 4000, 3000]
        })
        fig = create_scenario_comparison_chart(df)
        assert isinstance(fig, go.Figure)

    def test_cost_effectiveness_chart(self):
        """Test cost vs impact scatter."""
        df = pd.DataFrame({
            'scenario_name': ['A', 'B'],
            'treatment_spend': [5000, 3000],
            'vmt_reduction_pct': [12, 8],
            'speed_improvement_pct': [10, 6]
        })
        fig = create_cost_effectiveness_chart(df)
        assert isinstance(fig, go.Figure)


class TestMetricsComponents:
    """Test real-time metrics components."""

    def test_kpi_gauge(self):
        """Test KPI gauge creation."""
        fig = create_kpi_gauge(75, 'Test Metric', '%', 0, 100)
        assert isinstance(fig, go.Figure)

    def test_metric_card(self):
        """Test metric card creation."""
        fig = create_metric_card(42.5, 'Test Value', prefix='$', format_str='.1f')
        assert isinstance(fig, go.Figure)

    def test_sparkline(self):
        """Test sparkline creation."""
        values = [10, 12, 15, 14, 18, 20]
        fig = create_sparkline(values, 'Trend')
        assert isinstance(fig, go.Figure)


class TestMapComponents:
    """Test geospatial components."""

    def test_corridor_map(self):
        """Test corridor map creation."""
        df = pd.DataFrame({
            'segment_id': ['1', '2'],
            'segment_name': ['Seg 1', 'Seg 2'],
            'latitude': [36.10, 36.12],
            'longitude': [-86.72, -86.75],
            'avg_speed_mph': [35, 45],
            'congestion_level': ['Moderate', 'Light'],
            'vehicle_count': [400, 300]
        })
        fig = create_corridor_map(df)
        assert isinstance(fig, go.Figure)

    def test_zone_comparison(self):
        """Test zone comparison chart."""
        df = pd.DataFrame({
            'zone': ['Downtown', 'Suburbs'],
            'avg_speed': [25, 45],
            'carpool_rate': [0.25, 0.15],
            'incentive_uptake': [0.35, 0.20]
        })
        fig = create_zone_comparison(df)
        assert isinstance(fig, go.Figure)
