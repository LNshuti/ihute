"""Tests for SQL queries used by dashboard components."""

import pytest
import pandas as pd

from ..database import Database


@pytest.fixture
def db():
    """Create database with sample data."""
    database = Database(db_path=None)
    _ = database.connection  # Initialize
    return database


class TestTrafficQueries:
    """Test traffic flow component queries."""

    def test_speed_heatmap_query(self, db):
        """Test speed heatmap aggregation query."""
        result = db.query("""
            SELECT
                extract(hour from hour_bucket) as hour,
                extract(dow from hour_bucket) as day_of_week,
                avg(avg_speed_mph) as avg_speed
            FROM marts.fct_corridor_flows
            GROUP BY 1, 2
        """)

        assert 'hour' in result.columns
        assert 'day_of_week' in result.columns
        assert 'avg_speed' in result.columns
        # Hours should be 0-23
        assert result['hour'].min() >= 0
        assert result['hour'].max() <= 23

    def test_hourly_volume_query(self, db):
        """Test hourly volume aggregation."""
        result = db.query("""
            SELECT
                extract(hour from hour_bucket) as hour,
                sum(vehicle_count) as total_vehicles
            FROM marts.fct_corridor_flows
            GROUP BY 1
        """)

        assert 'hour' in result.columns
        assert 'total_vehicles' in result.columns
        assert all(result['total_vehicles'] >= 0)


class TestIncentiveQueries:
    """Test incentive analytics queries."""

    def test_funnel_query(self, db):
        """Test conversion funnel query."""
        result = db.query("""
            SELECT
                count(*) as total_offers,
                sum(case when was_accepted then 1 else 0 end) as accepts,
                sum(case when was_completed then 1 else 0 end) as completions
            FROM marts.fct_incentive_events
        """)

        assert result['total_offers'].iloc[0] >= result['accepts'].iloc[0]
        assert result['accepts'].iloc[0] >= result['completions'].iloc[0]

    def test_spend_by_type_query(self, db):
        """Test spending by type query."""
        result = db.query("""
            SELECT
                incentive_type,
                sum(actual_payout) as total_spend
            FROM marts.fct_incentive_events
            WHERE was_completed
            GROUP BY incentive_type
        """)

        assert 'incentive_type' in result.columns
        assert 'total_spend' in result.columns


class TestSimulationQueries:
    """Test simulation comparison queries."""

    def test_scenario_list_query(self, db):
        """Test scenario listing query."""
        result = db.query("""
            SELECT DISTINCT scenario_name
            FROM marts.fct_simulation_runs
        """)

        assert 'scenario_name' in result.columns
        assert len(result) > 0

    def test_scenario_metrics_query(self, db):
        """Test scenario metrics query."""
        result = db.query("""
            SELECT
                scenario_name,
                speed_improvement_pct,
                vmt_reduction_pct
            FROM marts.fct_simulation_runs
        """)

        assert 'speed_improvement_pct' in result.columns
        assert 'vmt_reduction_pct' in result.columns


class TestElasticityQueries:
    """Test behavioral model queries."""

    def test_elasticity_query(self, db):
        """Test elasticity curve query."""
        result = db.query("""
            SELECT
                incentive_bucket,
                carpool_rate,
                avg_incentive
            FROM marts.metrics_elasticity
        """)

        assert 'incentive_bucket' in result.columns
        assert 'carpool_rate' in result.columns
        # Carpool rate should be between 0 and 1
        assert all(result['carpool_rate'] >= 0)
        assert all(result['carpool_rate'] <= 1)
