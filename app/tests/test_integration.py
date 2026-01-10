"""Integration tests for the complete Gradio application."""

import pytest


class TestAppIntegration:
    """Test full application integration."""

    def test_app_creation(self):
        """Test that the Gradio app can be created."""
        from ..app import create_app

        app = create_app()
        assert app is not None

    def test_app_has_tabs(self):
        """Test that app has expected tabs."""
        from ..app import create_app

        app = create_app()
        # Gradio Blocks should have children
        assert hasattr(app, 'children') or hasattr(app, 'blocks')

    def test_database_available_to_app(self):
        """Test database is accessible from app context."""
        from ..database import get_database

        db = get_database()
        # Should be able to run a simple query
        result = db.query("SELECT 1 as test")
        assert len(result) == 1

    def test_all_component_imports(self):
        """Test all component modules import successfully."""
        from ..components import (
            create_speed_heatmap,
            create_funnel_chart,
            create_elasticity_curve,
            create_scenario_comparison_chart,
            create_kpi_gauge,
            create_corridor_map,
        )

        # All imports should succeed
        assert callable(create_speed_heatmap)
        assert callable(create_funnel_chart)
        assert callable(create_elasticity_curve)
        assert callable(create_scenario_comparison_chart)
        assert callable(create_kpi_gauge)
        assert callable(create_corridor_map)


class TestDataPipeline:
    """Test data pipeline integration."""

    def test_sample_data_populated(self):
        """Test that sample data tables are created."""
        from ..database import Database

        db = Database(db_path=None)
        _ = db.connection

        # Check key tables exist and have data
        tables_to_check = [
            'marts.fct_corridor_flows',
            'marts.fct_incentive_events',
            'marts.fct_simulation_runs',
            'marts.metrics_elasticity',
        ]

        for table in tables_to_check:
            result = db.query(f"SELECT count(*) as cnt FROM {table}")
            assert result['cnt'].iloc[0] > 0, f"Table {table} is empty"

    def test_data_consistency(self):
        """Test data relationships are consistent."""
        from ..database import Database

        db = Database(db_path=None)
        _ = db.connection

        # Incentive completions should be <= accepts
        result = db.query("""
            SELECT
                sum(case when was_accepted then 1 else 0 end) as accepts,
                sum(case when was_completed then 1 else 0 end) as completions
            FROM marts.fct_incentive_events
        """)

        assert result['accepts'].iloc[0] >= result['completions'].iloc[0]


class TestVisualizationRendering:
    """Test that visualizations render without errors."""

    def test_all_traffic_visualizations(self):
        """Test traffic tab visualizations render."""
        from ..components import (
            get_speed_heatmap_data,
            create_speed_heatmap,
            get_hourly_volume_data,
            create_hourly_volume_chart,
        )

        # These should not raise exceptions
        heatmap_data = get_speed_heatmap_data()
        create_speed_heatmap(heatmap_data)

        volume_data = get_hourly_volume_data()
        create_hourly_volume_chart(volume_data)

    def test_all_incentive_visualizations(self):
        """Test incentive tab visualizations render."""
        from ..components import (
            get_incentive_funnel_data,
            create_funnel_chart,
            get_spend_by_type_data,
            create_spend_chart,
        )

        funnel_data = get_incentive_funnel_data()
        create_funnel_chart(funnel_data)

        spend_data = get_spend_by_type_data()
        create_spend_chart(spend_data)

    def test_all_simulation_visualizations(self):
        """Test simulation tab visualizations render."""
        from ..components import (
            get_scenario_comparison_data,
            create_scenario_comparison_chart,
            create_cost_effectiveness_chart,
        )

        data = get_scenario_comparison_data()
        create_scenario_comparison_chart(data)
        create_cost_effectiveness_chart(data)
