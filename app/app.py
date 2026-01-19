"""
Nashville Transportation Incentive Simulation Dashboard

A Gradio-based interactive dashboard for visualizing traffic flow,
incentive effectiveness, behavioral calibration, and simulation results.
"""

import gradio as gr

from components import (
    # Traffic flow
    get_speed_heatmap_data, create_speed_heatmap,
    get_hourly_volume_data, create_hourly_volume_chart,
    get_congestion_timeline_data, create_congestion_timeline,
    # Incentive analytics
    get_incentive_funnel_data, create_funnel_chart,
    get_spend_by_type_data, create_spend_chart,
    get_effectiveness_data, create_effectiveness_chart,
    get_uptake_trend_data, create_uptake_chart,
    # Behavioral calibration
    get_elasticity_data, create_elasticity_curve,
    get_model_metrics, create_model_metrics_display,
    get_feature_importance, create_feature_importance_chart,
    create_predicted_vs_actual,
    # Simulation comparison
    get_scenario_comparison_data, create_scenario_comparison_chart,
    create_cost_effectiveness_chart, create_baseline_treatment_comparison,
    get_scenario_list,
    # Real-time metrics
    get_kpi_data, create_kpi_gauge, create_metric_card,
    get_trend_data, create_sparkline,
    # Geo map
    get_corridor_data, create_corridor_map,
    get_zone_stats, create_zone_comparison,
    # Demographics
    get_demographics_summary, create_summary_cards,
    get_income_distribution, create_income_distribution_chart,
    create_poverty_distribution_chart, get_zcta_details,
    create_zcta_table, create_behavioral_impact_chart,
)

from nashville_sim_integration import create_nashville_simulation_tab


def create_traffic_tab():
    """Create traffic flow analysis tab."""
    with gr.Column():
        gr.Markdown("## Traffic Flow Analysis")

        heatmap_data = get_speed_heatmap_data()
        gr.Plot(value=create_speed_heatmap(heatmap_data), label="Speed Heatmap")

        with gr.Row():
            volume_data = get_hourly_volume_data()
            gr.Plot(value=create_hourly_volume_chart(volume_data), label="Hourly Volume")

            timeline_data = get_congestion_timeline_data()
            gr.Plot(value=create_congestion_timeline(timeline_data), label="Speed Timeline")


def create_incentive_tab():
    """Create incentive analytics tab."""
    with gr.Column():
        gr.Markdown("## Incentive Analytics")
        gr.Markdown("Analyze incentive program performance and cost-effectiveness")

        with gr.Row():
            funnel_data = get_incentive_funnel_data()
            gr.Plot(value=create_funnel_chart(funnel_data), label="Conversion Funnel")

            spend_data = get_spend_by_type_data()
            gr.Plot(value=create_spend_chart(spend_data), label="Spending by Type")

        with gr.Row():
            effectiveness_data = get_effectiveness_data()
            gr.Plot(value=create_effectiveness_chart(effectiveness_data), label="Cost Effectiveness")

            uptake_data = get_uptake_trend_data()
            gr.Plot(value=create_uptake_chart(uptake_data), label="Outcomes by Type")


def create_behavioral_tab():
    """Create behavioral calibration tab."""
    with gr.Column():
        gr.Markdown("## Behavioral Model Calibration")
        gr.Markdown("ML model performance and incentive elasticity analysis")

        with gr.Row():
            elasticity_data = get_elasticity_data()
            gr.Plot(value=create_elasticity_curve(elasticity_data), label="Elasticity Curve")

            metrics = get_model_metrics()
            gr.Plot(value=create_model_metrics_display(metrics), label="Model Performance")

        with gr.Row():
            importance_data = get_feature_importance()
            gr.Plot(value=create_feature_importance_chart(importance_data), label="Feature Importance")

            gr.Plot(value=create_predicted_vs_actual(), label="Predicted vs Actual")


def create_simulation_tab():
    """Create simulation comparison tab."""
    with gr.Column():
        gr.Markdown("## Simulation Scenario Comparison")
        gr.Markdown("Compare treatment scenarios against baseline")

        scenario_data = get_scenario_comparison_data()

        gr.Plot(value=create_scenario_comparison_chart(scenario_data), label="Performance Comparison")

        with gr.Row():
            gr.Plot(value=create_cost_effectiveness_chart(scenario_data), label="Cost vs Impact")

            scenarios = get_scenario_list()
            if scenarios:
                scenario_dropdown = gr.Dropdown(
                    choices=scenarios,
                    value=scenarios[0] if scenarios else None,
                    label="Select Scenario"
                )
            gr.Plot(
                value=create_baseline_treatment_comparison(scenario_data, scenarios[0] if scenarios else None),
                label="Baseline vs Treatment"
            )


def create_metrics_tab():
    """Create real-time metrics tab."""
    with gr.Column():
        gr.Markdown("## Key Performance Indicators")
        gr.Markdown("Real-time metrics and trends")

        kpis = get_kpi_data()
        trends = get_trend_data()

        with gr.Row():
            gr.Plot(value=create_kpi_gauge(kpis['vmt_reduction_pct'], 'VMT Reduction', '%', 0, 25))
            gr.Plot(value=create_kpi_gauge(kpis['avg_occupancy'], 'Avg Occupancy', '', 1, 3, [1.5, 2.0, 3.0]))
            gr.Plot(value=create_kpi_gauge(kpis['peak_shift_pct'], 'Peak Shift', '%', 0, 20))

        with gr.Row():
            gr.Plot(value=create_metric_card(kpis['incentive_efficiency'], 'Cost per VMT Reduced', prefix='$', format_str='.2f'))
            gr.Plot(value=create_metric_card(kpis['carpool_rate'] * 100, 'Carpool Rate', suffix='%', format_str='.1f'))
            gr.Plot(value=create_metric_card(kpis['avg_speed_improvement'], 'Speed Improvement', suffix='%', format_str='.1f'))

        gr.Markdown("### Trends")
        with gr.Row():
            gr.Plot(value=create_sparkline(trends['vmt'], 'VMT Trend'))
            gr.Plot(value=create_sparkline(trends['speed'], 'Speed Trend'))
            gr.Plot(value=create_sparkline(trends['carpool'], 'Carpool Trend'))


def create_map_tab():
    """Create corridor map tab."""
    with gr.Column():
        gr.Markdown("## I-24 Corridor Map")
        gr.Markdown("Interactive map with traffic conditions and segment metrics")

        corridor_data = get_corridor_data()
        gr.Plot(value=create_corridor_map(corridor_data), label="Corridor Map")

        zone_data = get_zone_stats()
        gr.Plot(value=create_zone_comparison(zone_data), label="Zone Comparison")


def create_demographics_tab():
    """Create demographics analysis tab."""
    with gr.Column():
        gr.Markdown("## Demographics Analysis")
        gr.Markdown(
            """
            Agent behavioral heterogeneity calibrated from **population-dyna** platform.

            **376 Tennessee ZCTAs** with income, poverty, and behavioral parameters.
            """
        )

        # Summary statistics
        summary_data = get_demographics_summary()
        gr.Plot(value=create_summary_cards(summary_data), label="Demographics Summary")

        # Distribution charts
        gr.Markdown("### Income & Poverty Distribution")
        income_data = get_income_distribution()

        with gr.Row():
            gr.Plot(
                value=create_income_distribution_chart(income_data),
                label="Income by Quintile"
            )
            gr.Plot(
                value=create_poverty_distribution_chart(income_data),
                label="Poverty by Quintile"
            )

        # Behavioral impact
        gr.Markdown("### Behavioral Calibration")
        gr.Markdown(
            """
            Demographic data calibrates agent preferences:
            - **VOT (Value of Time)**: 50% of hourly wage
            - **Incentive Sensitivity**: 2.66x higher for low-income (Q1 vs Q5)
            - **Car Ownership**: Logistic function of income
            """
        )
        gr.Plot(
            value=create_behavioral_impact_chart(),
            label="Behavioral Parameters"
        )

        # ZCTA details table
        gr.Markdown("### ZCTA Details")
        zcta_data = get_zcta_details()
        gr.Plot(value=create_zcta_table(zcta_data), label="Top ZCTAs by Poverty")


def create_app():
    """Create the main Gradio application."""
    with gr.Blocks(title="Nashville Incentive Simulation Dashboard") as app:
        gr.Markdown(
            """
            # Nashville Transportation Incentive Simulation

            Explore traffic patterns, incentive effectiveness, behavioral models, and simulation results
            for the I-24 corridor in Nashville, TN.
            """
        )

        with gr.Tabs():

            with gr.TabItem("Incentive Analytics"):
                create_incentive_tab()

            with gr.TabItem("Behavioral Calibration"):
                create_behavioral_tab()

            with gr.TabItem("Demographics"):
                create_demographics_tab()

            with gr.TabItem("Simulation Comparison"):
                create_simulation_tab()

            with gr.TabItem("Live Metrics"):
                create_metrics_tab()

            with gr.TabItem("Corridor Map"):
                create_map_tab()

            with gr.TabItem("Nashville Simulation"):
                create_nashville_simulation_tab()

        gr.Markdown(
            """
            ---
            *Data sources: LADDMS I-24 MOTION trajectories, Hytch rideshare trips, population-dyna demographics, simulation outputs*

            Built with DuckDB, dbt, and Gradio | [GitHub](https://github.com/LNshuti/ihute)
            """
        )

    return app


# Create app instance for Hugging Face Spaces
demo = create_app()

# Main entry point for local development
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        share=False
    )
