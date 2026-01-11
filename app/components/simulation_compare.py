"""
Simulation comparison visualization component.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database import query


def get_scenario_list() -> list[str]:
    """Get list of available scenarios."""
    sql = """
    SELECT DISTINCT scenario_name
    FROM main_marts.fct_simulation_runs
    ORDER BY scenario_name
    """
    df = query(sql)
    return df['scenario_name'].tolist() if not df.empty else []


def get_scenario_comparison_data() -> pd.DataFrame:
    """Get scenario comparison metrics."""
    sql = """
    SELECT
        scenario_name,
        n_agents,
        treatment_avg_speed,
        baseline_avg_speed,
        speed_improvement_pct,
        vmt_reduction_pct,
        peak_reduction_pct,
        treatment_spend
    FROM main_marts.fct_simulation_runs
    """
    return query(sql)


def create_scenario_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Create scenario comparison bar chart."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    fig = go.Figure()

    metrics = ['speed_improvement_pct', 'vmt_reduction_pct', 'peak_reduction_pct']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    names = ['Speed Improvement', 'VMT Reduction', 'Peak Reduction']

    for metric, color, name in zip(metrics, colors, names):
        fig.add_trace(go.Bar(
            x=df['scenario_name'],
            y=df[metric],
            name=name,
            marker_color=color,
            text=df[metric].apply(lambda x: f'{x:.1f}%'),
            textposition='auto',
            hovertemplate='%{x}<br>%{y:.1f}%<extra></extra>'
        ))

    fig.update_layout(
        title='Scenario Performance Comparison',
        xaxis_title='Scenario',
        yaxis_title='Improvement (%)',
        barmode='group',
        height=400
    )

    return fig


def create_cost_effectiveness_chart(df: pd.DataFrame) -> go.Figure:
    """Create cost effectiveness comparison."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    # Calculate cost per % improvement
    df = df.copy()
    df['cost_per_vmt_pct'] = df['treatment_spend'] / df['vmt_reduction_pct'].clip(lower=0.1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['treatment_spend'],
        y=df['vmt_reduction_pct'],
        mode='markers+text',
        marker=dict(
            size=20,
            color=df['speed_improvement_pct'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Speed Imp. %')
        ),
        text=df['scenario_name'],
        textposition='top center',
        hovertemplate='%{text}<br>Spend: $%{x:,.0f}<br>VMT Reduction: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title='Cost vs. VMT Reduction by Scenario',
        xaxis_title='Total Spend ($)',
        yaxis_title='VMT Reduction (%)',
        height=400
    )

    return fig


def create_baseline_treatment_comparison(df: pd.DataFrame, scenario: str = None) -> go.Figure:
    """Create baseline vs treatment comparison for a scenario."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    if scenario:
        df = df[df['scenario_name'] == scenario]

    if df.empty:
        return go.Figure().add_annotation(text="Scenario not found", showarrow=False)

    row = df.iloc[0]

    fig = go.Figure()

    categories = ['Avg Speed (mph)', 'VMT (scaled)', 'Peak Demand (scaled)']
    baseline = [row['baseline_avg_speed'], 100, 100]  # Normalized
    treatment = [
        row['treatment_avg_speed'],
        100 - row['vmt_reduction_pct'],
        100 - row['peak_reduction_pct']
    ]

    fig.add_trace(go.Bar(
        name='Baseline',
        x=categories,
        y=baseline,
        marker_color='#95a5a6'
    ))

    fig.add_trace(go.Bar(
        name='Treatment',
        x=categories,
        y=treatment,
        marker_color='#3498db'
    ))

    fig.update_layout(
        title=f'Baseline vs Treatment: {scenario or "All Scenarios"}',
        yaxis_title='Value',
        barmode='group',
        height=400
    )

    return fig


def get_metrics_summary(df: pd.DataFrame) -> dict:
    """Get summary metrics across all scenarios."""
    if df.empty:
        return {}

    return {
        'avg_speed_improvement': df['speed_improvement_pct'].mean(),
        'avg_vmt_reduction': df['vmt_reduction_pct'].mean(),
        'avg_peak_reduction': df['peak_reduction_pct'].mean(),
        'total_spend': df['treatment_spend'].sum(),
        'n_scenarios': len(df)
    }
