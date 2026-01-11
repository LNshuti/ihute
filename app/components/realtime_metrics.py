"""
Real-time metrics KPI component.
"""

import pandas as pd
import plotly.graph_objects as go

from database import query


def get_kpi_data() -> dict:
    """Get current KPI values."""
    # In production, these would be computed from the latest data
    return {
        'vmt_reduction_pct': 12.5,
        'avg_occupancy': 1.85,
        'peak_shift_pct': 8.3,
        'incentive_efficiency': 2.15,
        'carpool_rate': 0.23,
        'avg_speed_improvement': 15.2
    }


def create_kpi_gauge(value: float, title: str, suffix: str = '%',
                     min_val: float = 0, max_val: float = 100,
                     thresholds: list = None) -> go.Figure:
    """Create a gauge chart for a KPI."""
    if thresholds is None:
        thresholds = [max_val * 0.3, max_val * 0.7, max_val]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': suffix},
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#3498db"},
            'steps': [
                {'range': [min_val, thresholds[0]], 'color': "#e74c3c"},
                {'range': [thresholds[0], thresholds[1]], 'color': "#f39c12"},
                {'range': [thresholds[1], max_val], 'color': "#27ae60"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def create_sparkline(values: list, title: str) -> go.Figure:
    """Create a sparkline chart."""
    fig = go.Figure(go.Scatter(
        y=values,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#3498db', width=2),
        fillcolor='rgba(52, 152, 219, 0.2)'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        height=100,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )

    return fig


def create_metric_card(value: float, label: str, delta: float = None,
                       format_str: str = '.1f', prefix: str = '',
                       suffix: str = '') -> go.Figure:
    """Create a metric card with optional delta."""
    delta_ref = None
    if delta is not None:
        delta_ref = {'reference': value - delta, 'relative': True}

    fig = go.Figure(go.Indicator(
        mode="number+delta" if delta is not None else "number",
        value=value,
        number={
            'prefix': prefix,
            'suffix': suffix,
            'valueformat': format_str
        },
        delta=delta_ref,
        title={'text': label, 'font': {'size': 14}}
    ))

    fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def get_trend_data() -> dict:
    """Get trend data for sparklines."""
    # Placeholder - would be computed from time series
    import numpy as np
    np.random.seed(42)

    return {
        'vmt': list(np.cumsum(np.random.randn(20)) + 100),
        'speed': list(45 + np.cumsum(np.random.randn(20) * 0.5)),
        'carpool': list(0.2 + np.cumsum(np.random.randn(20) * 0.01))
    }


def render_kpi_dashboard():
    """Render KPI dashboard layout."""
    import gradio as gr

    kpis = get_kpi_data()
    trends = get_trend_data()

    with gr.Column():
        gr.Markdown("## Key Performance Indicators")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Plot(
                    value=create_kpi_gauge(
                        kpis['vmt_reduction_pct'],
                        'VMT Reduction',
                        '%', 0, 25
                    )
                )
            with gr.Column(scale=1):
                gr.Plot(
                    value=create_kpi_gauge(
                        kpis['avg_occupancy'],
                        'Avg Occupancy',
                        '', 1, 3,
                        [1.5, 2.0, 3.0]
                    )
                )
            with gr.Column(scale=1):
                gr.Plot(
                    value=create_kpi_gauge(
                        kpis['peak_shift_pct'],
                        'Peak Shift',
                        '%', 0, 20
                    )
                )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Plot(
                    value=create_metric_card(
                        kpis['incentive_efficiency'],
                        'Cost per VMT Reduced',
                        prefix='$',
                        format_str='.2f'
                    )
                )
            with gr.Column(scale=1):
                gr.Plot(
                    value=create_metric_card(
                        kpis['carpool_rate'] * 100,
                        'Carpool Rate',
                        suffix='%',
                        format_str='.1f'
                    )
                )
            with gr.Column(scale=1):
                gr.Plot(
                    value=create_metric_card(
                        kpis['avg_speed_improvement'],
                        'Speed Improvement',
                        suffix='%',
                        format_str='.1f'
                    )
                )

        gr.Markdown("### Trends")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Plot(value=create_sparkline(trends['vmt'], 'VMT Trend'))
            with gr.Column(scale=1):
                gr.Plot(value=create_sparkline(trends['speed'], 'Speed Trend'))
            with gr.Column(scale=1):
                gr.Plot(value=create_sparkline(trends['carpool'], 'Carpool Trend'))
