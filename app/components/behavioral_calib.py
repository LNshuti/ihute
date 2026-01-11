"""
Behavioral calibration visualization component.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database import query


def get_elasticity_data() -> pd.DataFrame:
    """Get incentive elasticity curve data."""
    sql = """
    SELECT
        incentive_bucket,
        n_trips,
        carpool_rate,
        avg_incentive
    FROM main_marts.metrics_elasticity
    ORDER BY avg_incentive
    """
    return query(sql)


def create_elasticity_curve(df: pd.DataFrame) -> go.Figure:
    """Create elasticity curve visualization."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scatter(
        x=df['avg_incentive'],
        y=df['carpool_rate'],
        mode='lines+markers',
        name='Carpool Rate',
        line=dict(color='#3498db', width=3),
        marker=dict(size=10),
        hovertemplate='Incentive: $%{x:.2f}<br>Rate: %{y:.1%}<extra></extra>'
    ))

    # Add annotations for buckets
    for _, row in df.iterrows():
        fig.add_annotation(
            x=row['avg_incentive'],
            y=row['carpool_rate'],
            text=row['incentive_bucket'],
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-30
        )

    fig.update_layout(
        title='Incentive Elasticity Curve',
        xaxis_title='Average Incentive ($)',
        yaxis_title='Carpool Participation Rate',
        yaxis=dict(tickformat='.0%'),
        height=400
    )

    return fig


def get_model_metrics() -> dict:
    """Get model performance metrics (placeholder)."""
    # In production, these would come from the ML models table
    return {
        'auc': 0.78,
        'rmse': 0.15,
        'accuracy': 0.82,
        'n_samples': 369831
    }


def create_model_metrics_display(metrics: dict) -> go.Figure:
    """Create model metrics display."""
    fig = go.Figure()

    categories = ['AUC', 'Accuracy', '1-RMSE']
    values = [metrics['auc'], metrics['accuracy'], 1 - metrics['rmse']]

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Model Performance',
        line_color='#3498db'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=f"Model Performance (n={metrics['n_samples']:,})",
        height=400
    )

    return fig


def get_feature_importance() -> pd.DataFrame:
    """Get feature importance data (placeholder)."""
    return pd.DataFrame({
        'feature': ['incentive_amount', 'distance_miles', 'is_peak_hour',
                    'hour_of_day', 'day_of_week', 'avg_speed'],
        'importance': [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]
    })


def create_feature_importance_chart(df: pd.DataFrame) -> go.Figure:
    """Create feature importance bar chart."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    df = df.sort_values('importance', ascending=True)

    fig = go.Figure(go.Bar(
        x=df['importance'],
        y=df['feature'],
        orientation='h',
        marker_color='#3498db',
        text=df['importance'].apply(lambda x: f'{x:.1%}'),
        textposition='auto',
        hovertemplate='%{y}<br>Importance: %{x:.1%}<extra></extra>'
    ))

    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Feature',
        xaxis=dict(tickformat='.0%'),
        height=400
    )

    return fig


def create_predicted_vs_actual() -> go.Figure:
    """Create predicted vs actual scatter plot (placeholder data)."""
    import numpy as np

    np.random.seed(42)
    n = 100
    actual = np.random.uniform(0, 1, n)
    predicted = actual + np.random.normal(0, 0.1, n)
    predicted = np.clip(predicted, 0, 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual,
        y=predicted,
        mode='markers',
        marker=dict(color='#3498db', size=8, opacity=0.6),
        hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
    ))

    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))

    fig.update_layout(
        title='Predicted vs Actual Carpool Rate',
        xaxis_title='Actual Rate',
        yaxis_title='Predicted Rate',
        height=400,
        showlegend=False
    )

    return fig
