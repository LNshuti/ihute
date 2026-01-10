"""
Incentive analytics visualization component.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..database import query


def get_incentive_funnel_data() -> pd.DataFrame:
    """Get incentive conversion funnel data."""
    sql = """
    SELECT
        incentive_type,
        count(*) as total_offers,
        sum(case when was_accepted then 1 else 0 end) as accepts,
        sum(case when was_completed then 1 else 0 end) as completions
    FROM marts.fct_incentive_events
    GROUP BY incentive_type
    """
    return query(sql)


def create_funnel_chart(df: pd.DataFrame) -> go.Figure:
    """Create incentive conversion funnel."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    # Aggregate across types
    totals = df[['total_offers', 'accepts', 'completions']].sum()

    fig = go.Figure(go.Funnel(
        y=['Offers', 'Accepted', 'Completed'],
        x=[totals['total_offers'], totals['accepts'], totals['completions']],
        textinfo="value+percent initial",
        marker=dict(color=['#3498db', '#2ecc71', '#27ae60'])
    ))

    fig.update_layout(
        title='Incentive Conversion Funnel',
        height=400
    )

    return fig


def get_spend_by_type_data() -> pd.DataFrame:
    """Get spending by incentive type."""
    sql = """
    SELECT
        incentive_type,
        sum(actual_payout) as total_spend,
        count(*) as n_events,
        avg(actual_payout) as avg_payout
    FROM marts.fct_incentive_events
    WHERE was_completed
    GROUP BY incentive_type
    ORDER BY total_spend DESC
    """
    return query(sql)


def create_spend_chart(df: pd.DataFrame) -> go.Figure:
    """Create spending breakdown chart."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    fig = px.pie(
        df,
        values='total_spend',
        names='incentive_type',
        title='Spending by Incentive Type',
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label}<br>$%{value:,.2f}<br>%{percent}<extra></extra>'
    )

    fig.update_layout(height=400)

    return fig


def get_effectiveness_data() -> pd.DataFrame:
    """Get incentive effectiveness metrics."""
    sql = """
    SELECT
        incentive_type,
        count(*) as n_completed,
        sum(actual_payout) as total_cost,
        avg(actual_payout) as avg_cost
    FROM marts.fct_incentive_events
    WHERE was_completed
    GROUP BY incentive_type
    """
    return query(sql)


def create_effectiveness_chart(df: pd.DataFrame) -> go.Figure:
    """Create cost effectiveness chart."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['incentive_type'],
        y=df['avg_cost'],
        name='Avg Cost per Completion',
        marker_color='#3498db',
        text=df['avg_cost'].apply(lambda x: f'${x:.2f}'),
        textposition='auto',
        hovertemplate='%{x}<br>Avg Cost: $%{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title='Average Cost per Completed Incentive',
        xaxis_title='Incentive Type',
        yaxis_title='Average Cost ($)',
        height=400
    )

    return fig


def get_uptake_trend_data() -> pd.DataFrame:
    """Get incentive uptake trend over time."""
    sql = """
    SELECT
        incentive_type,
        final_outcome,
        count(*) as count
    FROM marts.fct_incentive_events
    GROUP BY incentive_type, final_outcome
    """
    return query(sql)


def create_uptake_chart(df: pd.DataFrame) -> go.Figure:
    """Create uptake by outcome chart."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    fig = px.bar(
        df,
        x='incentive_type',
        y='count',
        color='final_outcome',
        title='Incentive Outcomes by Type',
        barmode='stack',
        color_discrete_map={
            'COMPLETED': '#27ae60',
            'ACCEPTED_PENDING': '#f39c12',
            'REJECTED': '#e74c3c',
            'OFFERED_PENDING': '#95a5a6'
        }
    )

    fig.update_layout(
        xaxis_title='Incentive Type',
        yaxis_title='Count',
        height=400
    )

    return fig
