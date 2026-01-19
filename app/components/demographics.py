"""
Demographics visualization component.

Shows population demographics from population-dyna integration,
including income distribution, VOT calibration, and agent heterogeneity.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database import query


def get_demographics_summary() -> pd.DataFrame:
    """Get ZCTA demographics summary statistics."""
    sql = """
    SELECT
        COUNT(*) as n_zctas,
        ROUND(MIN(poverty_rate), 3) as min_poverty,
        ROUND(MAX(poverty_rate), 3) as max_poverty,
        ROUND(AVG(poverty_rate), 3) as avg_poverty,
        MIN(median_household_income_est) as min_income,
        MAX(median_household_income_est) as max_income,
        ROUND(AVG(median_household_income_est)) as avg_income
    FROM main_marts.dim_demographics
    """
    return query(sql)


def create_summary_cards(df: pd.DataFrame) -> go.Figure:
    """Create summary statistics cards."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    row = df.iloc[0]

    fig = go.Figure()

    # Create a table-style display
    fig.add_trace(go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>'],
            fill_color='#3498db',
            align='left',
            font=dict(color='white', size=14)
        ),
        cells=dict(
            values=[
                ['Total ZCTAs', 'Avg Income', 'Income Range', 'Avg Poverty Rate', 'Poverty Range'],
                [
                    f"{int(row['n_zctas']):,}",
                    f"${int(row['avg_income']):,}",
                    f"${int(row['min_income']):,} - ${int(row['max_income']):,}",
                    f"{row['avg_poverty']:.1%}",
                    f"{row['min_poverty']:.1%} - {row['max_poverty']:.1%}"
                ]
            ],
            fill_color='#ecf0f1',
            align='left',
            font=dict(size=13),
            height=30
        )
    ))

    fig.update_layout(
        title='Demographics Summary (Tennessee ZCTAs)',
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


def get_income_distribution() -> pd.DataFrame:
    """Get income distribution by quintile."""
    sql = """
    SELECT
        income_quintile,
        COUNT(*) as n_zctas,
        ROUND(AVG(median_household_income_est)) as avg_income,
        ROUND(AVG(poverty_rate), 3) as avg_poverty
    FROM main_marts.dim_demographics
    GROUP BY income_quintile
    ORDER BY income_quintile
    """
    return query(sql)


def create_income_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create income distribution by quintile."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    df['quintile_label'] = df['income_quintile'].apply(
        lambda x: f"Q{x} ({'Lowest' if x == 1 else 'Highest' if x == 5 else 'Mid'})"
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['quintile_label'],
        y=df['avg_income'],
        marker_color=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60'],
        text=df['avg_income'].apply(lambda x: f'${x:,.0f}'),
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Avg Income: $%{y:,.0f}<br>ZCTAs: %{customdata[0]}<extra></extra>',
        customdata=df[['n_zctas']].values
    ))

    fig.update_layout(
        title='Average Household Income by Quintile',
        xaxis_title='Income Quintile',
        yaxis_title='Average Income ($)',
        height=400,
        yaxis=dict(tickformat='$,.0f')
    )

    return fig


def create_poverty_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create poverty rate distribution by quintile."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    df['quintile_label'] = df['income_quintile'].apply(
        lambda x: f"Q{x}"
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['quintile_label'],
        y=df['avg_poverty'],
        marker_color=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60'],
        text=df['avg_poverty'].apply(lambda x: f'{x:.1%}'),
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Avg Poverty Rate: %{y:.1%}<extra></extra>'
    ))

    fig.update_layout(
        title='Average Poverty Rate by Quintile',
        xaxis_title='Income Quintile',
        yaxis_title='Poverty Rate',
        height=400,
        yaxis=dict(tickformat='.1%')
    )

    return fig


def get_zcta_details() -> pd.DataFrame:
    """Get detailed ZCTA information for table view."""
    sql = """
    SELECT
        zcta_code,
        ROUND(poverty_rate, 3) as poverty_rate,
        median_household_income_est as income,
        income_quintile as quintile
    FROM main_marts.dim_demographics
    ORDER BY poverty_rate DESC
    LIMIT 20
    """
    return query(sql)


def create_zcta_table(df: pd.DataFrame) -> go.Figure:
    """Create interactive ZCTA details table."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    # Format data for display
    df_display = df.copy()
    df_display['poverty_rate'] = df_display['poverty_rate'].apply(lambda x: f'{x:.1%}')
    df_display['income'] = df_display['income'].apply(lambda x: f'${x:,.0f}')
    df_display['quintile'] = df_display['quintile'].apply(lambda x: f'Q{x}')

    fig = go.Figure()

    fig.add_trace(go.Table(
        header=dict(
            values=['<b>ZCTA</b>', '<b>Poverty Rate</b>', '<b>Income</b>', '<b>Quintile</b>'],
            fill_color='#3498db',
            align='left',
            font=dict(color='white', size=13)
        ),
        cells=dict(
            values=[
                df_display['zcta_code'],
                df_display['poverty_rate'],
                df_display['income'],
                df_display['quintile']
            ],
            fill_color=[
                ['#ecf0f1', '#ffffff'] * (len(df) // 2 + 1)
            ],
            align='left',
            font=dict(size=12),
            height=25
        )
    ))

    fig.update_layout(
        title='Top 20 ZCTAs by Poverty Rate',
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


def create_behavioral_impact_chart() -> go.Figure:
    """
    Create chart showing theoretical behavioral impact.

    Demonstrates how demographics influence VOT and incentive sensitivity.
    This is a theoretical visualization based on calibration formulas.
    """
    # Create theoretical data
    quintiles = list(range(1, 6))
    incomes = [50000, 60000, 65000, 68000, 72000]  # Representative incomes

    # VOT = 50% of hourly wage
    vots = [inc / 2080 * 0.5 for inc in incomes]

    # Incentive sensitivity: 2.0 - (quintile * 0.3)
    base_beta = 0.15
    beta_incentives = [base_beta * (2.0 - (q * 0.3)) for q in quintiles]

    fig = go.Figure()

    # VOT bars
    fig.add_trace(go.Bar(
        x=[f'Q{q}' for q in quintiles],
        y=vots,
        name='Value of Time ($/hr)',
        marker_color='#3498db',
        yaxis='y',
        text=[f'${v:.2f}' for v in vots],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>VOT: $%{y:.2f}/hr<extra></extra>'
    ))

    # Beta incentive line
    fig.add_trace(go.Scatter(
        x=[f'Q{q}' for q in quintiles],
        y=beta_incentives,
        name='Incentive Sensitivity (β)',
        mode='lines+markers',
        marker=dict(size=10, color='#e74c3c'),
        line=dict(width=3, color='#e74c3c'),
        yaxis='y2',
        text=[f'{b:.3f}' for b in beta_incentives],
        textposition='top center',
        hovertemplate='<b>%{x}</b><br>β: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title='Behavioral Parameters by Income Quintile',
        xaxis_title='Income Quintile',
        yaxis=dict(
            title='Value of Time ($/hr)',
            titlefont=dict(color='#3498db'),
            tickfont=dict(color='#3498db')
        ),
        yaxis2=dict(
            title='Incentive Sensitivity (β)',
            titlefont=dict(color='#e74c3c'),
            tickfont=dict(color='#e74c3c'),
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.5, y=1.1, orientation='h', xanchor='center'),
        height=400,
        hovermode='x unified'
    )

    return fig
