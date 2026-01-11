"""
Geospatial map visualization component.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database import query


def get_corridor_data() -> pd.DataFrame:
    """Get corridor location and metrics data."""
    # Sample I-24 corridor segments
    return pd.DataFrame({
        'segment_id': ['seg_1', 'seg_2', 'seg_3', 'seg_4', 'seg_5'],
        'segment_name': ['I-24 @ Briley Pkwy', 'I-24 @ Harding Pl', 'I-24 @ Downtown',
                        'I-24 @ Shelby Ave', 'I-24 @ Spring St'],
        'latitude': [36.08, 36.10, 36.12, 36.14, 36.16],
        'longitude': [-86.70, -86.72, -86.75, -86.77, -86.79],
        'avg_speed_mph': [35, 28, 22, 30, 45],
        'congestion_level': ['Moderate', 'Severe', 'Severe', 'Moderate', 'Light'],
        'vehicle_count': [450, 520, 580, 490, 380]
    })


def create_corridor_map(df: pd.DataFrame) -> go.Figure:
    """Create interactive corridor map with Plotly."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    # Color scale based on speed
    colors = []
    for speed in df['avg_speed_mph']:
        if speed < 25:
            colors.append('#e74c3c')  # Red - congested
        elif speed < 40:
            colors.append('#f39c12')  # Orange - slow
        else:
            colors.append('#27ae60')  # Green - free flow

    fig = go.Figure()

    # Add corridor line
    fig.add_trace(go.Scattermapbox(
        lat=df['latitude'],
        lon=df['longitude'],
        mode='lines+markers',
        line=dict(width=4, color='#3498db'),
        marker=dict(
            size=15,
            color=colors,
            symbol='circle'
        ),
        text=df.apply(
            lambda r: f"{r['segment_name']}<br>Speed: {r['avg_speed_mph']} mph<br>"
                      f"Volume: {r['vehicle_count']} veh/hr",
            axis=1
        ),
        hoverinfo='text',
        name='I-24 Corridor'
    ))

    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=36.12, lon=-86.75),
            zoom=11
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title='I-24 Corridor Traffic Conditions',
        height=500
    )

    return fig


def create_heatmap_overlay(df: pd.DataFrame) -> go.Figure:
    """Create density heatmap for traffic."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    fig = go.Figure()

    fig.add_trace(go.Densitymapbox(
        lat=df['latitude'],
        lon=df['longitude'],
        z=df['vehicle_count'],
        radius=30,
        colorscale='YlOrRd',
        showscale=True,
        colorbar=dict(title='Volume')
    ))

    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=36.12, lon=-86.75),
            zoom=11
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title='Traffic Density Heatmap',
        height=500
    )

    return fig


def get_zone_stats() -> pd.DataFrame:
    """Get statistics by zone."""
    return pd.DataFrame({
        'zone': ['Downtown', 'Southeast', 'East Nashville', 'Antioch'],
        'avg_speed': [25, 35, 40, 45],
        'carpool_rate': [0.25, 0.18, 0.15, 0.12],
        'incentive_uptake': [0.35, 0.28, 0.22, 0.18]
    })


def create_zone_comparison(df: pd.DataFrame) -> go.Figure:
    """Create zone comparison chart."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['zone'],
        y=df['avg_speed'],
        name='Avg Speed (mph)',
        marker_color='#3498db'
    ))

    fig.add_trace(go.Scatter(
        x=df['zone'],
        y=df['carpool_rate'] * 100,
        name='Carpool Rate (%)',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=10)
    ))

    fig.update_layout(
        title='Zone Performance Comparison',
        xaxis_title='Zone',
        yaxis=dict(title='Speed (mph)', side='left'),
        yaxis2=dict(title='Carpool Rate (%)', side='right', overlaying='y'),
        height=400,
        legend=dict(x=0.7, y=1.1, orientation='h')
    )

    return fig


def get_legend_data() -> dict:
    """Get legend information for map."""
    return {
        'colors': {
            'Free Flow (>40 mph)': '#27ae60',
            'Slow (25-40 mph)': '#f39c12',
            'Congested (<25 mph)': '#e74c3c'
        }
    }
