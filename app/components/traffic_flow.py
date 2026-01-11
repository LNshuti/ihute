"""
Traffic flow visualization component.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database import query


def get_speed_heatmap_data() -> pd.DataFrame:
    """Get data for speed heatmap."""
    sql = """
    SELECT
        extract(hour from hour_bucket) as hour,
        extract(dow from hour_bucket) as day_of_week,
        avg(avg_speed_mph) as avg_speed
    FROM main_marts.fct_corridor_flows
    GROUP BY 1, 2
    ORDER BY 2, 1
    """
    return query(sql)


def create_speed_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create speed heatmap visualization."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    # Pivot for heatmap
    pivot = df.pivot(index='day_of_week', columns='hour', values='avg_speed')

    day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{h}:00" for h in range(24)],
        y=[day_names[int(d)] for d in pivot.index],
        colorscale='RdYlGn',
        colorbar=dict(title='Speed (mph)'),
        hovertemplate='%{y} %{x}<br>Speed: %{z:.1f} mph<extra></extra>'
    ))

    fig.update_layout(
        title='Average Speed by Hour and Day',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=400
    )

    return fig


def get_hourly_volume_data() -> pd.DataFrame:
    """Get hourly traffic volume data."""
    sql = """
    SELECT
        extract(hour from hour_bucket) as hour,
        time_period,
        sum(vehicle_count) as total_vehicles,
        avg(avg_speed_mph) as avg_speed
    FROM main_marts.fct_corridor_flows
    GROUP BY 1, 2
    ORDER BY 1
    """
    return query(sql)


def create_hourly_volume_chart(df: pd.DataFrame) -> go.Figure:
    """Create hourly volume bar chart."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    fig = go.Figure()

    # Add bars for each time period
    colors = {'AM_PEAK': '#e74c3c', 'PM_PEAK': '#e67e22', 'OFF_PEAK': '#27ae60'}

    for period in df['time_period'].unique():
        period_df = df[df['time_period'] == period]
        fig.add_trace(go.Bar(
            x=period_df['hour'],
            y=period_df['total_vehicles'],
            name=period.replace('_', ' ').title(),
            marker_color=colors.get(period, '#3498db'),
            hovertemplate='Hour: %{x}<br>Vehicles: %{y:,}<extra></extra>'
        ))

    fig.update_layout(
        title='Traffic Volume by Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Total Vehicles',
        barmode='stack',
        height=400,
        showlegend=True
    )

    return fig


def get_congestion_timeline_data() -> pd.DataFrame:
    """Get congestion timeline data."""
    sql = """
    SELECT
        hour_bucket,
        corridor_id,
        avg_speed_mph,
        level_of_service
    FROM main_marts.fct_corridor_flows
    ORDER BY hour_bucket
    """
    return query(sql)


def create_congestion_timeline(df: pd.DataFrame) -> go.Figure:
    """Create congestion timeline chart."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['hour_bucket'],
        y=df['avg_speed_mph'],
        mode='lines',
        name='Speed',
        line=dict(color='#3498db', width=2),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.2)',
        hovertemplate='%{x}<br>Speed: %{y:.1f} mph<extra></extra>'
    ))

    # Add threshold lines
    fig.add_hline(y=55, line_dash="dash", line_color="green",
                  annotation_text="Free Flow (55 mph)")
    fig.add_hline(y=30, line_dash="dash", line_color="orange",
                  annotation_text="Congested (30 mph)")
    fig.add_hline(y=15, line_dash="dash", line_color="red",
                  annotation_text="Severe (15 mph)")

    fig.update_layout(
        title='Speed Over Time',
        xaxis_title='Time',
        yaxis_title='Average Speed (mph)',
        height=400,
        yaxis=dict(range=[0, 80])
    )

    return fig


def render_traffic_flow_tab():
    """Render the traffic flow tab content."""
    import gradio as gr

    with gr.Column():
        gr.Markdown("## Traffic Flow Analysis")
        gr.Markdown("Real-time and historical traffic patterns on I-24 corridor")

        with gr.Row():
            with gr.Column(scale=2):
                heatmap_data = get_speed_heatmap_data()
                heatmap_plot = gr.Plot(
                    value=create_speed_heatmap(heatmap_data),
                    label="Speed Heatmap"
                )

        with gr.Row():
            with gr.Column():
                volume_data = get_hourly_volume_data()
                volume_plot = gr.Plot(
                    value=create_hourly_volume_chart(volume_data),
                    label="Hourly Volume"
                )
            with gr.Column():
                timeline_data = get_congestion_timeline_data()
                timeline_plot = gr.Plot(
                    value=create_congestion_timeline(timeline_data),
                    label="Congestion Timeline"
                )

    return heatmap_plot, volume_plot, timeline_plot
