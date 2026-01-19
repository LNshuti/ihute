"""
Nashville Transportation Incentive Simulation Dashboard - COMPLETE INTEGRATED APP

Full dashboard with all tabs including the new Nashville Transportation Simulation.
Based on 2020 Census DHC and ACS 2016-2020 commuting flow data.

Run with: python app.py
Access at: http://localhost:7860
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# IMPORT NASHVILLE SIMULATION COMPONENTS
# ============================================================================

from nashville_sim_integration import create_nashville_simulation_tab

# ============================================================================
# MOCK COMPONENTS FOR EXISTING TABS (Replace with your actual components)
# ============================================================================
# These are simplified versions - replace with your actual implementations

def get_speed_heatmap_data():
    hours = np.arange(24)
    corridors = ['I-24 Downtown', 'I-24 West', 'I-24 East', 'I-75', 'I-440']
    data = np.random.rand(len(corridors), 24) * 60 + 20
    return data, hours, corridors

def create_speed_heatmap(data):
    data_array, hours, corridors = data
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data_array, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h}:00" for h in hours])
    ax.set_yticks(range(len(corridors)))
    ax.set_yticklabels(corridors)
    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_ylabel('Corridor', fontweight='bold')
    ax.set_title('Speed Heatmap by Corridor and Time', fontweight='bold', fontsize=12)
    plt.colorbar(im, ax=ax, label='Speed (mph)')
    plt.tight_layout()
    return fig

def get_hourly_volume_data():
    hours = np.arange(24)
    volume = 1000 + 2000 * np.sin(np.pi * hours / 12) + np.random.normal(0, 200, 24)
    return hours, volume

def create_hourly_volume_chart(data):
    hours, volume = data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(hours, volume, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_ylabel('Vehicle Count', fontweight='bold')
    ax.set_title('Hourly Traffic Volume', fontweight='bold', fontsize=12)
    ax.set_xticks(range(24))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def get_congestion_timeline_data():
    hours = np.arange(24)
    speed = 55 - 20 * np.abs(np.sin(np.pi * hours / 12))
    return hours, speed

def create_congestion_timeline(data):
    hours, speed = data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hours, speed, marker='o', linewidth=2.5, markersize=8, color='darkred')
    ax.fill_between(hours, speed, alpha=0.3, color='red')
    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_ylabel('Average Speed (mph)', fontweight='bold')
    ax.set_title('Traffic Speed Timeline', fontweight='bold', fontsize=12)
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def get_incentive_funnel_data():
    stages = ['Awareness', 'Interest', 'Signup', 'Active', 'Retained']
    users = [10000, 4500, 2200, 1800, 1500]
    return stages, users

def create_funnel_chart(data):
    stages, users = data
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(stages)))
    ax.barh(stages, users, color=colors, edgecolor='black', linewidth=1.5)
    for i, (stage, user) in enumerate(zip(stages, users)):
        ax.text(user + 100, i, f'{user:,}', va='center', fontweight='bold')
    ax.set_xlabel('Number of Users', fontweight='bold')
    ax.set_title('Incentive Program Conversion Funnel', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

def get_spend_by_type_data():
    types = ['Carpool Bonus', 'Transit Pass', 'HOV Incentive', 'Admin']
    spend = [45000, 32000, 18000, 5000]
    return types, spend

def create_spend_chart(data):
    types, spend = data
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    wedges, texts, autotexts = ax.pie(spend, labels=types, autopct='%1.1f%%', 
                                        colors=colors, startangle=90, textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('Program Spending by Type', fontweight='bold', fontsize=12)
    plt.tight_layout()
    return fig

def get_effectiveness_data():
    programs = ['Carpool Bonus', 'Transit Pass', 'HOV Incentive']
    cost_per_vmt = [2.50, 1.80, 3.20]
    return programs, cost_per_vmt

def create_effectiveness_chart(data):
    programs, cost = data
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(programs, cost, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Cost per VMT Reduced ($)', fontweight='bold')
    ax.set_title('Cost Effectiveness by Program', fontweight='bold', fontsize=12)
    for bar, c in zip(bars, cost):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'${height:.2f}', ha='center', va='bottom', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def get_uptake_trend_data():
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    carpool = [120, 185, 310, 480, 620, 800]
    transit = [85, 125, 190, 280, 365, 450]
    return months, carpool, transit

def create_uptake_chart(data):
    months, carpool, transit = data
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(months))
    width = 0.35
    ax.bar(x - width/2, carpool, width, label='Carpool', color='#FF6B6B', edgecolor='black')
    ax.bar(x + width/2, transit, width, label='Transit', color='#4ECDC4', edgecolor='black')
    ax.set_xlabel('Month', fontweight='bold')
    ax.set_ylabel('New Users', fontweight='bold')
    ax.set_title('Program Uptake by Mode', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def get_elasticity_data():
    incentive = np.linspace(0, 100, 50)
    elasticity = 0.5 * (1 - np.exp(-0.05 * incentive)) + np.random.normal(0, 0.02, 50)
    return incentive, elasticity

def create_elasticity_curve(data):
    incentive, elasticity = data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(incentive, elasticity, alpha=0.6, s=50, color='#4ECDC4', edgecolors='black')
    z = np.polyfit(incentive, elasticity, 2)
    p = np.poly1d(z)
    ax.plot(incentive, p(incentive), "r-", linewidth=2.5, label='Fitted Curve')
    ax.set_xlabel('Incentive Amount ($)', fontweight='bold')
    ax.set_ylabel('Mode Shift Elasticity', fontweight='bold')
    ax.set_title('Incentive Elasticity Curve', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def get_model_metrics():
    return {'rmse': 0.082, 'r2': 0.876, 'mae': 0.051, 'accuracy': 0.923}

def create_model_metrics_display(metrics):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    for idx, (ax, name, value) in enumerate(zip(axes.flat, metric_names, metric_values)):
        ax.barh([0], [value], color=['#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'][idx])
        ax.set_xlim(0, 1)
        ax.set_title(name.upper(), fontweight='bold')
        ax.text(value + 0.02, 0, f'{value:.3f}', va='center', fontweight='bold')
        ax.set_yticks([])
        ax.set_xlabel('Value')
    
    plt.tight_layout()
    return fig

def get_feature_importance():
    features = ['Incentive Size', 'Distance', 'Income', 'Current Mode', 'Weather']
    importance = [0.35, 0.22, 0.18, 0.15, 0.10]
    return features, importance

def create_feature_importance_chart(data):
    features, importance = data
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    ax.barh(features, importance, color=colors, edgecolor='black', linewidth=1.5)
    for i, (feat, imp) in enumerate(zip(features, importance)):
        ax.text(imp + 0.01, i, f'{imp:.2f}', va='center', fontweight='bold')
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title('Model Feature Importance', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

def create_predicted_vs_actual():
    actual = np.random.normal(100, 20, 100)
    predicted = actual + np.random.normal(0, 15, 100)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(actual, predicted, alpha=0.6, s=50, color='#FF6B6B', edgecolors='black')
    min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual', fontweight='bold')
    ax.set_ylabel('Predicted', fontweight='bold')
    ax.set_title('Predicted vs Actual', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def get_scenario_comparison_data():
    scenarios = ['Baseline', 'Moderate', 'Aggressive']
    vmt_reduction = [0, 8.5, 15.2]
    cost = [0, 250000, 500000]
    return scenarios, vmt_reduction, cost

def create_scenario_comparison_chart(data):
    scenarios, vmt, cost = data
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(scenarios))
    ax.bar(x, vmt, color=['#95E1D3', '#56AB2F', '#003d82'], edgecolor='black', linewidth=1.5)
    ax.set_ylabel('VMT Reduction (%)', fontweight='bold')
    ax.set_title('Scenario Comparison: VMT Reduction', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    for i, v in enumerate(vmt):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def create_cost_effectiveness_chart(data):
    scenarios, vmt, cost = data
    fig, ax = plt.subplots(figsize=(12, 6))
    cost_per_vmt = [c / v if v > 0 else 0 for c, v in zip(cost, vmt)]
    colors = ['#95E1D3', '#56AB2F', '#003d82']
    ax.bar(scenarios, cost_per_vmt, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Cost per % VMT Reduction', fontweight='bold')
    ax.set_title('Cost Effectiveness by Scenario', fontweight='bold', fontsize=12)
    for i, v in enumerate(cost_per_vmt):
        if v > 0:
            ax.text(i, v + 1000, f'${v:,.0f}', ha='center', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def create_baseline_treatment_comparison(data, scenario):
    scenarios, vmt, cost = data
    fig, ax = plt.subplots(figsize=(12, 6))
    baseline_vmt = [0, 8.5, 15.2]
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, [0, 0, 0], width, label='Baseline', color='gray', edgecolor='black')
    ax.bar(x + width/2, baseline_vmt, width, label=f'{scenario if scenario else "Treatment"}', 
           color='#4ECDC4', edgecolor='black')
    ax.set_ylabel('VMT Reduction (%)', fontweight='bold')
    ax.set_title('Baseline vs Treatment', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Carpool', 'Transit', 'HOV'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def get_scenario_list():
    return ['Moderate Incentive', 'Aggressive Incentive', 'Comprehensive Program']

def get_kpi_data():
    return {
        'vmt_reduction_pct': 12.5,
        'avg_occupancy': 2.3,
        'peak_shift_pct': 8.2,
        'incentive_efficiency': 45.50,
        'carpool_rate': 0.145,
        'avg_speed_improvement': 7.3
    }

def create_kpi_gauge(value, label, suffix='', min_val=0, max_val=100, thresholds=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Draw gauge
    theta = np.linspace(0, np.pi, 100)
    r = 1
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    ax.plot(x, y, 'k-', linewidth=2)
    ax.fill_between(x, 0, y, alpha=0.1, color='gray')
    
    # Add colored ranges
    if thresholds:
        n_ranges = len(thresholds) + 1
        colors = ['red', 'yellow', 'green'][:n_ranges]
        for i, thresh in enumerate(thresholds):
            fraction = (thresh - min_val) / (max_val - min_val)
            theta_thresh = fraction * np.pi
            ax.axvline(x=np.cos(theta_thresh), ymin=0, ymax=np.sin(theta_thresh), 
                       color=colors[i], linewidth=2, alpha=0.5)
    
    # Needle
    value_fraction = (value - min_val) / (max_val - min_val)
    theta_val = value_fraction * np.pi
    ax.arrow(0, 0, 0.8 * np.cos(theta_val), 0.8 * np.sin(theta_val), 
            head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=3)
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.2, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.text(0, -0.15, f'{value:.1f}{suffix}', ha='center', fontsize=20, fontweight='bold')
    ax.text(0, 1.2, label, ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_metric_card(value, label, prefix='', suffix='', format_str='.1f'):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.6, f'{prefix}{value:{format_str}}{suffix}', 
           ha='center', va='center', fontsize=32, fontweight='bold', 
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(0.5, 0.2, label, ha='center', va='center', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    return fig

def get_trend_data():
    months = np.arange(12)
    vmt = 1000 - 50 * months + np.random.normal(0, 30, 12)
    speed = 45 + 5 * np.sin(months / 6) + np.random.normal(0, 2, 12)
    carpool = 100 + 20 * months + np.random.normal(0, 10, 12)
    return {'vmt': vmt, 'speed': speed, 'carpool': carpool}

def create_sparkline(data, label):
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(len(data))
    ax.plot(x, data, marker='o', linewidth=2, markersize=6, color='#4ECDC4')
    ax.fill_between(x, data, alpha=0.2, color='#4ECDC4')
    ax.set_title(label, fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    plt.tight_layout()
    return fig

def get_corridor_data():
    corridors = ['I-24 Downtown', 'I-24 West', 'I-24 East', 'I-75', 'I-440']
    avg_speed = [35, 45, 42, 55, 50]
    return corridors, avg_speed

def create_corridor_map(data):
    corridors, speeds = data
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['red' if s < 40 else 'yellow' if s < 50 else 'green' for s in speeds]
    bars = ax.barh(corridors, speeds, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Average Speed (mph)', fontweight='bold')
    ax.set_title('I-24 Corridor Speed Map', fontweight='bold', fontsize=12)
    for bar, speed in zip(bars, speeds):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
               f'{speed} mph', ha='left', va='center', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

def get_zone_stats():
    zones = ['Downtown', 'Midtown', 'West End', 'Bellevue', 'Airport']
    congestion = [85, 65, 72, 45, 55]
    return zones, congestion

def create_zone_comparison(data):
    zones, congestion = data
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn_r(np.array(congestion) / 100)
    ax.bar(zones, congestion, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Congestion Level (%)', fontweight='bold')
    ax.set_title('Zone Congestion Comparison', fontweight='bold', fontsize=12)
    for i, (zone, cong) in enumerate(zip(zones, congestion)):
        ax.text(i, cong + 2, f'{cong}%', ha='center', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

# ============================================================================
# TAB CREATION FUNCTIONS
# ============================================================================

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


def create_app():
    """Create the main Gradio application with all tabs including Nashville simulation."""
    with gr.Blocks(
        title="Nashville Incentive Simulation Dashboard",
        theme=gr.themes.Soft()
    ) as app:
        gr.Markdown(
            """
            # 🚗 Nashville Transportation Incentive Simulation Dashboard
            
            **Comprehensive Analysis Platform** for the IHUTE Project
            
            Analyze traffic flow, incentive effectiveness, behavioral modeling, transportation simulations,
            and Nashville-Davidson MSA transportation patterns using official government data.
            
            ---
            """
        )

        with gr.Tabs():

            # Existing tabs
            with gr.TabItem("🚦 Incentive Analytics"):
                create_incentive_tab()

            with gr.TabItem("🧠 Behavioral Calibration"):
                create_behavioral_tab()

            with gr.TabItem("🎯 Simulation Comparison"):
                create_simulation_tab()

            with gr.TabItem("📊 Live Metrics"):
                create_metrics_tab()

            with gr.TabItem("🗺️ Corridor Map"):
                create_map_tab()

            # NEW TAB: Nashville Transportation Simulation
            with gr.TabItem("🌐 Nashville Simulation"):
                create_nashville_simulation_tab()

        gr.Markdown(
            """
            ---
            
            ### 📚 Data Sources & Attribution
            
            **Traffic Data:**
            - LADDMS I-24 MOTION trajectories
            - Hytch rideshare trip data
            - Simulation outputs
            
            **Transportation Analysis:**
            - U.S. Census Bureau, 2020 Demographic and Housing Characteristics File (DHC)
            - U.S. Census Bureau, American Community Survey (ACS) 2016-2020 5-Year Estimates
            - U.S. Bureau of Labor Statistics Employment Data
            - LODES (Longitudinal Employer-Household Dynamics)
            
            **Technology Stack:** Python, Pandas, Matplotlib, Gradio, DuckDB, dbt
            
            **Repository:** [GitHub - IHUTE Project](https://github.com/LNshuti/ihute)
            
            **Created:** January 2026 | **Version:** 1.0
            """
        )

    return app


def main():
    """Main entry point for the Gradio app."""
    print("\n" + "="*80)
    print("🚀 NASHVILLE TRANSPORTATION INCENTIVE SIMULATION DASHBOARD")
    print("="*80)
    print("\n✅ Starting dashboard server...\n")
    
    demo = create_app()
    
    print("="*80)
    print("📊 DASHBOARD RUNNING")
    print("="*80)
    print("\n🌐 Open your browser and navigate to:")
    print("   → http://localhost:7860\n")
    print("📑 Tabs Available:")
    print("   1. 🚦 Incentive Analytics")
    print("   2. 🧠 Behavioral Calibration")
    print("   3. 🎯 Simulation Comparison")
    print("   4. 📊 Live Metrics")
    print("   5. 🗺️  Corridor Map")
    print("   6. 🌐 Nashville Simulation (NEW!) ⭐")
    print("      └─ 8 sub-tabs with geographic, employment, commuting, and impact analysis")
    print("\n⏹️  Press Ctrl+C to stop the server\n")
    print("="*80 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard stopped gracefully.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nMake sure you have installed the required dependencies:")
        print("   pip install -r requirements.txt")
        raise
