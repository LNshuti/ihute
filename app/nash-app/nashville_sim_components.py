"""
Nashville Transportation Simulation Visualization Components

Creates interactive maps and charts for county, ZIP code, and commuting zone analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


# Geographic coordinates for Nashville area (simplified grid)
NASHVILLE_BOUNDS = {
    'min_lat': 35.92,
    'max_lat': 36.45,
    'min_lon': -87.12,
    'max_lon': -86.45
}

COUNTY_BOUNDS = {
    'Davidson': {'lat': (36.10, 36.22), 'lon': (-86.78, -86.58)},
    'Williamson': {'lat': (35.92, 36.15), 'lon': (-86.82, -86.45)},
    'Wilson': {'lat': (36.25, 36.40), 'lon': (-86.62, -86.20)},
    'Rutherford': {'lat': (35.90, 36.10), 'lon': (-86.45, -86.15)},
    'Sumner': {'lat': (36.25, 36.45), 'lon': (-86.45, -86.10)},
    'Robertson': {'lat': (36.30, 36.50), 'lon': (-87.05, -86.60)},
}

# Major employment centers
EMPLOYMENT_CENTERS = {
    'Downtown': (36.160, -86.780, 185234),
    'West End': (36.135, -86.810, 67234),
    'Green Hills': (36.090, -86.760, 89567),
    'Brentwood': (35.985, -86.785, 76234),
    'Williamson Tech': (35.945, -86.515, 94567),
}


def create_county_map(counties_data: pd.DataFrame) -> plt.Figure:
    """Create map of Nashville MSA counties with population data."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Normalize population for color intensity
    pop_min, pop_max = counties_data['population_2020'].min(), counties_data['population_2020'].max()
    colors = plt.cm.RdYlGn((counties_data['population_2020'] - pop_min) / (pop_max - pop_min))
    
    # Plot counties as rectangles
    for idx, row in counties_data.iterrows():
        county = row['county']
        if county in COUNTY_BOUNDS:
            bounds = COUNTY_BOUNDS[county]
            lat_range = bounds['lat']
            lon_range = bounds['lon']
            
            rect = Rectangle(
                (lon_range[0], lat_range[0]),
                lon_range[1] - lon_range[0],
                lat_range[1] - lat_range[0],
                linewidth=2,
                edgecolor='black',
                facecolor=colors[idx],
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add county label and stats
            center_lat = (lat_range[0] + lat_range[1]) / 2
            center_lon = (lon_range[0] + lon_range[1]) / 2
            
            ax.text(center_lon, center_lat, f"{county}\n{row['population_2020']:,}", 
                   ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Add commuting info below
            ax.text(center_lon, center_lat - 0.04, 
                   f"Carpool: {row['percent_carpool']:.1f}% | Transit: {row['percent_transit']:.1f}%",
                   ha='center', va='top', fontsize=8, style='italic')
    
    ax.set_xlim(NASHVILLE_BOUNDS['min_lon'] - 0.1, NASHVILLE_BOUNDS['max_lon'] + 0.1)
    ax.set_ylim(NASHVILLE_BOUNDS['min_lat'] - 0.1, NASHVILLE_BOUNDS['max_lat'] + 0.1)
    ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
    ax.set_title('Nashville-Davidson MSA Counties\n2020 Census Population Data', 
                fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.RdYlGn,
        norm=plt.Normalize(vmin=pop_min, vmax=pop_max)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='2020 Population', pad=0.02)
    
    plt.tight_layout()
    return fig


def create_zip_code_heatmap(zip_data: pd.DataFrame) -> plt.Figure:
    """Create heatmap of Nashville ZIP codes with commute times and employment ratios."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Sort by mean commute time for better visualization
    zip_sorted = zip_data.sort_values('mean_commute_time')
    
    # Plot 1: Mean Commute Time
    colors_commute = plt.cm.RdYlGn_r(
        (zip_sorted['mean_commute_time'] - zip_sorted['mean_commute_time'].min()) / 
        (zip_sorted['mean_commute_time'].max() - zip_sorted['mean_commute_time'].min())
    )
    
    bars1 = axes[0].barh(range(len(zip_sorted)), zip_sorted['mean_commute_time'], color=colors_commute)
    axes[0].set_yticks(range(len(zip_sorted)))
    axes[0].set_yticklabels(zip_sorted['area'], fontsize=9)
    axes[0].set_xlabel('Mean Commute Time (minutes)', fontsize=11, fontweight='bold')
    axes[0].set_title('Mean Commute Time by ZIP Code Area\n(ACS 2016-2020)', 
                     fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Employment-Population Ratio
    colors_emp = plt.cm.YlGn(
        (zip_sorted['employment_population_ratio'] - zip_sorted['employment_population_ratio'].min()) / 
        (zip_sorted['employment_population_ratio'].max() - zip_sorted['employment_population_ratio'].min())
    )
    
    bars2 = axes[1].barh(range(len(zip_sorted)), zip_sorted['employment_population_ratio'], 
                         color=colors_emp)
    axes[1].set_yticks(range(len(zip_sorted)))
    axes[1].set_yticklabels(zip_sorted['area'], fontsize=9)
    axes[1].set_xlabel('Employment-Population Ratio', fontsize=11, fontweight='bold')
    axes[1].set_title('Employment-Population Ratio by ZIP Code\n(Identifies Job Centers)', 
                     fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='High Employment', alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    return fig


def create_commuting_zone_map(zones_data: pd.DataFrame) -> plt.Figure:
    """Create map visualization of commuting zones with employment and population data."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create a grid for zones visualization
    grid_cols, grid_rows = 4, 3
    zone_positions = [
        (0, 2), (1, 2), (2, 2), (3, 2),  # Row 0 (North)
        (0, 1), (1, 1), (2, 1), (3, 1),  # Row 1 (Middle)
        (0, 0), (1, 0), (2, 0), (3, 0),  # Row 2 (South)
    ]
    
    # Normalize data for visualization
    emp_norm = (zones_data['employment_count'] - zones_data['employment_count'].min()) / \
              (zones_data['employment_count'].max() - zones_data['employment_count'].min())
    
    pop_norm = (zones_data['resident_population'] - zones_data['resident_population'].min()) / \
              (zones_data['resident_population'].max() - zones_data['resident_population'].min())
    
    # Color code: Employment centers (red), Mixed (yellow), Residential (green)
    zone_colors = []
    for z_type in zones_data['zone_type']:
        if z_type == 'employment':
            zone_colors.append('#FF6B6B')
        elif z_type == 'mixed':
            zone_colors.append('#FFD93D')
        else:
            zone_colors.append('#6BCB77')
    
    for idx, (row, col) in enumerate(zone_positions[:len(zones_data)]):
        if idx < len(zones_data):
            x = col * 1.2
            y = row * 1.2
            
            zone = zones_data.iloc[idx]
            color = zone_colors[idx]
            
            # Draw zone box with size representing population
            size = 0.3 + (pop_norm.iloc[idx] * 0.4)
            rect = Rectangle((x - size/2, y - size/2), size, size,
                            linewidth=2, edgecolor='black', facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Add zone label
            ax.text(x, y + 0.25, zone['zone_name'], ha='center', va='bottom',
                   fontsize=9, fontweight='bold', wrap=True)
            
            # Add employment count (inner label)
            ax.text(x, y, f"{zone['employment_count']:,}\nemployees", 
                   ha='center', va='center', fontsize=8, style='italic')
            
            # Add population below
            ax.text(x, y - 0.25, f"{zone['resident_population']:,} residents", 
                   ha='center', va='top', fontsize=8)
    
    ax.set_xlim(-0.5, 4.8)
    ax.set_ylim(-0.5, 3.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', label='Employment Center'),
        mpatches.Patch(facecolor='#FFD93D', edgecolor='black', label='Mixed Use'),
        mpatches.Patch(facecolor='#6BCB77', edgecolor='black', label='Residential'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_title('Nashville Commuting Zones\n(Box size = population, Color = zone type)', 
                fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def create_commuting_flow_sankey(flows_data: pd.DataFrame, zones_data: pd.DataFrame) -> plt.Figure:
    """Create simplified commuting flow visualization."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get top flows
    top_flows = flows_data.nlargest(15, 'commuting_trips')
    
    # Create flow visualization
    y_positions = {}
    zone_names = zones_data['zone_name'].tolist()
    
    # Assign vertical positions
    for i, name in enumerate(zone_names):
        y_positions[name] = i * 1.2
    
    # Draw zones on left and right
    for name, y in y_positions.items():
        ax.scatter([0, 3], [y, y], s=200, c='#3498db', alpha=0.6, zorder=3)
        ax.text(-0.2, y, name, ha='right', va='center', fontsize=9, fontweight='bold')
        ax.text(3.2, y, name, ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Draw flow lines
    for _, flow in top_flows.iterrows():
        origin = flow['origin_name']
        dest = flow['destination_name']
        
        if origin in y_positions and dest in y_positions:
            y_orig = y_positions[origin]
            y_dest = y_positions[dest]
            
            # Line width proportional to trip volume
            width = max(1, flow['commuting_trips'] / 500)
            
            ax.plot([0.1, 2.9], [y_orig, y_dest], 'gray', linewidth=width, alpha=0.3, zorder=1)
            
            # Add trip count label
            mid_x = 1.5
            mid_y = (y_orig + y_dest) / 2
            ax.text(mid_x, mid_y, f"{flow['commuting_trips']:,.0f}", 
                   ha='center', va='bottom', fontsize=7, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlim(-1.5, 4)
    ax.set_ylim(-1, max(y_positions.values()) + 1)
    ax.axis('off')
    ax.set_title('Major Commuting Flows between Zones\n(2016-2020 ACS Estimates)', 
                fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def create_employment_centers_map(emp_data: pd.DataFrame) -> plt.Figure:
    """Create map of major employment centers with sector information."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Normalize employment size for bubble size
    emp_norm = (emp_data['employment'] - emp_data['employment'].min()) / \
              (emp_data['employment'].max() - emp_data['employment'].min())
    bubble_sizes = 100 + (emp_norm * 2000)
    
    # Map sector to color
    sector_colors = {
        'Mixed': '#FF6B6B',
        'Healthcare/Music': '#4ECDC4',
        'Corporate': '#45B7D1',
        'Technology': '#96CEB4',
        'Industrial': '#FFEAA7',
        'Logistics': '#DDA0DD',
        'Manufacturing': '#F4A460'
    }
    
    colors = [sector_colors.get(s, '#999999') for s in emp_data['sector']]
    
    # Create scatter plot with actual Nashville coordinates
    lons = []
    lats = []
    
    for _, emp in emp_data.iterrows():
        if emp['center_name'] in EMPLOYMENT_CENTERS:
            lat, lon, _ = EMPLOYMENT_CENTERS[emp['center_name']]
            lats.append(lat)
            lons.append(lon)
        else:
            # Default to random location in Nashville bounds
            lons.append(np.random.uniform(NASHVILLE_BOUNDS['min_lon'], NASHVILLE_BOUNDS['max_lon']))
            lats.append(np.random.uniform(NASHVILLE_BOUNDS['min_lat'], NASHVILLE_BOUNDS['max_lat']))
    
    scatter = ax.scatter(lons, lats, s=bubble_sizes, c=colors, alpha=0.6, 
                        edgecolors='black', linewidth=2, zorder=3)
    
    # Add labels
    for idx, (lon, lat) in enumerate(zip(lons, lats)):
        emp = emp_data.iloc[idx]
        ax.text(lon, lat, emp['center_name'].split()[0], 
               ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(lon, lat - 0.015, f"{emp['employment']:,}", 
               ha='center', va='top', fontsize=8, style='italic')
    
    ax.set_xlim(NASHVILLE_BOUNDS['min_lon'] - 0.1, NASHVILLE_BOUNDS['max_lon'] + 0.1)
    ax.set_ylim(NASHVILLE_BOUNDS['min_lat'] - 0.1, NASHVILLE_BOUNDS['max_lat'] + 0.1)
    ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', label=sector)
                      for sector, color in sector_colors.items()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    ax.set_title('Major Employment Centers in Nashville MSA\n(Bubble size = Employment count)', 
                fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def create_mode_share_chart(mode_share_data: pd.DataFrame) -> plt.Figure:
    """Create commute mode share comparison by county."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Get mode columns (exclude county)
    mode_cols = [col for col in mode_share_data.columns if col != 'county' and '_pct' in col]
    mode_labels = [col.replace('_pct', '').replace('_', ' ').title() for col in mode_cols]
    
    # Plot stacked bar chart
    ax_main = axes[0, :].flatten()[0]
    
    counties = mode_share_data['county'].tolist()
    mode_data = mode_share_data[[col for col in mode_cols]].values
    
    x = np.arange(len(counties))
    width = 0.6
    bottom = np.zeros(len(counties))
    
    colors_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for i, (mode, label) in enumerate(zip(mode_cols, mode_labels)):
        values = mode_share_data[mode].values
        ax_main.bar(x, values, width, label=label, bottom=bottom, color=colors_palette[i])
        bottom += values
    
    ax_main.set_xlabel('County', fontsize=11, fontweight='bold')
    ax_main.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax_main.set_title('Commute Mode Share by County\n(2016-2020 ACS 5-Year Estimates)', 
                     fontsize=12, fontweight='bold')
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(counties, rotation=45, ha='right')
    ax_main.legend(loc='upper right', fontsize=9)
    ax_main.grid(axis='y', alpha=0.3)
    
    # Individual mode comparisons
    for idx, (mode, label) in enumerate(zip(mode_cols[:-1], mode_labels[:-1])):
        ax = axes.flatten()[idx + 1]
        values = mode_share_data[mode].values
        bars = ax.bar(counties, values, color=colors_palette[idx], alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xticklabels(counties, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, values.max() * 1.15)
    
    plt.tight_layout()
    return fig


def create_incentive_impact_summary(incentive_data: pd.DataFrame) -> plt.Figure:
    """Create summary of incentive impact potential by zone."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Sort by potential impact
    sorted_data = incentive_data.sort_values('vmt_reduction_potential_pct', ascending=True)
    
    # Plot 1: Carpool uplift potential
    colors_cp = plt.cm.Blues(
        (sorted_data['potential_carpool_uplift_pct'] - sorted_data['potential_carpool_uplift_pct'].min()) /
        (sorted_data['potential_carpool_uplift_pct'].max() - sorted_data['potential_carpool_uplift_pct'].min())
    )
    axes[0].barh(range(len(sorted_data)), sorted_data['potential_carpool_uplift_pct'], color=colors_cp)
    axes[0].set_yticks(range(len(sorted_data)))
    axes[0].set_yticklabels(sorted_data['zone_name'], fontsize=9)
    axes[0].set_xlabel('Carpool Uplift Potential (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Carpool Incentive Impact\nby Zone', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Transit uplift potential
    colors_tr = plt.cm.Greens(
        (sorted_data['potential_transit_uplift_pct'] - sorted_data['potential_transit_uplift_pct'].min()) /
        (sorted_data['potential_transit_uplift_pct'].max() - sorted_data['potential_transit_uplift_pct'].min())
    )
    axes[1].barh(range(len(sorted_data)), sorted_data['potential_transit_uplift_pct'], color=colors_tr)
    axes[1].set_yticks(range(len(sorted_data)))
    axes[1].set_yticklabels(sorted_data['zone_name'], fontsize=9)
    axes[1].set_xlabel('Transit Uplift Potential (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Transit Incentive Impact\nby Zone', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Plot 3: Overall VMT reduction potential
    colors_vmt = plt.cm.Reds(
        (sorted_data['vmt_reduction_potential_pct'] - sorted_data['vmt_reduction_potential_pct'].min()) /
        (sorted_data['vmt_reduction_potential_pct'].max() - sorted_data['vmt_reduction_potential_pct'].min())
    )
    bars = axes[2].barh(range(len(sorted_data)), sorted_data['vmt_reduction_potential_pct'], color=colors_vmt)
    axes[2].set_yticks(range(len(sorted_data)))
    axes[2].set_yticklabels(sorted_data['zone_name'], fontsize=9)
    axes[2].set_xlabel('VMT Reduction Potential (%)', fontsize=11, fontweight='bold')
    axes[2].set_title('Overall VMT Reduction Impact\nby Zone', fontsize=12, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (ax, col) in enumerate([(axes[0], 'potential_carpool_uplift_pct'),
                                     (axes[1], 'potential_transit_uplift_pct'),
                                     (axes[2], 'vmt_reduction_potential_pct')]):
        for j, v in enumerate(sorted_data[col].values):
            ax.text(v + 0.1, j, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig
