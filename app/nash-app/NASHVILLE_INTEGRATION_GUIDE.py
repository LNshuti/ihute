"""
NASHVILLE TRANSPORTATION SIMULATION - INTEGRATION GUIDE

This guide explains how to integrate the Nashville Transportation Simulation tab
into your existing Gradio dashboard for the IHUTE (Incentive Heterogeneous Urban 
Transportation Equilibrium) project.

=============================================================================
FILE STRUCTURE
=============================================================================

Your project structure should be:

```
ihute/
├── components/
│   ├── __init__.py
│   ├── traffic_flow.py
│   ├── incentive_analytics.py
│   ├── behavioral_calibration.py
│   ├── simulation_comparison.py
│   ├── real_time_metrics.py
│   └── geo_map.py
├── nashville_sim_data.py          [NEW]
├── nashville_sim_components.py     [NEW]
├── nashville_sim_integration.py    [NEW]
├── main_dashboard.py               [MODIFIED]
├── requirements.txt                [UPDATE]
└── README.md
```

=============================================================================
STEP 1: INSTALL DEPENDENCIES
=============================================================================

Add these packages to your requirements.txt:

    gradio>=4.0.0
    pandas>=1.5.0
    numpy>=1.23.0
    matplotlib>=3.6.0
    geopandas>=0.13.0
    shapely>=2.0.0
    folium>=0.14.0

Then install:
    pip install -r requirements.txt

=============================================================================
STEP 2: ADD NEW FILES TO YOUR PROJECT
=============================================================================

Copy these three new files to your project directory:
1. nashville_sim_data.py
2. nashville_sim_components.py  
3. nashville_sim_integration.py

=============================================================================
STEP 3: UPDATE YOUR MAIN DASHBOARD FILE
=============================================================================

Modify your existing dashboard file (e.g., main_dashboard.py or the file
that creates the Gradio app) to import and use the new Nashville simulation tab.

Here's the updated create_app() function:

```python
import gradio as gr

# Existing imports
from components import (
    # Traffic flow
    get_speed_heatmap_data, create_speed_heatmap,
    # ... other existing imports ...
)

# NEW IMPORTS for Nashville simulation
from nashville_sim_integration import create_nashville_simulation_tab


def create_traffic_tab():
    # ... existing code ...
    pass


def create_incentive_tab():
    # ... existing code ...
    pass


def create_behavioral_tab():
    # ... existing code ...
    pass


def create_simulation_tab():
    # ... existing code ...
    pass


def create_metrics_tab():
    # ... existing code ...
    pass


def create_map_tab():
    # ... existing code ...
    pass


def create_app():
    \"\"\"Create the main Gradio application with all tabs.\"\"\"
    with gr.Blocks(title="Nashville Incentive Simulation Dashboard") as app:
        gr.Markdown(
            \"\"\"
            # Nashville Transportation Incentive Simulation

            Comprehensive dashboard for analyzing traffic flow, incentive effectiveness,
            behavioral calibration, and transportation simulation scenarios.
            \"\"\"
        )

        with gr.Tabs():
            
            # Existing tabs
            with gr.TabItem("Incentive Analytics"):
                create_incentive_tab()

            with gr.TabItem("Behavioral Calibration"):
                create_behavioral_tab()

            with gr.TabItem("Simulation Comparison"):
                create_simulation_tab()

            with gr.TabItem("Live Metrics"):
                create_metrics_tab()

            with gr.TabItem("Corridor Map"):
                create_map_tab()

            # NEW TAB: Nashville Transportation Simulation
            with gr.TabItem("Nashville Simulation"):
                create_nashville_simulation_tab()

        gr.Markdown(
            \"\"\"
            ---
            *Data sources: 2020 Census DHC, ACS 2016-2020, LADDMS I-24 MOTION, 
            Hytch rideshare trips, simulation outputs*

            Built with DuckDB, dbt, and Gradio | [GitHub](https://github.com/LNshuti/ihute)
            \"\"\"
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
```

=============================================================================
STEP 4: RUN YOUR UPDATED DASHBOARD
=============================================================================

Run your dashboard with:

    python main_dashboard.py

Or if using Hugging Face Spaces:

    python main_dashboard.py

The new "Nashville Simulation" tab should now appear alongside your existing tabs.

=============================================================================
DATA COMPONENTS OVERVIEW
=============================================================================

The Nashville Simulation tab includes 8 sub-tabs:

1. **Geographic Overview**
   - County-level population and commuting statistics
   - 2020 Census data
   - Carpool and transit percentages by county

2. **ZIP Code Analysis**
   - 30 Nashville-area ZIP codes analyzed
   - Mean commute times (ACS data)
   - Employment-population ratios
   - Work-from-home rates

3. **Commuting Zones**
   - 12 commuting zones identified from ACS journey-to-work data
   - Color-coded by zone type: Employment, Mixed, Residential
   - Employment counts and resident population

4. **Employment Centers**
   - 9 major employment centers mapped
   - Bubble size = employment count
   - Current carpool and transit mode shares

5. **Commuting Flows**
   - Origin-destination matrix between zones
   - Based on ACS 2016-2020 5-year estimates
   - Top 10 commuting corridors displayed

6. **Mode Share Analysis**
   - County-by-county breakdown of commute modes
   - Drove alone, carpool, transit, walk/bike
   - Work-from-home rates

7. **Incentive Impact Potential**
   - Estimated carpool uplift by zone
   - Transit adoption potential
   - Overall VMT reduction potential

8. **Data Summary & Sources**
   - Complete documentation
   - Data source citations
   - Methodology explanation
   - Quick statistics

=============================================================================
DATA SOURCES & CITATIONS
=============================================================================

The Nashville Transportation Simulation uses authoritative government and
census data:

**2020 Census Demographic and Housing Characteristics File (DHC)**
- U.S. Census Bureau
- Total population, housing units, demographic characteristics

**American Community Survey (ACS) 2016-2020 5-Year Estimates**
- U.S. Census Bureau
- Journey-to-work commuting data
- Mode of transportation to work
- Commute time statistics
- Work-from-home percentages
- Source: https://data.census.gov/

**LODES (Longitudinal Employer-Household Dynamics)**
- U.S. Census Bureau
- Employment center locations

=============================================================================
CUSTOMIZING THE SIMULATION
=============================================================================

To customize the simulation data:

1. **Modify Commuting Zones**: Edit NashvilleTransportationData.get_commuting_zones()
2. **Update Employment Centers**: Edit NashvilleTransportationData.get_employment_centers()
3. **Adjust Incentive Impacts**: Edit NashvilleTransportationData.get_incentive_impact_potential()
4. **Change Commuting Flows**: Edit NashvilleTransportationData.get_commuting_flows()

All data can be based on actual Census/ACS downloads from data.census.gov.

=============================================================================
PERFORMANCE OPTIMIZATION
=============================================================================

For production deployments:

1. **Cache data loading**:
```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_all_nashville_data():
    return load_all_nashville_data()
```

2. **Pre-generate visualizations**:
   - Generate all plots once on startup
   - Store as PNG files
   - Serve from disk rather than regenerating

3. **Use Gradio caching**:
```python
@gr.cache_examples
def expensive_visualization():
    # Pre-computed visualization
    pass
```

=============================================================================
TROUBLESHOOTING
=============================================================================

**Issue**: ModuleNotFoundError: No module named 'nashville_sim_data'
**Solution**: Make sure all three .py files are in the same directory as main_dashboard.py

**Issue**: Visualization looks distorted
**Solution**: Check that matplotlib is properly configured and figures are using consistent DPI settings

**Issue**: Large memory usage
**Solution**: The data is relatively lightweight (~50MB), but if using on limited resources,
consider sampling the data or using data aggregation.

=============================================================================
EXTENDING THE SIMULATION
=============================================================================

To add real geographic maps with actual boundary data:

1. Download shapefiles for Tennessee counties from: https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-files.html

2. Use geopandas and folium for interactive maps:

```python
import geopandas as gpd
import folium
from folium import GeoJson

# Load shapefile
gdf = gpd.read_file('path/to/counties.shp')

# Create folium map
m = folium.Map(location=[36.16, -86.78], zoom_start=9)

# Add GeoJSON layer
for idx, row in gdf.iterrows():
    GeoJson(row['geometry']).add_to(m)

# Convert to Gradio Plot or HTML
```

=============================================================================
NEXT STEPS
=============================================================================

1. Integrate the new files into your existing project
2. Test the dashboard locally
3. Add real Census/ACS data exports if needed
4. Deploy to Hugging Face Spaces or other platform
5. Collect user feedback and iterate

For questions or issues with the IHUTE project, visit:
https://github.com/LNshuti/ihute

=============================================================================
"""

if __name__ == "__main__":
    print(__doc__)
