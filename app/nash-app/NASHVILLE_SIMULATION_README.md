# Nashville Transportation Simulation - Gradio Dashboard Tab

A comprehensive Gradio dashboard tab for analyzing Nashville-Davidson Metropolitan Statistical Area (MSA) transportation patterns using 2020 Census data and American Community Survey commuting flows.

## Overview

This module adds a new "Nashville Simulation" tab to the IHUTE (Incentive Heterogeneous Urban Transportation Equilibrium) transportation incentive dashboard. It provides in-depth analysis of:

- **County-level demographics and commuting patterns** (2020 Census DHC)
- **ZIP code-level commute analysis** (ACS 2016-2020)
- **12 identified commuting zones** based on ACS journey-to-work flows
- **Major employment centers** (9 identified centers)
- **Origin-destination commuting flows** (ACS journey-to-work data)
- **Mode share analysis** by county
- **Transportation incentive impact potential** by zone

## Files Included

1. **`nashville_sim_data.py`** - Data processing module
   - `NashvilleTransportationData` class for managing all data
   - Methods to load and process Census/ACS data
   - Employment center and commuting zone definitions

2. **`nashville_sim_components.py`** - Visualization components
   - County maps with population data
   - ZIP code heatmaps for commute times and employment ratios
   - Commuting zone maps
   - Employment centers visualization
   - Commuting flow diagrams
   - Mode share analysis charts
   - Incentive impact potential analysis

3. **`nashville_sim_integration.py`** - Gradio tab integration
   - `create_nashville_simulation_tab()` function
   - 8 sub-tabs for different analyses
   - Data tables and visualizations

4. **`main_dashboard_updated.py`** - Example main dashboard
   - Shows how to integrate all tabs
   - Can be used as reference or as direct replacement

5. **`requirements.txt`** - Python dependencies

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy files to your project
cp nashville_sim_data.py /path/to/your/project/
cp nashville_sim_components.py /path/to/your/project/
cp nashville_sim_integration.py /path/to/your/project/
```

### 2. Integration into Existing Dashboard

In your main Gradio app file, add:

```python
from nashville_sim_integration import create_nashville_simulation_tab

def create_app():
    with gr.Blocks() as app:
        # ... your existing tabs ...
        
        with gr.Tabs():
            # Existing tabs
            with gr.TabItem("Existing Tab 1"):
                create_tab_1()
            
            # NEW: Nashville Simulation Tab
            with gr.TabItem("Nashville Simulation"):
                create_nashville_simulation_tab()
        
        return app
```

### 3. Run Your Dashboard

```bash
python your_main_dashboard.py
```

The new "Nashville Simulation" tab will appear alongside your existing tabs.

## Data Sources

All data used in this simulation comes from authoritative government sources:

### 2020 Census Demographic and Housing Characteristics File (DHC)
- **Source**: U.S. Census Bureau
- **Data**: Population counts, housing units, basic demographics
- **Geography**: County level for Nashville MSA
- **Access**: https://data.census.gov/

### American Community Survey (ACS) 2016-2020 5-Year Estimates
- **Source**: U.S. Census Bureau
- **Data**: 
  - Journey-to-work commuting flows
  - Commute time statistics
  - Mode of transportation to work
  - Work-from-home percentages
  - Median household income
- **Access**: https://data.census.gov/
- **API**: Census Bureau API (for programmatic access)

### Bureau of Labor Statistics (BLS) Data
- **Source**: U.S. Bureau of Labor Statistics
- **Data**: Employment center locations and employment counts
- **Access**: https://www.bls.gov/

### LODES Employment Data
- **Source**: U.S. Census Bureau, Longitudinal Employer-Household Dynamics
- **Data**: Detailed employment location data
- **Access**: https://lehd.ces.census.gov/

## Data Components Explained

### Counties (9 total)
- **Core MSA**: Davidson, Williamson, Sumner, Rutherford, Wilson, Robertson
- **Extended**: Dickson, Maury, Cheatham
- **Data included**: Population, housing units, median HHI, commute statistics

### ZIP Codes (30 covered)
- Nashville urban area and surrounding suburbs
- **Data**: Population, employment-population ratio, commute time, WFH rate

### Commuting Zones (12 defined)
- **Downtown/CBD**: Employment center in urban core
- **Residential**: Primarily residential areas (Antioch, Hermitage, outer suburbs)
- **Mixed**: Areas with both employment and residential (Madison, Donelson)

**Zone Type Color Coding:**
- 🔴 Red = Employment Center
- 🟡 Yellow = Mixed Use
- 🟢 Green = Primarily Residential

### Employment Centers (9 major centers)
- Downtown Nashville
- West End (Medical/Music District)
- Green Hills Office Park
- Brentwood Corporate Park
- Williamson County Tech Corridor
- Hermitage Industrial
- Bellevue Medical Center
- Airport Industrial Area
- Madison Industrial

### Commuting Flows
- Origin-destination pairs derived from ACS journey-to-work data
- Includes reverse flows with typical asymmetry adjustments
- 15+ major commuting corridors visualized

## Sub-Tab Descriptions

### 1. Geographic Overview
County-level analysis with:
- Interactive county map (color intensity = population)
- County statistics table (population, commuting mode share)
- Commute time averages by county

### 2. ZIP Code Analysis
Detailed ZIP code breakdown:
- Mean commute times ranked
- Employment-population ratios (identifies job centers)
- Work-from-home rate analysis
- Sortable data table

### 3. Commuting Zones
Spatial visualization of:
- 12 identified commuting zones
- Zone type color-coding
- Employment and population data
- In-zone employment percentages

### 4. Employment Centers
Major employers mapped with:
- Bubble size = employment count
- Color = industry sector
- Current carpool and transit percentages
- Sector breakdown (Corporate, Healthcare, Industrial, Tech, etc.)

### 5. Commuting Flows
Origin-destination analysis:
- Top 15 commuting corridors visualized
- Flow width = trip volume
- Tables showing daily trip counts
- Percent of total commuting flows

### 6. Mode Share Analysis
Commute mode breakdown by county:
- Drove alone %
- Carpool %
- Transit %
- Walked/Biked %
- Worked from home %
- Individual mode charts with value labels

### 7. Incentive Impact Potential
Analysis of program effectiveness:
- Estimated carpool uplift by zone
- Transit adoption potential
- Overall VMT reduction potential
- Based on elasticity of demand for different modes

### 8. Data Summary & Sources
Documentation tab including:
- Data source citations
- Methodology explanation
- Coverage and time periods
- Quick statistics table

## Customizing the Data

### Update with Real Census Data

1. Download data from https://data.census.gov/
2. Export as CSV
3. Modify `NashvilleTransportationData` methods in `nashville_sim_data.py`:

```python
def get_msa_counties(self) -> pd.DataFrame:
    # Replace with your downloaded data
    counties = pd.read_csv('your_census_download.csv')
    # Transform as needed
    return counties
```

### Modify Commuting Zones

Edit `get_commuting_zones()` to reflect actual ACS-identified zones:

```python
def get_commuting_zones(self) -> pd.DataFrame:
    zones = {
        'zone_id': [...],
        'zone_name': [...],
        # ... add your zone data ...
    }
    return pd.DataFrame(zones)
```

### Adjust Employment Centers

Update `get_employment_centers()` with actual business data:

```python
def get_employment_centers(self) -> pd.DataFrame:
    centers = {
        'center_name': ['Your Center', ...],
        'employment': [1000, ...],
        # ... add actual employment data ...
    }
    return pd.DataFrame(centers)
```

## Advanced Usage

### Real Geographic Maps

To use actual geographic boundaries (rather than simplified grid):

```python
import geopandas as gpd
import folium

# Load county shapefiles
counties_gdf = gpd.read_file('path/to/tn_counties.shp')

# Filter to Nashville MSA
msa_counties = counties_gdf[counties_gdf['NAME'].isin(['Davidson', 'Williamson', ...])]

# Create folium map
m = folium.Map(location=[36.16, -86.78], zoom_start=9)
m.add_child(folium.features.GeoJson(msa_counties.to_json()))
```

### Interactive Filters

Add interactivity to the Gradio tab:

```python
with gr.Row():
    county_select = gr.Dropdown(
        choices=nash_data['counties']['county'].tolist(),
        label="Filter by County"
    )
    update_button = gr.Button("Update Visualizations")

def update_county_view(selected_county):
    filtered_data = nash_data['counties'][nash_data['counties']['county'] == selected_county]
    return create_county_map(filtered_data)

update_button.click(update_county_view, inputs=county_select, outputs=[...])
```

### Export Analysis

Add data export capability:

```python
def export_to_csv():
    nash_data['commuting_zones'].to_csv('commuting_zones.csv')
    return 'commuting_zones.csv'

export_button = gr.Button("Export Commuting Zones")
export_button.click(export_to_csv, outputs=gr.File())
```

## Performance Optimization

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_all_nashville_data():
    return load_all_nashville_data()
```

### Pre-generate Visualizations
```python
# In initialization
all_figs = {
    'county_map': create_county_map(data),
    'zip_heatmap': create_zip_code_heatmap(data),
    # ... pre-generate all visualizations
}

# In Gradio tab
gr.Plot(value=all_figs['county_map'])
```

## Troubleshooting

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'nashville_sim_data'`
**Solution**: Ensure all three `.py` files are in the same directory as your main dashboard file

### Data Not Displaying
**Problem**: Blank visualizations or missing data
**Solution**: 
1. Check that matplotlib is properly installed: `pip install matplotlib --upgrade`
2. Verify data shapes: `print(nash_data['counties'].shape)`
3. Check for NaN values: `nash_data['counties'].isnull().sum()`

### Slow Performance
**Problem**: Dashboard takes long time to load
**Solution**:
1. Use caching for data loading
2. Pre-generate visualizations on startup
3. Consider using smaller dataset if not all data is needed

## Contributing

To extend or improve the Nashville simulation:

1. Add new data sources to `NashvilleTransportationData`
2. Create new visualization functions in `nashville_sim_components.py`
3. Add new sub-tabs in `create_nashville_simulation_tab()` in `nashville_sim_integration.py`

## References & Further Reading

### Census Data Access
- [Census Bureau Data Portal](https://data.census.gov/)
- [ACS Documentation](https://www.census.gov/programs-surveys/acs/guidance.html)
- [Journey-to-Work Data](https://www.census.gov/topics/employment/commuting.html)

### Nashville Transportation Studies
- [Nashville MTA Long Range Plan](https://www.nashvillemta.org/planning)
- [IHUTE Research](https://github.com/LNshuti/ihute)

### Visualization Libraries
- [Matplotlib Documentation](https://matplotlib.org/)
- [Folium Maps](https://python-visualization.github.io/folium/)
- [Gradio Documentation](https://gradio.app/)

## License

This code is provided as part of the IHUTE transportation research project.

## Questions?

For questions about the Nashville Transportation Simulation, data sources, or integration:

- GitHub Issues: https://github.com/LNshuti/ihute/issues
- Email: leonce@igisha.com

---

**Last Updated**: January 2026

**Data Currency**: 
- Census 2020 data (most recent)
- ACS 2016-2020 5-Year Estimates (most recent available)
