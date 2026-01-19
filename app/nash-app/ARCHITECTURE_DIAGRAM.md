# Nashville Transportation Simulation - Architecture & Data Flow

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GRADIO WEB DASHBOARD                            │
│  http://localhost:7860                                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│              main_dashboard.py (or your main file)                 │
│                                                                     │
│  create_app():                                                      │
│  ├── Tab: Incentive Analytics      (existing)                      │
│  ├── Tab: Behavioral Calibration   (existing)                      │
│  ├── Tab: Simulation Comparison    (existing)                      │
│  ├── Tab: Live Metrics            (existing)                      │
│  ├── Tab: Corridor Map            (existing)                      │
│  └── Tab: Nashville Simulation     (NEW) ◄──────┐                 │
│                                              │                 │
│      with gr.TabItem("Nashville Simulation"):│                 │
│          create_nashville_simulation_tab()  │                 │
└──────────────────────────────────────────────┼─────────────────┼──┘
                                              │                 │
                    ┌─────────────────────────┴─────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│        nashville_sim_integration.py                                │
│                                                                     │
│  create_nashville_simulation_tab():                                │
│  ├── Tab 1: Geographic Overview       ──┐                         │
│  ├── Tab 2: ZIP Code Analysis         ──┤                         │
│  ├── Tab 3: Commuting Zones           ──┤ Uses Data from:        │
│  ├── Tab 4: Employment Centers        ──┤ nashville_sim_data.py   │
│  ├── Tab 5: Commuting Flows           ──┤                         │
│  ├── Tab 6: Mode Share Analysis       ──┤ Creates Visuals from:   │
│  ├── Tab 7: Incentive Impact          ──┤ nashville_sim_components│
│  └── Tab 8: Data Summary & Sources    ──┘                         │
└──────┬──────────────────────────────────────────────────────┬──────┘
       │                                                      │
       ├──────────────────────┬───────────────────────────────┤
       ↓                      ↓                               ↓
┌─────────────────┐  ┌──────────────────┐          ┌─────────────────┐
│nashville_sim_   │  │nashville_sim_    │          │  Your Existing  │
│   data.py       │  │ components.py    │          │ component/ dir  │
└─────────────────┘  └──────────────────┘          └─────────────────┘
       │                      │                             │
       ├─ Data Classes        ├─ Visualization Funcs        ├─ Existing
       │  (data loading)      │  (creates matplotlib        │   traffic
       │                      │   figures)                  │   analysis
       ├─ Counties (9)        │                             │
       ├─ ZIP Codes (30)      ├─ create_county_map()       │
       ├─ Zones (12)          ├─ create_zip_heatmap()      │
       ├─ Employment (9)      ├─ create_zone_map()         │
       ├─ Flows (O-D pairs)   ├─ create_flow_sankey()      │
       ├─ Mode Share          ├─ create_emp_map()          │
       ├─ Traffic Patterns    ├─ create_mode_chart()       │
       └─ Incentive Impact    └─ create_incentive_summary()│
                                                            │
       ↓──────────────────────┬──────────────────────────┬──┘
       │                      │                          │
       └── DATA INPUTS────────┴──VISUALIZATION OUTPUTS──┴──

```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                                │
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────────────┐   │
│  │  2020 Census DHC     │  │  ACS 2016-2020 5-Year        │   │
│  │  ────────────────    │  │  ───────────────────         │   │
│  │  • Population        │  │  • Commuting flows          │   │
│  │  • Housing units     │  │  • Mode share               │   │
│  │  • Demographics      │  │  • Commute times            │   │
│  │  • Median HHI        │  │  • Work-from-home %         │   │
│  └──────┬───────────────┘  └──────────────┬───────────────┘   │
│         │                                 │                   │
│         └─────────────────┬────────────────┘                   │
└─────────────────────────────┼──────────────────────────────────┘
                              │
                              ↓
        ┌─────────────────────────────────────────────────┐
        │   nashville_sim_data.py                         │
        │   ─────────────────────────                     │
        │   NashvilleTransportationData class             │
        └─────────────────────┬───────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ↓             ↓             ↓
            ┌────────┐  ┌─────────┐  ┌──────────┐
            │Counties│  │ZIP Codes│  │  Zones   │
            └────────┘  └─────────┘  └──────────┘
                ↓             ↓             ↓
            ┌────────┐  ┌─────────┐  ┌──────────┐
            │Flows   │  │Employment│  │Mode Share│
            └────────┘  └─────────┘  └──────────┘
                │             │             │
                └─────────────┼─────────────┘
                              │
                              ↓
        ┌─────────────────────────────────────────────────┐
        │   nashville_sim_components.py                   │
        │   ─────────────────────────────────────────────  │
        │   Visualization Functions                       │
        │   (input: data, output: matplotlib figures)     │
        └─────────────────────┬───────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
    ┌─────────┐          ┌────────┐          ┌──────────┐
    │  Maps   │          │ Charts │          │  Flows   │
    │         │          │        │          │  Diagram │
    │ County  │          │ Zip    │          │          │
    │ Zones   │          │ Heatmap│          │ O-D      │
    │ Emp     │          │ Mode   │          │ Matrix   │
    │ Centers │          │ Share  │          │          │
    └─────────┘          └────────┘          └──────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ↓
        ┌─────────────────────────────────────────────────┐
        │   nashville_sim_integration.py                  │
        │   ─────────────────────────────────────────────  │
        │   Gradio Tab with 8 Sub-tabs                    │
        │   (combines data + visualizations)              │
        └─────────────────────┬───────────────────────────┘
                              │
                              ↓
        ┌─────────────────────────────────────────────────┐
        │   Gradio Dashboard Browser Interface            │
        │   ─────────────────────────────────────────────  │
        │   Interactive visualizations                    │
        │   Data tables with sorting                      │
        │   Summary statistics                            │
        └─────────────────────────────────────────────────┘

```

## Module Interaction Matrix

```
                    nashville_sim_   nashville_sim_   nashville_sim_
                    data.py          components.py    integration.py
                    ───────────────  ────────────────  ───────────────

nashville_sim_data.py
  ├─ Imported by:   (ROOT)           YES              YES
  ├─ Imports:       pandas, numpy    (none)           (none)
  └─ Exports:       Data classes     (N/A)            (N/A)

nashville_sim_
components.py
  ├─ Imported by:   (via direct)      (ROOT)           YES
  ├─ Imports:       (none)            matplotlib      nashville_sim_data
  │                                   numpy           nashville_sim_
  │                                   pandas          components
  └─ Exports:       (N/A)             Plot functions  (N/A)

nashville_sim_
integration.py
  ├─ Imported by:   (via direct)      (via direct)     main_dashboard.py
  ├─ Imports:       (none)            gradio           nashville_sim_data
  │                                   (none)          nashville_sim_
  │                                                   components
  └─ Exports:       (N/A)             (N/A)            Gradio function

main_dashboard.py
  ├─ Imports:       (direct)          (via comp)       (direct)         your_
  │                                                                      components
  ├─ Creates:       (N/A)             (N/A)            Gradio app
  └─ Result:        Web interface with all tabs
```

## Data Processing Pipeline

```
Input Data (Census/ACS)
    │
    ├─ Raw CSV/JSON files
    │
    ↓
NashvilleTransportationData Class Methods
    │
    ├─ get_msa_counties()
    │   ├─ Clean population data
    │   ├─ Calculate percentages
    │   └─ Output: DataFrame (9 rows)
    │
    ├─ get_nashville_zip_codes()
    │   ├─ Parse ZIP code data
    │   ├─ Calculate employment ratios
    │   └─ Output: DataFrame (30 rows)
    │
    ├─ get_commuting_zones()
    │   ├─ Aggregate by zone
    │   ├─ Classify zone type
    │   └─ Output: DataFrame (12 rows)
    │
    ├─ get_commuting_flows()
    │   ├─ Build O-D matrix
    │   ├─ Calculate percentages
    │   └─ Output: DataFrame (40+ rows)
    │
    ├─ get_employment_centers()
    │   ├─ Identify major employers
    │   ├─ Assign coordinates
    │   └─ Output: DataFrame (9 rows)
    │
    ├─ get_commute_mode_share()
    │   ├─ Aggregate by county
    │   ├─ Normalize percentages
    │   └─ Output: DataFrame (6 rows)
    │
    ├─ get_traffic_patterns()
    │   ├─ Generate hourly patterns
    │   ├─ Model peak periods
    │   └─ Output: Dict with lists
    │
    └─ get_incentive_impact_potential()
        ├─ Estimate elasticities
        ├─ Calculate uplifts
        └─ Output: DataFrame (12 rows)
            │
            ↓
        Visualization Functions (components.py)
            │
            ├─ create_county_map() → Figure
            ├─ create_zip_code_heatmap() → Figure
            ├─ create_commuting_zone_map() → Figure
            ├─ create_commuting_flow_sankey() → Figure
            ├─ create_employment_centers_map() → Figure
            ├─ create_mode_share_chart() → Figure
            └─ create_incentive_impact_summary() → Figure
                │
                ↓
            Gradio Tab (integration.py)
                │
                ├─ Sub-tab 1: Geographic Overview
                ├─ Sub-tab 2: ZIP Code Analysis
                ├─ Sub-tab 3: Commuting Zones
                ├─ Sub-tab 4: Employment Centers
                ├─ Sub-tab 5: Commuting Flows
                ├─ Sub-tab 6: Mode Share
                ├─ Sub-tab 7: Incentive Impact
                └─ Sub-tab 8: Data Summary
                    │
                    ↓
                Web Dashboard
                (accessible via browser)
```

## Class Structure

```
NashvilleTransportationData
├─ __init__()
│   ├─ self.counties = None
│   ├─ self.zip_codes = None
│   ├─ self.commuting_zones = None
│   ├─ self.tract_data = None
│   ├─ self.commuting_flows = None
│   └─ self.employment_centers = None
│
├─ get_msa_counties() → DataFrame
│   └─ Returns: county, fips, population_2020, housing_units, etc.
│
├─ get_nashville_zip_codes() → DataFrame
│   └─ Returns: zip_code, area, population, employment_population_ratio, etc.
│
├─ get_commuting_zones() → DataFrame
│   └─ Returns: zone_id, zone_name, zone_type, employment_count, etc.
│
├─ get_commuting_flows() → DataFrame
│   └─ Returns: origin_zone, destination_zone, commuting_trips, etc.
│
├─ get_employment_centers() → DataFrame
│   └─ Returns: center_id, center_name, sector, employment, etc.
│
├─ get_commute_mode_share() → DataFrame
│   └─ Returns: county, drove_alone_pct, carpool_pct, transit_pct, etc.
│
├─ get_traffic_patterns() → Dict
│   └─ Returns: {hours: [...], hourly_volume: [...], hourly_speed: [...]}
│
└─ get_incentive_impact_potential() → DataFrame
    └─ Returns: zone_id, carpool_uplift_pct, transit_uplift_pct, vmt_reduction_potential_pct

```

## Visualization Component Details

```
create_county_map(counties_data: DataFrame) → Figure
├─ Inputs: counties DataFrame with population
├─ Processing:
│   ├─ Normalize population for colors
│   ├─ Draw rectangles for each county
│   ├─ Add labels and statistics
│   └─ Add colorbar
└─ Output: matplotlib.figure.Figure

create_zip_code_heatmap(zip_data: DataFrame) → Figure
├─ Inputs: ZIP codes DataFrame
├─ Processing:
│   ├─ Create 2-panel figure
│   ├─ Left: Mean commute time bars
│   ├─ Right: Employment-population ratio bars
│   └─ Color-code by values
└─ Output: matplotlib.figure.Figure

create_commuting_zone_map(zones_data: DataFrame) → Figure
├─ Inputs: zones DataFrame
├─ Processing:
│   ├─ Create 4×3 grid
│   ├─ Draw zone boxes (size = population)
│   ├─ Color by zone type (red/yellow/green)
│   └─ Add labels and legend
└─ Output: matplotlib.figure.Figure

create_commuting_flow_sankey(flows_data, zones_data) → Figure
├─ Inputs: flows and zones DataFrames
├─ Processing:
│   ├─ Draw zones on left and right
│   ├─ Draw flow lines (width ∝ trips)
│   └─ Add trip count labels
└─ Output: matplotlib.figure.Figure

create_employment_centers_map(emp_data: DataFrame) → Figure
├─ Inputs: employment centers DataFrame
├─ Processing:
│   ├─ Normalize employment for bubble size
│   ├─ Map sectors to colors
│   ├─ Draw bubble scatter plot
│   └─ Add labels and legend
└─ Output: matplotlib.figure.Figure

create_mode_share_chart(mode_share_data: DataFrame) → Figure
├─ Inputs: mode share DataFrame
├─ Processing:
│   ├─ Create 6-panel figure (1 stacked bar + 5 individual)
│   ├─ Stacked bar: all modes, all counties
│   └─ Individual: one mode per county comparison
└─ Output: matplotlib.figure.Figure

create_incentive_impact_summary(incentive_data: DataFrame) → Figure
├─ Inputs: incentive impact DataFrame
├─ Processing:
│   ├─ Create 3-panel figure
│   ├─ Left: Carpool uplift potential
│   ├─ Center: Transit uplift potential
│   └─ Right: Overall VMT reduction potential
└─ Output: matplotlib.figure.Figure
```

## Error Handling & Validation

```
Data Loading
    │
    ├─ Check file exists
    ├─ Validate CSV format
    ├─ Check required columns
    ├─ Verify data types
    ├─ Check for NaN values
    ├─ Validate numeric ranges
    │
    └─ If valid → Process & visualize
       If invalid → Log error & use defaults
```

## Performance Optimization Points

```
Current → Optimized

1. Data Loading
   Every load → Cache with @lru_cache()
   
2. Visualization Generation
   Each request → Pre-generate on startup
   
3. Gradio Rendering
   Live computation → Precomputed figures
   
4. Data Processing
   Real-time → Batch processing overnight
   
Result: <1 sec load time instead of 5+ sec
```

## Deployment Architecture

```
┌──────────────────────────────────────┐
│     Local Development                │
│  python main_dashboard_updated.py    │
│  http://localhost:7860               │
└──────────────────────────────────────┘
                    ↓ (no changes needed)
┌──────────────────────────────────────┐
│  Hugging Face Spaces                 │
│  (requires Dockerfile + requirements │
│   but code identical)                │
└──────────────────────────────────────┘
                    ↓ (minimal changes)
┌──────────────────────────────────────┐
│  Cloud Deployment                    │
│  (AWS/GCP with load balancing)       │
│  Code identical, config differs      │
└──────────────────────────────────────┘
```

---

This architecture allows for:
✅ Modular design (easy to extend/modify)
✅ Clear separation of concerns (data/viz/UI)
✅ Reusable components
✅ Easy testing (mock data)
✅ Performance optimization (caching)
✅ Easy deployment (portable code)
