"""
Integration code for adding Nashville Transportation Simulation tab to the existing dashboard.

This shows how to integrate the Nashville simulation tab into your main Gradio app.
Add this to your main dashboard file (around where other tabs are created).
"""

import gradio as gr
from nashville_sim_data import load_all_nashville_data
from nashville_sim_components import (
    create_county_map,
    create_zip_code_heatmap,
    create_commuting_zone_map,
    create_commuting_flow_sankey,
    create_employment_centers_map,
    create_mode_share_chart,
    create_incentive_impact_summary
)


def create_nashville_simulation_tab():
    """
    Create the Nashville Transportation Simulation tab.
    
    This tab includes:
    - County-level analysis with population and commuting data
    - ZIP code analysis with commute times and employment ratios
    - Commuting zone mapping with employment centers
    - Commuting flow visualization
    - Major employment centers map
    - Commute mode share analysis
    - Incentive impact potential by zone
    """
    
    # Load all Nashville transportation data
    nash_data = load_all_nashville_data()
    
    with gr.Column():
        gr.Markdown("""
        ## Nashville Transportation Simulation
        
        Interactive analysis of Nashville-Davidson MSA transportation patterns based on:
        - **2020 Census Demographic and Housing Characteristics File (DHC)**
        - **2016-2020 American Community Survey (ACS) 5-Year Estimates**
        
        Explore commuting flows, employment centers, and transportation incentive impacts across
        counties, ZIP codes, and commuting zones.
        """)
        
        # Tab 1: Geographic Overview
        with gr.Tab("Geographic Overview"):
            with gr.Column():
                gr.Markdown("### Nashville MSA - County-Level Analysis")
                gr.Markdown("""
                This map shows the Nashville-Davidson Metropolitan Statistical Area (MSA) 
                with 2020 Census population data and commuting statistics from the ACS.
                Darker shades indicate larger populations.
                """)
                
                county_fig = create_county_map(nash_data['counties'])
                gr.Plot(value=county_fig, label="County Population Map")
                
                # County statistics table
                with gr.Row():
                    gr.Dataframe(
                        value=nash_data['counties'][['county', 'population_2020', 'housing_units', 
                                                     'mean_commute_time', 'percent_carpool', 
                                                     'percent_transit']],
                        label="County Statistics",
                        interactive=False,
                        headers=['County', 'Population', 'Housing Units', 'Avg Commute (min)', 
                                'Carpool %', 'Transit %']
                    )
        
        # Tab 2: ZIP Code Analysis
        with gr.Tab("ZIP Code Analysis"):
            with gr.Column():
                gr.Markdown("### Nashville ZIP Codes - Commuting Patterns")
                gr.Markdown("""
                Analysis of 30 Nashville-area ZIP codes showing:
                - **Mean Commute Time**: Based on ACS 2016-2020 data
                - **Employment-Population Ratio**: Identifies employment centers vs residential areas
                - **Work-from-Home Rate**: Percentage working from home
                """)
                
                zip_fig = create_zip_code_heatmap(nash_data['zip_codes'])
                gr.Plot(value=zip_fig, label="ZIP Code Heatmap")
                
                # ZIP code data table
                with gr.Row():
                    gr.Dataframe(
                        value=nash_data['zip_codes'][['zip_code', 'area', 'population', 
                                                      'mean_commute_time', 'employment_population_ratio',
                                                      'percent_working_from_home']],
                        label="ZIP Code Data",
                        interactive=False,
                        headers=['ZIP', 'Area', 'Population', 'Avg Commute (min)', 
                                'Emp-Pop Ratio', 'WFH %']
                    )
        
        # Tab 3: Commuting Zones
        with gr.Tab("Commuting Zones"):
            with gr.Column():
                gr.Markdown("### Nashville Commuting Zones")
                gr.Markdown("""
                12 identified commuting zones based on ACS commuting patterns:
                - **Red zones**: Employment centers (Downtown, West End, Green Hills, etc.)
                - **Yellow zones**: Mixed-use areas
                - **Green zones**: Primarily residential areas
                
                Box size represents residential population; employment count shown inside.
                """)
                
                zones_fig = create_commuting_zone_map(nash_data['commuting_zones'])
                gr.Plot(value=zones_fig, label="Commuting Zones Map")
                
                # Commuting zones statistics
                with gr.Row():
                    gr.Dataframe(
                        value=nash_data['commuting_zones'][['zone_name', 'zone_type', 'employment_count', 
                                                           'resident_population', 'avg_commute_time_min',
                                                           'percent_in_zone_employment']],
                        label="Zone Statistics",
                        interactive=False,
                        headers=['Zone', 'Type', 'Employment', 'Population', 'Avg Commute (min)', 
                                'In-Zone Emp %']
                    )
        
        # Tab 4: Employment Centers
        with gr.Tab("Employment Centers"):
            with gr.Column():
                gr.Markdown("### Major Employment Centers in Nashville MSA")
                gr.Markdown("""
                9 major employment centers identified based on ACS data:
                - **Bubble size**: Number of employees (2020-2024 estimates)
                - **Color**: Industry sector
                - **Bottom label**: Current carpool and transit percentages
                """)
                
                emp_fig = create_employment_centers_map(nash_data['employment_centers'])
                gr.Plot(value=emp_fig, label="Employment Centers")
                
                # Employment centers statistics
                with gr.Row():
                    gr.Dataframe(
                        value=nash_data['employment_centers'][['center_name', 'sector', 'employment', 
                                                              'percent_carpool', 'percent_transit']],
                        label="Employment Center Statistics",
                        interactive=False,
                        headers=['Center', 'Sector', 'Employment', 'Carpool %', 'Transit %']
                    )
        
        # Tab 5: Commuting Flows
        with gr.Tab("Commuting Flows"):
            with gr.Column():
                gr.Markdown("### Origin-Destination Commuting Flows")
                gr.Markdown("""
                Visualization of the 15 largest commuting flows between zones.
                Based on ACS 2016-2020 journey-to-work data.
                
                - **Line width**: Proportional to number of commuting trips
                - **Numbers**: Estimated daily commuting trips
                """)
                
                flow_fig = create_commuting_flow_sankey(nash_data['commuting_flows'], 
                                                       nash_data['commuting_zones'])
                gr.Plot(value=flow_fig, label="Commuting Flows")
                
                # Top flows table
                top_flows = nash_data['commuting_flows'].nlargest(10, 'commuting_trips')
                with gr.Row():
                    gr.Dataframe(
                        value=top_flows[['origin_name', 'destination_name', 'commuting_trips', 
                                        'percent_of_total']],
                        label="Top 10 Commuting Flows",
                        interactive=False,
                        headers=['Origin Zone', 'Destination Zone', 'Daily Trips', 'Percent of Total']
                    )
        
        # Tab 6: Mode Share
        with gr.Tab("Mode Share Analysis"):
            with gr.Column():
                gr.Markdown("### Commute Mode Share by County")
                gr.Markdown("""
                Breakdown of commuting modes by county based on ACS 2016-2020 5-Year Estimates:
                - **Drove Alone**: Percent driving alone
                - **Carpool**: Shared driving
                - **Transit**: Public transportation
                - **Walked/Other**: Active and other modes
                - **Work from Home**: Percent working remotely
                """)
                
                mode_fig = create_mode_share_chart(nash_data['commute_mode_share'])
                gr.Plot(value=mode_fig, label="Mode Share")
                
                # Mode share statistics
                with gr.Row():
                    gr.Dataframe(
                        value=nash_data['commute_mode_share'],
                        label="Mode Share Statistics (%)",
                        interactive=False
                    )
        
        # Tab 7: Incentive Impact Analysis
        with gr.Tab("Incentive Impact Potential"):
            with gr.Column():
                gr.Markdown("### Transportation Incentive Impact by Zone")
                gr.Markdown("""
                Analysis of potential VMT reduction and mode shift from transportation incentive programs:
                
                **Left Chart**: Potential carpool rate increase with incentives
                **Middle Chart**: Potential transit adoption increase  
                **Right Chart**: Overall VMT reduction potential
                
                These estimates are based on elasticity of demand for different commuting modes
                and current mode share in each zone.
                """)
                
                incentive_fig = create_incentive_impact_summary(nash_data['incentive_impact'])
                gr.Plot(value=incentive_fig, label="Incentive Impact")
                
                # Impact potential table
                with gr.Row():
                    gr.Dataframe(
                        value=nash_data['incentive_impact'][['zone_name', 'current_carpool_pct',
                                                            'potential_carpool_uplift_pct',
                                                            'current_transit_pct',
                                                            'potential_transit_uplift_pct',
                                                            'vmt_reduction_potential_pct']],
                        label="Incentive Impact Potential",
                        interactive=False,
                        headers=['Zone', 'Current Carpool %', 'Carpool Uplift Potential %',
                                'Current Transit %', 'Transit Uplift Potential %',
                                'VMT Reduction Potential %']
                    )
        
        # Tab 8: Data Summary
        with gr.Tab("Data Summary & Sources"):
            with gr.Column():
                gr.Markdown("""
                ## Data Sources & Methodology
                
                ### Primary Data Sources
                
                **2020 Census Demographic and Housing Characteristics File (DHC)**
                - Population counts by county and geographic area
                - Housing unit counts
                - Basic demographic characteristics
                
                **American Community Survey (ACS) 2016-2020 5-Year Estimates**
                - Journey-to-work commuting data
                - Commute time by geography
                - Mode of transportation to work
                - Work-from-home statistics
                - Median household income
                
                ### Data Coverage
                - **Geographic scope**: Nashville-Davidson Metropolitan Statistical Area (MSA)
                - **Core counties**: Davidson, Williamson, Sumner, Rutherford, Wilson, Robertson
                - **Extended coverage**: Additional counties including Dickson, Maury, Cheatham
                - **Time period**: ACS data from 2016-2020; Census data from 2020
                
                ### Commuting Zones
                Zones are defined based on:
                1. ACS journey-to-work flow patterns
                2. Employment center locations
                3. Geographic connectivity
                4. Commuting time isochrones
                
                ### Employment Centers
                Identified through:
                - Bureau of Labor Statistics data
                - ACS place-of-work statistics
                - LODES employment data
                - Survey of commercial real estate markets
                
                ### Incentive Impact Analysis
                Impact potential is estimated based on:
                - Current mode share in each zone
                - Elasticity of demand for carpooling and transit
                - Comparable programs in similar MSAs
                - Local market characteristics
                
                ---
                
                **Repository**: https://github.com/LNshuti/ihute
                
                Built with: Python, Pandas, Matplotlib, Gradio
                """)
                
                # Quick statistics
                gr.Markdown("### Quick Statistics")
                
                total_pop = nash_data['counties']['population_2020'].sum()
                total_emp = nash_data['employment_centers']['employment'].sum()
                avg_commute = nash_data['commute_mode_share']['drove_alone_pct'].mean()
                avg_carpool = nash_data['commute_mode_share']['carpool_pct'].mean()
                avg_transit = nash_data['commute_mode_share']['transit_pct'].mean()
                
                stats_text = f"""
                | Metric | Value |
                |--------|-------|
                | **Total MSA Population (2020)** | {total_pop:,} |
                | **Total Employment** | {total_emp:,} |
                | **Counties in MSA** | {len(nash_data['counties'])} |
                | **Commuting Zones** | {len(nash_data['commuting_zones'])} |
                | **Average Drove Alone %** | {avg_commute:.1f}% |
                | **Average Carpool %** | {avg_carpool:.1f}% |
                | **Average Transit %** | {avg_transit:.1f}% |
                """
                
                gr.Markdown(stats_text)


# ============================================================================
# HOW TO INTEGRATE INTO YOUR EXISTING DASHBOARD
# ============================================================================
# 
# In your main dashboard file (where you have the other tabs), add this to
# the gr.Tabs() section:
#
#     with gr.TabItem("Nashville Simulation"):
#         create_nashville_simulation_tab()
#
# Full example integration in create_app():
#
#     def create_app():
#         with gr.Blocks(title="Nashville Incentive Simulation Dashboard") as app:
#             gr.Markdown("# Nashville Transportation Incentive Simulation")
#
#             with gr.Tabs():
#                 with gr.TabItem("Incentive Analytics"):
#                     create_incentive_tab()
#
#                 with gr.TabItem("Behavioral Calibration"):
#                     create_behavioral_tab()
#
#                 with gr.TabItem("Simulation Comparison"):
#                     create_simulation_tab()
#
#                 with gr.TabItem("Live Metrics"):
#                     create_metrics_tab()
#
#                 with gr.TabItem("Corridor Map"):
#                     create_map_tab()
#
#                 # ADD THIS NEW TAB:
#                 with gr.TabItem("Nashville Simulation"):
#                     create_nashville_simulation_tab()
#
#             return app
#
# ============================================================================


if __name__ == "__main__":
    # Test/demo the Nashville simulation tab standalone
    with gr.Blocks(title="Nashville Transportation Simulation") as demo:
        create_nashville_simulation_tab()
    
    demo.launch(server_name="0.0.0.0", share=False)
