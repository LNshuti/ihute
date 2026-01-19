#!/usr/bin/env python3
"""
QUICK START: Run Nashville Transportation Simulation Dashboard

This script launches just the Nashville Simulation tab as a standalone Gradio app.
Useful for testing, development, and demo purposes.

Usage:
    python run_nashville_demo.py

Then open http://localhost:7860 in your browser.
"""

import gradio as gr
from nashville_sim_integration import create_nashville_simulation_tab


def main():
    """Launch the Nashville simulation dashboard."""
    
    print("=" * 70)
    print("Nashville Transportation Simulation Dashboard")
    print("=" * 70)
    print()
    print("Starting dashboard...")
    print()
    
    with gr.Blocks(
        title="Nashville Transportation Simulation",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
        # Nashville Transportation Simulation
        
        **Standalone Dashboard for the IHUTE Project**
        
        This dashboard provides comprehensive analysis of Nashville-Davidson MSA
        transportation patterns based on:
        - 2020 Census Demographic and Housing Characteristics File (DHC)
        - 2016-2020 American Community Survey (ACS) Journey-to-Work Data
        
        Use the tabs below to explore:
        - Geographic Overview (counties, ZIP codes)
        - Commuting Zones & Employment Centers
        - Mode Share Analysis
        - Transportation Incentive Impact Potential
        """)
        
        create_nashville_simulation_tab()
        
        gr.Markdown("""
        ---
        
        ### About This Dashboard
        
        This tool is part of the IHUTE (Incentive Heterogeneous Urban Transportation 
        Equilibrium) research project led by Leonce Shuti at Igisha.
        
        **Data Sources:**
        - U.S. Census Bureau 2020 Census DHC
        - U.S. Census Bureau ACS 2016-2020 5-Year Estimates
        - Bureau of Labor Statistics Employment Data
        
        **Repository:** https://github.com/LNshuti/ihute
        
        **Contact:** leonce@igisha.com
        """)
    
    print("=" * 70)
    print("DASHBOARD RUNNING")
    print("=" * 70)
    print()
    print("📊 Open your browser and navigate to:")
    print("   → http://localhost:7860")
    print()
    print("ℹ️  Press Ctrl+C to stop the server")
    print()
    print("=" * 70)
    print()
    
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
        print("\n\nDashboard stopped.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nMake sure you have installed the required dependencies:")
        print("   pip install -r requirements.txt")
        raise
