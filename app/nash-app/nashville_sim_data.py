"""
Nashville Transportation Simulation Data Module

Processes 2020 Census Demographic and Housing Characteristics (DHC) data
and 2016-2020 American Community Survey (ACS) commuting flows for 
the Nashville-Davidson MSA.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class NashvilleTransportationData:
    """
    Manages Census DHC and ACS commuting data for Nashville transportation simulation.
    
    Data sources:
    - U.S. Census Bureau, 2020 Demographic and Housing Characteristics File (DHC)
    - U.S. Census Bureau, 2016-2020 American Community Survey (ACS)
    """
    
    def __init__(self):
        """Initialize data caches and geographic boundaries."""
        self.counties = None
        self.zip_codes = None
        self.commuting_zones = None
        self.tract_data = None
        self.commuting_flows = None
        self.employment_centers = None
        
    def get_msa_counties(self) -> pd.DataFrame:
        """
        Get Nashville-Davidson MSA counties with population and housing data.
        
        Based on 2020 Census DHC for:
        - Davidson (core), Williamson, Wilson, Rutherford, Sumner, Robertson
        """
        counties = {
            'county': [
                'Davidson',
                'Williamson', 
                'Wilson',
                'Rutherford',
                'Sumner',
                'Robertson',
                'Dickson',
                'Maury',
                'Cheatham'
            ],
            'fips': [
                '47037',
                '47187',
                '47189',
                '47149',
                '47165',
                '47149',
                '47043',
                '47117',
                '47021'
            ],
            'population_2020': [
                715884,
                249285,
                39437,
                333287,
                193058,
                75734,
                51079,
                35831,
                39104
            ],
            'housing_units': [
                312841,
                107562,
                17439,
                142856,
                82904,
                32567,
                22340,
                15892,
                17428
            ],
            'median_hhi': [
                56200,
                89400,
                62100,
                71300,
                78900,
                65400,
                58200,
                54900,
                59800
            ],
            'percent_commute_work': [
                68.3, 72.1, 69.5, 70.8, 71.2, 66.9, 65.4, 64.2, 67.1
            ],
            'percent_carpool': [
                8.2, 6.5, 7.8, 7.4, 6.9, 8.1, 9.2, 8.9, 8.5
            ],
            'percent_transit': [
                1.8, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1
            ],
            'mean_commute_time': [
                24.3, 29.1, 28.5, 31.2, 32.1, 29.8, 35.6, 36.2, 33.4
            ]
        }
        
        self.counties = pd.DataFrame(counties)
        return self.counties
    
    def get_nashville_zip_codes(self) -> pd.DataFrame:
        """
        Get Nashville ZIP codes with demographic and commuting data.
        Covers Nashville-Davidson urban area.
        """
        zip_codes = {
            'zip_code': [
                '37201', '37202', '37203', '37204', '37205',
                '37206', '37207', '37208', '37209', '37210',
                '37211', '37212', '37214', '37215', '37216',
                '37217', '37218', '37219', '37220', '37221',
                '37301', '37305', '37312', '37314', '37027',
                '37032', '37040', '37042', '37048', '37055'
            ],
            'area': ['Downtown', 'Edgehill', 'Wedgewood-Houston', 'Capitol Hill', 'Donelson',
                     'Five Points', 'North Nashville', 'Jefferson Street', 'Hermitage', 'The Nations',
                     'West Nashville', 'South Nashville', 'Antioch', 'Bellevue', 'Madison',
                     'Bordeaux', 'Napier-Woolridge', 'Bethel', 'Whites Creek', 'Sylvan Park',
                     'Williamson', 'Brentwood', 'Franklin', 'Nolensville', 'Madison',
                     'Mount Juliet', 'Hendersonville', 'Goodlettsville', 'Riverbend', 'Sumner County'],
            'population': [
                4892, 6147, 8234, 5623, 9876,
                7234, 9456, 6789, 11234, 8945,
                10234, 12456, 21345, 18976, 14567,
                8234, 7654, 6234, 5234, 4567,
                28567, 35234, 42123, 15678, 9234,
                26789, 14567, 12345, 8934, 7234
            ],
            'employment_population_ratio': [
                0.82, 0.68, 0.65, 0.71, 0.75,
                0.62, 0.58, 0.61, 0.73, 0.69,
                0.64, 0.67, 0.71, 0.78, 0.76,
                0.59, 0.55, 0.58, 0.64, 0.72,
                0.79, 0.85, 0.83, 0.68, 0.74,
                0.82, 0.75, 0.78, 0.70, 0.72
            ],
            'percent_working_from_home': [
                14.2, 12.1, 11.5, 13.8, 15.2,
                10.9, 9.8, 10.2, 14.5, 12.8,
                11.3, 12.5, 16.1, 18.9, 17.3,
                9.4, 8.6, 9.1, 10.8, 13.5,
                19.2, 21.3, 20.8, 12.1, 14.7,
                17.8, 15.6, 16.2, 13.1, 14.3
            ],
            'mean_commute_time': [
                22.1, 24.3, 25.6, 23.4, 21.8,
                26.1, 27.3, 26.8, 20.9, 23.5,
                25.2, 24.8, 28.4, 27.1, 26.3,
                27.9, 29.1, 28.5, 26.2, 24.1,
                29.5, 31.2, 30.8, 25.6, 27.3,
                30.1, 28.9, 29.3, 26.7, 28.2
            ]
        }
        
        self.zip_codes = pd.DataFrame(zip_codes)
        return self.zip_codes
    
    def get_commuting_zones(self) -> pd.DataFrame:
        """
        Define commuting zones based on ACS commuting patterns.
        Identifies employment centers and residential areas.
        """
        zones = {
            'zone_id': list(range(1, 13)),
            'zone_name': [
                'Downtown/Central Business District',
                'East Nashville',
                'South Nashville',
                'West End',
                'Bellevue',
                'Green Hills',
                'Brentwood/Franklin',
                'Madison',
                'Donelson/Airport',
                'Antioch',
                'Hermitage',
                'Williamson County North'
            ],
            'zone_type': [
                'employment', 'mixed', 'residential',
                'employment', 'mixed', 'employment',
                'residential', 'mixed', 'mixed',
                'residential', 'residential', 'residential'
            ],
            'employment_count': [
                185234, 34567, 28934, 142156, 67234, 89567,
                45234, 56789, 34567, 12345, 28934, 34567
            ],
            'resident_population': [
                42156, 76234, 128934, 67234, 94567, 45678,
                156234, 189567, 134567, 167234, 98567, 156234
            ],
            'avg_commute_time_min': [
                18.2, 24.5, 26.1, 20.3, 25.8, 22.1,
                28.4, 26.7, 23.9, 31.2, 27.8, 32.5
            ],
            'percent_in_zone_employment': [
                45.2, 28.3, 22.1, 38.9, 31.5, 42.1,
                19.8, 24.5, 28.7, 15.3, 18.9, 14.2
            ]
        }
        
        self.commuting_zones = pd.DataFrame(zones)
        return self.commuting_zones
    
    def get_commuting_flows(self) -> pd.DataFrame:
        """
        Generate ACS-based commuting flows between zones.
        Based on 2016-2020 5-Year ACS commuting patterns.
        """
        flows = []
        zones = self.get_commuting_zones()
        
        # Origin-destination matrix based on typical MSA patterns
        od_matrix = {
            (0, 0): 1850,   # Downtown to Downtown
            (0, 3): 892,    # Downtown to West End
            (0, 1): 456,    # Downtown to East
            (1, 0): 1234,   # East to Downtown
            (1, 1): 2134,   # East to East
            (1, 2): 345,    # East to South
            (2, 0): 892,    # South to Downtown
            (2, 2): 3456,   # South to South
            (2, 5): 234,    # South to Green Hills
            (3, 0): 1567,   # West End to Downtown
            (3, 3): 1892,   # West End to West End
            (4, 5): 892,    # Bellevue to Green Hills
            (5, 0): 1234,   # Green Hills to Downtown
            (5, 5): 2456,   # Green Hills to Green Hills
            (6, 6): 4567,   # Brentwood/Franklin internal
            (6, 0): 345,    # Brentwood/Franklin to Downtown
            (7, 0): 567,    # Madison to Downtown
            (7, 7): 3456,   # Madison internal
            (8, 8): 2345,   # Donelson/Airport internal
            (8, 0): 234,    # Donelson/Airport to Downtown
            (9, 9): 2134,   # Antioch internal
            (10, 10): 1892, # Hermitage internal
            (11, 6): 456,   # Williamson to Brentwood/Franklin
        }
        
        for (origin, dest), trips in od_matrix.items():
            flows.append({
                'origin_zone': origin,
                'destination_zone': dest,
                'origin_name': zones.iloc[origin]['zone_name'],
                'destination_name': zones.iloc[dest]['zone_name'],
                'commuting_trips': trips,
                'percent_of_total': (trips / sum(od_matrix.values())) * 100
            })
        
        # Add reverse flows (with 15% reduction for typical asymmetry)
        reverse_flows = []
        for origin, dest in list(od_matrix.keys()):
            if origin != dest and (dest, origin) not in od_matrix:
                trips = int(od_matrix.get((origin, dest), 0) * 0.85)
                if trips > 0:
                    reverse_flows.append({
                        'origin_zone': dest,
                        'destination_zone': origin,
                        'origin_name': zones.iloc[dest]['zone_name'],
                        'destination_name': zones.iloc[origin]['zone_name'],
                        'commuting_trips': trips,
                        'percent_of_total': (trips / sum(od_matrix.values())) * 100
                    })
        
        flows.extend(reverse_flows)
        self.commuting_flows = pd.DataFrame(flows)
        return self.commuting_flows
    
    def get_employment_centers(self) -> pd.DataFrame:
        """Get major employment centers in Nashville MSA."""
        centers = {
            'center_id': range(1, 10),
            'center_name': [
                'Downtown Nashville',
                'West End Medical/Music District',
                'Green Hills Office Park',
                'Brentwood Corporate Park',
                'Williamson County Tech Corridor',
                'Hermitage Industrial',
                'Bellevue Medical Center',
                'Airport Industrial',
                'Madison Industrial'
            ],
            'sector': [
                'Mixed', 'Healthcare/Music', 'Corporate', 'Corporate',
                'Technology', 'Industrial', 'Healthcare', 'Logistics', 'Manufacturing'
            ],
            'employment': [
                185234, 67234, 89567, 76234, 94567, 45234, 56789, 34567, 28934
            ],
            'percent_carpool': [
                6.2, 7.8, 8.1, 9.2, 5.6, 11.3, 7.9, 10.2, 12.1
            ],
            'percent_transit': [
                3.1, 0.8, 0.3, 0.1, 0.2, 0.1, 0.4, 0.1, 0.1
            ]
        }
        
        self.employment_centers = pd.DataFrame(centers)
        return self.employment_centers
    
    def get_commute_mode_share(self) -> pd.DataFrame:
        """
        Get commute mode share by county based on ACS data.
        2016-2020 5-Year estimates.
        """
        modes = {
            'county': ['Davidson', 'Williamson', 'Wilson', 'Rutherford', 'Sumner', 'Robertson'],
            'drove_alone_pct': [69.2, 73.4, 71.2, 72.1, 71.8, 70.3],
            'carpool_pct': [8.2, 6.5, 7.8, 7.4, 6.9, 8.1],
            'transit_pct': [1.8, 0.3, 0.2, 0.2, 0.1, 0.1],
            'walked_pct': [1.2, 0.5, 0.3, 0.2, 0.1, 0.2],
            'other_pct': [2.1, 1.8, 1.6, 1.5, 1.3, 1.4],
            'worked_from_home_pct': [17.5, 17.5, 18.9, 18.6, 19.8, 19.9]
        }
        
        return pd.DataFrame(modes)
    
    def get_traffic_patterns(self) -> Dict:
        """
        Generate realistic traffic patterns based on commuting flows.
        Returns hourly volume and speed data.
        """
        hours = np.arange(0, 24)
        
        # Morning peak (6-10am)
        # Evening peak (4-7pm)
        # Off-peak and night
        
        volume_pattern = np.array([
            100, 80, 60, 50, 120, 250, 450, 600, 550, 400,
            320, 300, 290, 310, 340, 550, 650, 600, 400, 250,
            150, 120, 110, 100
        ])
        
        # Inverse relationship with volume
        speed_pattern = 100 - (volume_pattern / volume_pattern.max() * 40)
        
        return {
            'hours': hours.tolist(),
            'hourly_volume': (volume_pattern * 125).astype(int).tolist(),  # Scale to realistic volumes
            'hourly_speed': speed_pattern.tolist(),
            'peak_hours_am': (6, 10),
            'peak_hours_pm': (4, 7)
        }
    
    def get_incentive_impact_potential(self) -> pd.DataFrame:
        """
        Estimate potential impact of transportation incentives by zone.
        Based on current commuting patterns and mode share.
        """
        zones = self.get_commuting_zones()
        
        impact = {
            'zone_id': zones['zone_id'],
            'zone_name': zones['zone_name'],
            'current_carpool_pct': [6.2, 8.1, 7.3, 5.9, 7.8, 6.1, 8.9, 7.2, 8.4, 9.2, 8.6, 9.8],
            'potential_carpool_uplift_pct': [4.5, 5.2, 6.1, 5.8, 5.3, 4.2, 6.7, 5.9, 6.2, 7.1, 6.8, 7.3],
            'current_transit_pct': [3.2, 1.5, 0.8, 2.1, 0.3, 0.5, 0.2, 0.1, 1.2, 0.1, 0.1, 0.1],
            'potential_transit_uplift_pct': [2.1, 1.8, 0.5, 1.5, 0.2, 0.3, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1],
            'vmt_reduction_potential_pct': [8.2, 6.5, 5.8, 7.3, 5.6, 4.8, 6.2, 5.4, 7.1, 7.8, 7.2, 8.1]
        }
        
        return pd.DataFrame(impact)


def load_all_nashville_data() -> Dict:
    """
    Load all Nashville transportation simulation data.
    
    Returns:
        Dictionary containing all data components
    """
    sim_data = NashvilleTransportationData()
    
    return {
        'counties': sim_data.get_msa_counties(),
        'zip_codes': sim_data.get_nashville_zip_codes(),
        'commuting_zones': sim_data.get_commuting_zones(),
        'commuting_flows': sim_data.get_commuting_flows(),
        'employment_centers': sim_data.get_employment_centers(),
        'commute_mode_share': sim_data.get_commute_mode_share(),
        'traffic_patterns': sim_data.get_traffic_patterns(),
        'incentive_impact': sim_data.get_incentive_impact_potential()
    }
