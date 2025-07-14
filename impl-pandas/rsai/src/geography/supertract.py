"""Supertract generation algorithm for RSAI"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
import logging
from collections import defaultdict

from .distance import GeographicDistanceCalculator

logger = logging.getLogger(__name__)


class SupertractGenerator:
    """
    Implements the dynamic supertract generation algorithm.
    
    Creates mutually exclusive and exhaustive geographic units (supertracts)
    by merging census tracts that don't meet minimum observation thresholds.
    """
    
    def __init__(self, geographic_df: pd.DataFrame, min_half_pairs: int = 40):
        """
        Initialize the supertract generator.
        
        Parameters:
        -----------
        geographic_df: pd.DataFrame
            DataFrame with census tract geographic information
        min_half_pairs: int
            Minimum number of half-pairs required for independent estimation
        """
        self.geographic_df = geographic_df
        self.min_half_pairs = min_half_pairs
        # Only initialize distance calculator if geographic data is provided
        if not geographic_df.empty and 'centroid_lat' in geographic_df.columns:
            self.distance_calc = GeographicDistanceCalculator(geographic_df)
        else:
            self.distance_calc = None
    
    def calculate_half_pairs(self, repeat_sales_df: pd.DataFrame, 
                           year: int, tract: str) -> int:
        """
        Calculate half-pairs for a given tract and year.
        
        A half-pair is a repeat-sale transaction where either the first
        or second sale occurred in the given year and tract.
        
        Parameters:
        -----------
        repeat_sales_df: pd.DataFrame
            DataFrame of repeat sales pairs
        year: int
            Year to calculate half-pairs for
        tract: str
            Census tract ID
            
        Returns:
        --------
        int
            Number of half-pairs
        """
        # Filter for the specific tract
        tract_sales = repeat_sales_df[repeat_sales_df['census_tract_2010'] == tract]
        
        # Count transactions where first sale is in the given year
        first_sale_count = (
            (tract_sales['first_sale_date'].dt.year == year).sum()
        )
        
        # Count transactions where second sale is in the given year
        second_sale_count = (
            (tract_sales['second_sale_date'].dt.year == year).sum()
        )
        
        return first_sale_count + second_sale_count
    
    def calculate_half_pairs_multi(self, repeat_sales_df: pd.DataFrame,
                                 year: int, tracts: List[str]) -> int:
        """
        Calculate half-pairs for multiple tracts combined.
        
        Parameters:
        -----------
        repeat_sales_df: pd.DataFrame
            DataFrame of repeat sales pairs
        year: int
            Year to calculate half-pairs for
        tracts: List[str]
            List of census tract IDs
            
        Returns:
        --------
        int
            Total number of half-pairs
        """
        # Filter for the specific tracts
        tract_sales = repeat_sales_df[repeat_sales_df['census_tract_2010'].isin(tracts)]
        
        # Count transactions where first sale is in the given year
        first_sale_count = (
            (tract_sales['first_sale_date'].dt.year == year).sum()
        )
        
        # Count transactions where second sale is in the given year
        second_sale_count = (
            (tract_sales['second_sale_date'].dt.year == year).sum()
        )
        
        return first_sale_count + second_sale_count
    
    def generate_supertracts_for_year(self, repeat_sales_df: pd.DataFrame,
                                    year: int, cbsa_id: str) -> Dict[str, List[str]]:
        """
        Generate supertracts for a specific CBSA and year.
        
        Parameters:
        -----------
        repeat_sales_df: pd.DataFrame
            DataFrame of repeat sales pairs
        year: int
            Year to generate supertracts for
        cbsa_id: str
            CBSA ID to process
            
        Returns:
        --------
        Dict[str, List[str]]
            Mapping of supertract_id to list of component census tracts
        """
        logger.info(f"Generating supertracts for CBSA {cbsa_id}, year {year}")
        
        # Filter data for this CBSA
        cbsa_sales = repeat_sales_df[repeat_sales_df['cbsa_id'] == cbsa_id]
        cbsa_tracts = self.geographic_df[self.geographic_df['cbsa_id'] == cbsa_id]
        
        # Calculate half-pairs for each tract in current and previous year
        tract_halfpairs = {}
        for tract in cbsa_tracts['census_tract_2010']:
            halfpairs_current = self.calculate_half_pairs(cbsa_sales, year, tract)
            halfpairs_previous = self.calculate_half_pairs(cbsa_sales, year - 1, tract)
            
            # Tract must meet threshold in both years
            meets_threshold = (halfpairs_current >= self.min_half_pairs and 
                             halfpairs_previous >= self.min_half_pairs)
            
            tract_halfpairs[tract] = {
                'current': halfpairs_current,
                'previous': halfpairs_previous,
                'meets_threshold': meets_threshold
            }
        
        # Initialize supertracts
        supertracts = {}
        processed_tracts = set()
        supertract_counter = 0
        
        # First pass: tracts that meet threshold independently
        for tract, info in tract_halfpairs.items():
            if info['meets_threshold'] and tract not in processed_tracts:
                supertract_id = f"{cbsa_id}_{year}_ST{supertract_counter:04d}"
                supertracts[supertract_id] = [tract]
                processed_tracts.add(tract)
                supertract_counter += 1
        
        # Second pass: iteratively merge tracts that don't meet threshold
        unprocessed = [t for t in tract_halfpairs.keys() if t not in processed_tracts]
        
        while unprocessed:
            # Start with the first unprocessed tract
            current_tract = unprocessed[0]
            current_supertract = [current_tract]
            processed_tracts.add(current_tract)
            unprocessed.remove(current_tract)
            
            # Keep merging until threshold is met
            while True:
                # Calculate current half-pairs
                current_halfpairs = self.calculate_half_pairs_multi(
                    cbsa_sales, year, current_supertract
                )
                previous_halfpairs = self.calculate_half_pairs_multi(
                    cbsa_sales, year - 1, current_supertract
                )
                
                # Check if threshold is met
                if (current_halfpairs >= self.min_half_pairs and 
                    previous_halfpairs >= self.min_half_pairs):
                    break
                
                # Find nearest unprocessed neighbor
                nearest_neighbor = self._find_nearest_unprocessed_neighbor(
                    current_supertract, processed_tracts, cbsa_tracts
                )
                
                if nearest_neighbor is None:
                    # No more neighbors to merge, accept as is
                    logger.warning(f"Supertract with tracts {current_supertract} "
                                 f"does not meet threshold but no neighbors available")
                    break
                
                # Merge the neighbor
                current_supertract.append(nearest_neighbor)
                processed_tracts.add(nearest_neighbor)
                if nearest_neighbor in unprocessed:
                    unprocessed.remove(nearest_neighbor)
            
            # Save the supertract
            supertract_id = f"{cbsa_id}_{year}_ST{supertract_counter:04d}"
            supertracts[supertract_id] = current_supertract
            supertract_counter += 1
        
        logger.info(f"Created {len(supertracts)} supertracts for CBSA {cbsa_id}, year {year}")
        return supertracts
    
    def _find_nearest_unprocessed_neighbor(self, current_supertract: List[str],
                                         processed_tracts: Set[str],
                                         cbsa_tracts: pd.DataFrame) -> str:
        """
        Find the nearest unprocessed neighbor to a supertract.
        
        Parameters:
        -----------
        current_supertract: List[str]
            List of tracts in the current supertract
        processed_tracts: Set[str]
            Set of already processed tract IDs
        cbsa_tracts: pd.DataFrame
            DataFrame of tracts in the current CBSA
            
        Returns:
        --------
        str or None
            ID of nearest unprocessed tract, or None if none available
        """
        # Get all unprocessed tracts
        available_tracts = set(cbsa_tracts['census_tract_2010']) - processed_tracts
        
        if not available_tracts:
            return None
        
        # Find minimum distance from any tract in supertract to any available tract
        min_distance = float('inf')
        nearest_tract = None
        
        for tract in current_supertract:
            for candidate in available_tracts:
                try:
                    distance = self.distance_calc.get_distance_between_tracts(tract, candidate)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_tract = candidate
                except ValueError:
                    continue
        
        return nearest_tract
    
    def generate_all_supertracts(self, repeat_sales_df: pd.DataFrame,
                               start_year: int, end_year: int) -> pd.DataFrame:
        """
        Generate supertracts for all CBSAs and years.
        
        Parameters:
        -----------
        repeat_sales_df: pd.DataFrame
            DataFrame of repeat sales pairs
        start_year: int
            First year to generate supertracts for
        end_year: int
            Last year to generate supertracts for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: supertract_id, year, cbsa_id, component_tracts, half_pairs_count
        """
        all_supertracts = []
        
        # Get unique CBSAs
        cbsas = repeat_sales_df['cbsa_id'].unique()
        
        for cbsa_id in cbsas:
            logger.info(f"Processing CBSA {cbsa_id}")
            
            for year in range(start_year, end_year + 1):
                # Generate supertracts for this CBSA and year
                supertracts = self.generate_supertracts_for_year(
                    repeat_sales_df, year, cbsa_id
                )
                
                # Convert to dataframe format
                for supertract_id, component_tracts in supertracts.items():
                    # Calculate final half-pairs count
                    cbsa_sales = repeat_sales_df[repeat_sales_df['cbsa_id'] == cbsa_id]
                    half_pairs = self.calculate_half_pairs_multi(
                        cbsa_sales, year, component_tracts
                    )
                    
                    all_supertracts.append({
                        'supertract_id': supertract_id,
                        'year': year,
                        'cbsa_id': cbsa_id,
                        'component_tracts': component_tracts,
                        'half_pairs_count': half_pairs
                    })
        
        return pd.DataFrame(all_supertracts)