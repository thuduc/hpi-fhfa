"""City-level aggregation module for RSAI"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from .bmn_regression import run_bmn_for_supertract
from .weights import WeightCalculator

logger = logging.getLogger(__name__)


class CityLevelAggregator:
    """
    Aggregates supertract-level appreciation rates to city (CBSA) level indices.
    
    This class implements the core RSAI methodology by:
    1. Running BMN regressions for each supertract
    2. Calculating various weighting schemes
    3. Aggregating to city-level appreciation rates
    """
    
    def __init__(self, weight_calculator: Optional[WeightCalculator] = None):
        """
        Initialize aggregator.
        
        Parameters:
        -----------
        weight_calculator: WeightCalculator, optional
            Custom weight calculator (if None, uses default)
        """
        self.weight_calculator = weight_calculator or WeightCalculator()
        self.supertract_results = {}
        self.city_results = {}
    
    def calculate_supertract_appreciation(self, 
                                        repeat_sales_df: pd.DataFrame,
                                        supertracts_df: pd.DataFrame,
                                        year: int) -> pd.DataFrame:
        """
        Calculate appreciation rates for all supertracts in a year.
        
        Parameters:
        -----------
        repeat_sales_df: pd.DataFrame
            Repeat sales data
        supertracts_df: pd.DataFrame
            Supertract definitions
        year: int
            Year to calculate appreciation for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: supertract_id, appreciation_rate, n_observations
        """
        logger.info(f"Calculating appreciation rates for year {year}")
        
        year_supertracts = supertracts_df[supertracts_df['year'] == year]
        results = []
        
        for _, row in year_supertracts.iterrows():
            supertract_id = row['supertract_id']
            component_tracts = row['component_tracts']
            
            # Run BMN regression for this supertract
            appreciation_rate, _ = run_bmn_for_supertract(
                repeat_sales_df, component_tracts, year
            )
            
            # Count observations
            n_obs = len(repeat_sales_df[
                repeat_sales_df['census_tract_2010'].isin(component_tracts)
            ])
            
            results.append({
                'supertract_id': supertract_id,
                'cbsa_id': row['cbsa_id'],
                'appreciation_rate': appreciation_rate,
                'n_observations': n_obs
            })
        
        results_df = pd.DataFrame(results)
        
        # Store for later use
        self.supertract_results[year] = results_df
        
        return results_df
    
    def aggregate_to_city_level(self,
                              supertract_appreciation: pd.DataFrame,
                              supertracts_df: pd.DataFrame,
                              year: int,
                              weighting_data: Optional[pd.DataFrame] = None,
                              weighting_schemes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Aggregate supertract appreciation to city level using various weights.
        
        Parameters:
        -----------
        supertract_appreciation: pd.DataFrame
            Supertract-level appreciation rates
        supertracts_df: pd.DataFrame
            Supertract definitions
        year: int
            Year being processed
        weighting_data: pd.DataFrame, optional
            Data for calculating weights
        weighting_schemes: List[str], optional
            List of weighting schemes to use (if None, uses all)
            
        Returns:
        --------
        pd.DataFrame
            City-level appreciation rates for each weighting scheme
        """
        if weighting_schemes is None:
            weighting_schemes = list(self.weight_calculator.schemes.keys())
        
        # Get unique CBSAs
        cbsas = supertract_appreciation['cbsa_id'].unique()
        
        results = []
        
        for cbsa_id in cbsas:
            # Filter for this CBSA
            cbsa_supertracts = supertract_appreciation[
                supertract_appreciation['cbsa_id'] == cbsa_id
            ]
            cbsa_definitions = supertracts_df[
                (supertracts_df['cbsa_id'] == cbsa_id) & 
                (supertracts_df['year'] == year)
            ]
            
            # Calculate weights for each scheme
            for scheme in weighting_schemes:
                try:
                    weights = self.weight_calculator.calculate_weights(
                        scheme, cbsa_definitions, year, weighting_data
                    )
                    
                    # Align weights with appreciation rates
                    aligned_weights = weights.reindex(
                        cbsa_supertracts['supertract_id']
                    ).fillna(0)
                    
                    # Calculate weighted average appreciation
                    weighted_appreciation = np.sum(
                        aligned_weights.values * 
                        cbsa_supertracts['appreciation_rate'].values
                    )
                    
                    results.append({
                        'cbsa_id': cbsa_id,
                        'year': year,
                        'weighting_scheme': scheme,
                        'appreciation_rate': weighted_appreciation,
                        'n_supertracts': len(cbsa_supertracts),
                        'total_observations': cbsa_supertracts['n_observations'].sum()
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to calculate {scheme} weights for "
                               f"CBSA {cbsa_id}, year {year}: {str(e)}")
                    # Add zero appreciation as fallback
                    results.append({
                        'cbsa_id': cbsa_id,
                        'year': year,
                        'weighting_scheme': scheme,
                        'appreciation_rate': 0.0,
                        'n_supertracts': len(cbsa_supertracts),
                        'total_observations': cbsa_supertracts['n_observations'].sum()
                    })
        
        results_df = pd.DataFrame(results)
        
        # Store for later use
        if year not in self.city_results:
            self.city_results[year] = {}
        for scheme in weighting_schemes:
            scheme_results = results_df[results_df['weighting_scheme'] == scheme]
            self.city_results[year][scheme] = scheme_results
        
        return results_df
    
    def process_all_years(self,
                         repeat_sales_df: pd.DataFrame,
                         supertracts_df: pd.DataFrame,
                         start_year: int,
                         end_year: int,
                         weighting_data: Optional[pd.DataFrame] = None,
                         weighting_schemes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Process all years and generate city-level appreciation rates.
        
        Parameters:
        -----------
        repeat_sales_df: pd.DataFrame
            All repeat sales data
        supertracts_df: pd.DataFrame
            All supertract definitions
        start_year: int
            First year to process
        end_year: int
            Last year to process
        weighting_data: pd.DataFrame, optional
            Data for calculating weights
        weighting_schemes: List[str], optional
            List of weighting schemes to use
            
        Returns:
        --------
        pd.DataFrame
            All city-level appreciation rates
        """
        all_results = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Processing year {year}")
            
            # Calculate supertract appreciation
            supertract_results = self.calculate_supertract_appreciation(
                repeat_sales_df, supertracts_df, year
            )
            
            # Aggregate to city level
            city_results = self.aggregate_to_city_level(
                supertract_results, supertracts_df, year,
                weighting_data, weighting_schemes
            )
            
            all_results.append(city_results)
        
        return pd.concat(all_results, ignore_index=True)
    
    def get_appreciation_matrix(self, cbsa_id: str, 
                              weighting_scheme: str) -> pd.DataFrame:
        """
        Get appreciation rates for a specific CBSA and weighting scheme.
        
        Parameters:
        -----------
        cbsa_id: str
            CBSA identifier
        weighting_scheme: str
            Weighting scheme name
            
        Returns:
        --------
        pd.DataFrame
            Time series of appreciation rates
        """
        data = []
        
        for year, year_results in self.city_results.items():
            if weighting_scheme in year_results:
                scheme_df = year_results[weighting_scheme]
                cbsa_data = scheme_df[scheme_df['cbsa_id'] == cbsa_id]
                
                if not cbsa_data.empty:
                    data.append({
                        'year': year,
                        'appreciation_rate': cbsa_data.iloc[0]['appreciation_rate'],
                        'n_observations': cbsa_data.iloc[0]['total_observations']
                    })
        
        return pd.DataFrame(data).sort_values('year')