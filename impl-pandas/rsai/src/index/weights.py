"""Weighting scheme implementations for RSAI aggregation"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class WeightingScheme(ABC):
    """Abstract base class for weighting schemes"""
    
    @abstractmethod
    def calculate_weights(self, supertract_data: pd.DataFrame, 
                         year: int, **kwargs) -> pd.Series:
        """
        Calculate weights for supertracts.
        
        Parameters:
        -----------
        supertract_data: pd.DataFrame
            DataFrame with supertract information
        year: int
            Year to calculate weights for
        **kwargs: 
            Additional data needed for specific weighting schemes
            
        Returns:
        --------
        pd.Series
            Series with supertract_id as index and weights as values
        """
        pass
    
    def normalize_weights(self, weights: pd.Series) -> pd.Series:
        """Ensure weights sum to 1"""
        weight_sum = weights.sum()
        if weight_sum == 0:
            logger.warning("All weights are zero, using equal weights")
            return pd.Series(1.0 / len(weights), index=weights.index)
        return weights / weight_sum


class SampleWeighting(WeightingScheme):
    """Sample-based weighting using half-pairs count"""
    
    def calculate_weights(self, supertract_data: pd.DataFrame, 
                         year: int, **kwargs) -> pd.Series:
        """Calculate weights based on share of half-pairs"""
        year_data = supertract_data[supertract_data['year'] == year]
        
        if 'half_pairs_count' not in year_data.columns:
            raise ValueError("half_pairs_count column required for sample weighting")
        
        weights = year_data.set_index('supertract_id')['half_pairs_count']
        return self.normalize_weights(weights)


class ValueWeighting(WeightingScheme):
    """Value-based (Laspeyres) weighting using housing values"""
    
    def calculate_weights(self, supertract_data: pd.DataFrame, 
                         year: int, **kwargs) -> pd.Series:
        """Calculate weights based on aggregate housing value"""
        weighting_df = kwargs.get('weighting_data')
        if weighting_df is None:
            raise ValueError("weighting_data required for value weighting")
        
        year_data = supertract_data[supertract_data['year'] == year]
        
        # Calculate aggregate value for each supertract
        weights = pd.Series(index=year_data['supertract_id'], dtype=float)
        
        # Use previous year's values (Laspeyres index)
        value_year = year - 1
        
        for _, row in year_data.iterrows():
            supertract_id = row['supertract_id']
            component_tracts = row['component_tracts']
            
            # Sum values across component tracts
            tract_values = weighting_df[
                (weighting_df['census_tract_2010'].isin(component_tracts)) &
                (weighting_df['year'] == value_year)
            ]['total_housing_value'].sum()
            
            weights[supertract_id] = tract_values
        
        return self.normalize_weights(weights)


class UnitWeighting(WeightingScheme):
    """Unit-based weighting using housing unit counts"""
    
    def calculate_weights(self, supertract_data: pd.DataFrame, 
                         year: int, **kwargs) -> pd.Series:
        """Calculate weights based on number of housing units"""
        weighting_df = kwargs.get('weighting_data')
        if weighting_df is None:
            raise ValueError("weighting_data required for unit weighting")
        
        year_data = supertract_data[supertract_data['year'] == year]
        
        weights = pd.Series(index=year_data['supertract_id'], dtype=float)
        
        for _, row in year_data.iterrows():
            supertract_id = row['supertract_id']
            component_tracts = row['component_tracts']
            
            # Sum units across component tracts
            tract_units = weighting_df[
                (weighting_df['census_tract_2010'].isin(component_tracts)) &
                (weighting_df['year'] == year)
            ]['total_housing_units'].sum()
            
            weights[supertract_id] = tract_units
        
        return self.normalize_weights(weights)


class UPBWeighting(WeightingScheme):
    """Unpaid Principal Balance weighting"""
    
    def calculate_weights(self, supertract_data: pd.DataFrame, 
                         year: int, **kwargs) -> pd.Series:
        """Calculate weights based on aggregate UPB"""
        weighting_df = kwargs.get('weighting_data')
        if weighting_df is None:
            raise ValueError("weighting_data required for UPB weighting")
        
        year_data = supertract_data[supertract_data['year'] == year]
        
        weights = pd.Series(index=year_data['supertract_id'], dtype=float)
        
        for _, row in year_data.iterrows():
            supertract_id = row['supertract_id']
            component_tracts = row['component_tracts']
            
            # Sum UPB across component tracts
            tract_upb = weighting_df[
                (weighting_df['census_tract_2010'].isin(component_tracts)) &
                (weighting_df['year'] == year)
            ]['total_upb'].sum()
            
            weights[supertract_id] = tract_upb
        
        return self.normalize_weights(weights)


class DemographicWeighting(WeightingScheme):
    """Base class for demographic-based weightings (College/Non-White)"""
    
    def __init__(self, demographic_column: str):
        self.demographic_column = demographic_column
    
    def calculate_weights(self, supertract_data: pd.DataFrame, 
                         year: int, **kwargs) -> pd.Series:
        """Calculate weights based on demographic data"""
        weighting_df = kwargs.get('weighting_data')
        if weighting_df is None:
            raise ValueError("weighting_data required for demographic weighting")
        
        year_data = supertract_data[supertract_data['year'] == year]
        
        weights = pd.Series(index=year_data['supertract_id'], dtype=float)
        
        # Use static 2010 demographic data
        demo_data = weighting_df[weighting_df['year'] == 2010]
        
        for _, row in year_data.iterrows():
            supertract_id = row['supertract_id']
            component_tracts = row['component_tracts']
            
            # Sum demographic values across component tracts
            tract_demo = demo_data[
                demo_data['census_tract_2010'].isin(component_tracts)
            ][self.demographic_column].sum()
            
            weights[supertract_id] = tract_demo
        
        return self.normalize_weights(weights)


class CollegeWeighting(DemographicWeighting):
    """College-educated population weighting"""
    
    def __init__(self):
        super().__init__('college_population')


class NonWhiteWeighting(DemographicWeighting):
    """Non-white population weighting"""
    
    def __init__(self):
        super().__init__('non_white_population')


class WeightCalculator:
    """Factory and calculator for all weighting schemes"""
    
    def __init__(self):
        self.schemes = {
            'sample': SampleWeighting(),
            'value': ValueWeighting(),
            'unit': UnitWeighting(),
            'upb': UPBWeighting(),
            'college': CollegeWeighting(),
            'non_white': NonWhiteWeighting()
        }
    
    def calculate_weights(self, scheme_name: str,
                         supertract_data: pd.DataFrame,
                         year: int,
                         weighting_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Calculate weights using specified scheme.
        
        Parameters:
        -----------
        scheme_name: str
            Name of weighting scheme ('sample', 'value', 'unit', 'upb', 'college', 'non_white')
        supertract_data: pd.DataFrame
            DataFrame with supertract definitions
        year: int
            Year to calculate weights for
        weighting_data: pd.DataFrame, optional
            Additional data needed for non-sample weighting schemes
            
        Returns:
        --------
        pd.Series
            Normalized weights for each supertract
        """
        if scheme_name not in self.schemes:
            raise ValueError(f"Unknown weighting scheme: {scheme_name}")
        
        scheme = self.schemes[scheme_name]
        
        kwargs = {}
        if weighting_data is not None:
            kwargs['weighting_data'] = weighting_data
        
        return scheme.calculate_weights(supertract_data, year, **kwargs)
    
    def calculate_all_weights(self, supertract_data: pd.DataFrame,
                            year: int,
                            weighting_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate weights for all schemes.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with supertract_id as index and weight schemes as columns
        """
        all_weights = {}
        
        for scheme_name in self.schemes:
            try:
                weights = self.calculate_weights(
                    scheme_name, supertract_data, year, weighting_data
                )
                all_weights[scheme_name] = weights
            except Exception as e:
                logger.warning(f"Failed to calculate {scheme_name} weights: {str(e)}")
                # Use equal weights as fallback
                year_data = supertract_data[supertract_data['year'] == year]
                n_supertracts = len(year_data)
                all_weights[scheme_name] = pd.Series(
                    1.0 / n_supertracts,
                    index=year_data['supertract_id']
                )
        
        return pd.DataFrame(all_weights)
    
    def add_custom_scheme(self, name: str, scheme: WeightingScheme):
        """Add a custom weighting scheme"""
        self.schemes[name] = scheme