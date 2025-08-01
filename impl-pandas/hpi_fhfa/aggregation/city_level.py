"""City-level index aggregation from tract/supertract indices."""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from .index_builder import IndexBuilder, HPIIndex
from .weights import WeightCalculator, WeightType
from ..geography.census_tract import CensusTract
from ..geography.supertract import SupertractAlgorithm, Supertract
from ..config.constants import BASE_YEAR
import logging

logger = logging.getLogger(__name__)


class CityLevelIndexBuilder:
    """Build city-level indices by aggregating tract/supertract indices."""
    
    def __init__(self, 
                 base_year: int = BASE_YEAR,
                 min_half_pairs: int = 40):
        """Initialize city-level index builder.
        
        Args:
            base_year: Base year for index construction
            min_half_pairs: Minimum half-pairs for supertract formation
        """
        self.base_year = base_year
        self.min_half_pairs = min_half_pairs
        self.index_builder = IndexBuilder(base_year)
        self.supertract_algorithm = SupertractAlgorithm(min_half_pairs)
        logger.info(f"Initialized CityLevelIndexBuilder with base_year={base_year}")
    
    def build_annual_index(self,
                          transactions: pd.DataFrame,
                          census_tracts: List[CensusTract],
                          weight_type: Union[str, WeightType],
                          start_year: int = 1989,
                          end_year: int = 2021,
                          additional_data: Optional[pd.DataFrame] = None) -> HPIIndex:
        """Build annual city-level index using specified weighting scheme.
        
        This is the main entry point for city-level index construction.
        
        Args:
            transactions: Transaction data with required columns
            census_tracts: List of census tracts in the CBSA
            weight_type: Type of weighting scheme to use
            start_year: Starting year for index
            end_year: Ending year for index
            additional_data: Additional data for certain weight types
            
        Returns:
            City-level HPIIndex
        """
        logger.info(f"Building city-level index with {weight_type} weights "
                   f"from {start_year} to {end_year}")
        
        # Validate inputs
        if not transactions.empty:
            required_cols = ['property_id', 'transaction_date', 'transaction_price',
                           'census_tract']
            missing = set(required_cols) - set(transactions.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
        
        # Extract CBSA code
        cbsa_codes = {tract.cbsa_code for tract in census_tracts}
        if len(cbsa_codes) > 1:
            raise ValueError(f"Multiple CBSA codes found: {cbsa_codes}")
        cbsa_code = cbsa_codes.pop()
        
        # Build repeat sales pairs
        from ..models.repeat_sales import construct_repeat_sales_pairs
        repeat_sales_pairs = construct_repeat_sales_pairs(
            transactions, 
            apply_filters=True
        )
        
        if repeat_sales_pairs.empty:
            logger.warning("No repeat sales pairs found")
            # Return default index
            return self._create_default_index(cbsa_code, start_year, end_year)
        
        # Build indices for each period
        period_indices = {}
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Processing year {year}")
            
            # Build supertracts for this period
            supertracts = self.supertract_algorithm.build_supertracts(
                census_tracts, repeat_sales_pairs, year
            )
            
            if not supertracts:
                logger.warning(f"No supertracts created for year {year}")
                continue
            
            # Build indices for supertracts
            supertract_indices = self.index_builder.build_indices_for_cbsa(
                supertracts, repeat_sales_pairs, start_year, end_year
            )
            
            if not supertract_indices:
                logger.warning(f"No indices built for year {year}")
                continue
            
            # Calculate weights
            weights = WeightCalculator.calculate_weights(
                weight_type, supertracts, repeat_sales_pairs, additional_data
            )
            
            # Aggregate to city level for this period
            city_index_value = self._aggregate_indices_for_period(
                supertract_indices, weights, year
            )
            
            if city_index_value is not None:
                period_indices[year] = city_index_value
        
        # Create final city-level index
        if not period_indices:
            logger.warning("No period indices calculated")
            return self._create_default_index(cbsa_code, start_year, end_year)
        
        # Normalize to base year
        if self.base_year in period_indices:
            base_value = period_indices[self.base_year]
            period_indices = {
                year: value / base_value 
                for year, value in period_indices.items()
            }
        else:
            # Use first available year as base
            first_year = min(period_indices.keys())
            base_value = period_indices[first_year]
            period_indices = {
                year: value / base_value 
                for year, value in period_indices.items()
            }
            logger.warning(f"Base year {self.base_year} not available, "
                         f"using {first_year} as base")
        
        return HPIIndex(
            index_values=period_indices,
            entity_id=cbsa_code,
            entity_type='cbsa',
            base_period=self.base_year,
            metadata={
                'weight_type': str(weight_type),
                'n_tracts': len(census_tracts),
                'n_pairs': len(repeat_sales_pairs),
                'start_year': start_year,
                'end_year': end_year
            }
        )
    
    def build_indices_all_weights(self,
                                 transactions: pd.DataFrame,
                                 census_tracts: List[CensusTract],
                                 start_year: int = 1989,
                                 end_year: int = 2021,
                                 additional_data: Optional[Dict[str, pd.DataFrame]] = None
                                 ) -> Dict[str, HPIIndex]:
        """Build city-level indices for all weight types.
        
        Args:
            transactions: Transaction data
            census_tracts: List of census tracts
            start_year: Starting year
            end_year: Ending year
            additional_data: Dict with keys 'value', 'upb' containing DataFrames
            
        Returns:
            Dictionary mapping weight types to indices
        """
        indices = {}
        
        # Build for each weight type
        for weight_type in WeightType:
            logger.info(f"Building index for weight type: {weight_type.value}")
            
            # Get appropriate additional data
            add_data = None
            if additional_data:
                if weight_type == WeightType.VALUE:
                    add_data = additional_data.get('value')
                elif weight_type == WeightType.UPB:
                    add_data = additional_data.get('upb')
            
            try:
                index = self.build_annual_index(
                    transactions, census_tracts, weight_type,
                    start_year, end_year, add_data
                )
                indices[weight_type.value] = index
            except Exception as e:
                logger.error(f"Failed to build {weight_type.value} index: {e}")
        
        return indices
    
    def _aggregate_indices_for_period(self,
                                     supertract_indices: Dict[str, HPIIndex],
                                     weights: Dict[str, float],
                                     period: int) -> Optional[float]:
        """Aggregate supertract indices to city level for a specific period.
        
        Args:
            supertract_indices: Dictionary of supertract indices
            weights: Dictionary of weights
            period: Time period to aggregate
            
        Returns:
            Aggregated index value or None
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for supertract_id, index in supertract_indices.items():
            if supertract_id in weights and period in index.index_values:
                weight = weights[supertract_id]
                value = index.index_values[period]
                weighted_sum += weight * value
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return None
    
    def _create_default_index(self,
                             cbsa_code: str,
                             start_year: int,
                             end_year: int) -> HPIIndex:
        """Create default index with all values = 1.0.
        
        Args:
            cbsa_code: CBSA code
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            Default HPIIndex
        """
        index_values = {year: 1.0 for year in range(start_year, end_year + 1)}
        
        return HPIIndex(
            index_values=index_values,
            entity_id=cbsa_code,
            entity_type='cbsa',
            base_period=self.base_year,
            metadata={
                'is_default': True,
                'reason': 'Insufficient data for index construction'
            }
        )
    
    def calculate_pooled_appreciation(self,
                                    transactions: pd.DataFrame,
                                    census_tracts: List[CensusTract],
                                    period1: int,
                                    period2: int) -> float:
        """Calculate pooled appreciation rate between two periods.
        
        This implements the p̂_pooled(t,t-1) calculation from the PRD.
        
        Args:
            transactions: Transaction data
            census_tracts: List of census tracts
            period1: First period
            period2: Second period
            
        Returns:
            Pooled appreciation rate
        """
        # Build repeat sales pairs
        from ..models.repeat_sales import construct_repeat_sales_pairs
        pairs = construct_repeat_sales_pairs(transactions, apply_filters=True)
        
        # Filter for pairs involving these two periods
        period_pairs = pairs[
            ((pairs['sale1_year'] == period1) & (pairs['sale2_year'] == period2)) |
            ((pairs['sale1_year'] == period2) & (pairs['sale2_year'] == period1))
        ].copy()
        
        if period_pairs.empty:
            logger.warning(f"No pairs found between periods {period1} and {period2}")
            return 0.0
        
        # Calculate pooled appreciation using BMN regression
        from ..models.bmn_regression import BMNRegressor
        regressor = BMNRegressor()
        
        try:
            results = regressor.fit(period_pairs)
            # Extract the coefficient difference
            coeffs = results.coefficients
            
            # Find coefficients for the two periods
            coeff1 = coeffs.get(f'period_{period1}', 0.0)
            coeff2 = coeffs.get(f'period_{period2}', 0.0)
            
            # Appreciation is the difference
            return coeff2 - coeff1
            
        except Exception as e:
            logger.error(f"Error calculating pooled appreciation: {e}")
            return 0.0
    
    def export_results(self,
                      index: HPIIndex,
                      output_path: str,
                      format: str = 'csv'):
        """Export index results to file.
        
        Args:
            index: HPIIndex to export
            output_path: Path to output file
            format: Output format ('csv', 'parquet', 'excel')
        """
        df = index.to_dataframe()
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Exported index to {output_path}")
    
    def create_summary_statistics(self,
                                 indices: Dict[str, HPIIndex]) -> pd.DataFrame:
        """Create summary statistics for a set of indices.
        
        Args:
            indices: Dictionary of indices by weight type
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for weight_type, index in indices.items():
            # Get index series
            series = index.to_series()
            
            # Calculate statistics
            stats = {
                'weight_type': weight_type,
                'start_year': series.index.min(),
                'end_year': series.index.max(),
                'n_periods': len(series),
                'total_appreciation': series.iloc[-1] / series.iloc[0] - 1,
                'avg_annual_growth': (series.iloc[-1] / series.iloc[0]) ** 
                                   (1 / (series.index[-1] - series.index[0])) - 1,
                'volatility': series.pct_change().std(),
                'max_annual_change': series.pct_change().max(),
                'min_annual_change': series.pct_change().min()
            }
            
            summary_data.append(stats)
        
        return pd.DataFrame(summary_data)