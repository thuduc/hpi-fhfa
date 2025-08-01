"""House Price Index construction at tract/supertract level."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from ..models.bmn_regression import BMNRegressor, BMNResults
from ..models.repeat_sales import construct_repeat_sales_pairs
from ..geography.supertract import Supertract
from ..config.constants import BASE_YEAR
import logging

logger = logging.getLogger(__name__)


@dataclass
class HPIIndex:
    """House Price Index with metadata and methods.
    
    Attributes:
        index_values: Dictionary mapping periods to index values
        entity_id: ID of the geographic entity (tract/supertract)
        entity_type: Type of entity ('tract' or 'supertract')
        base_period: Base period for index (default BASE_YEAR)
        metadata: Additional index properties
    """
    
    index_values: Dict[int, float]
    entity_id: str
    entity_type: str
    base_period: int = BASE_YEAR
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize index."""
        if not self.index_values:
            raise ValueError("Index must have at least one value")
        
        # Ensure base period has value 1.0
        if self.base_period in self.index_values:
            if abs(self.index_values[self.base_period] - 1.0) > 1e-6:
                logger.warning(f"Normalizing index to base period {self.base_period}")
                self.normalize_to_base()
        else:
            logger.warning(f"Base period {self.base_period} not in index")
    
    def normalize_to_base(self, new_base: Optional[int] = None):
        """Normalize index to specified base period = 1.0.
        
        Args:
            new_base: New base period (uses current base if None)
        """
        if new_base:
            self.base_period = new_base
        
        if self.base_period not in self.index_values:
            raise ValueError(f"Base period {self.base_period} not in index")
        
        base_value = self.index_values[self.base_period]
        if base_value == 0:
            raise ValueError("Cannot normalize with base value of 0")
        
        # Normalize all values
        self.index_values = {
            period: value / base_value 
            for period, value in self.index_values.items()
        }
    
    def get_appreciation_rate(self, period1: int, period2: int) -> float:
        """Calculate appreciation rate between two periods.
        
        Args:
            period1: Starting period
            period2: Ending period
            
        Returns:
            Appreciation rate (e.g., 0.05 for 5% appreciation)
        """
        if period1 not in self.index_values:
            raise ValueError(f"Period {period1} not in index")
        if period2 not in self.index_values:
            raise ValueError(f"Period {period2} not in index")
        
        return (self.index_values[period2] / self.index_values[period1]) - 1.0
    
    def get_cagr(self, period1: int, period2: int) -> float:
        """Calculate compound annual growth rate between two periods.
        
        Args:
            period1: Starting period
            period2: Ending period
            
        Returns:
            CAGR (e.g., 0.05 for 5% annual growth)
        """
        if period1 >= period2:
            raise ValueError("Period2 must be after period1")
        
        appreciation = self.get_appreciation_rate(period1, period2)
        years = period2 - period1
        
        return (1 + appreciation) ** (1 / years) - 1
    
    def to_series(self) -> pd.Series:
        """Convert index to pandas Series."""
        return pd.Series(self.index_values, name=self.entity_id).sort_index()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert index to DataFrame with metadata."""
        df = pd.DataFrame({
            'period': list(self.index_values.keys()),
            'index_value': list(self.index_values.values()),
            'entity_id': self.entity_id,
            'entity_type': self.entity_type
        })
        
        # Add metadata columns
        for key, value in self.metadata.items():
            df[key] = value
        
        return df.sort_values('period')


class IndexBuilder:
    """Build house price indices at tract/supertract level."""
    
    def __init__(self, base_year: int = BASE_YEAR):
        """Initialize index builder.
        
        Args:
            base_year: Base year for index construction
        """
        self.base_year = base_year
        self.bmn_regressor = BMNRegressor()
        logger.info(f"Initialized IndexBuilder with base_year={base_year}")
    
    def build_index_for_entity(self,
                              entity: Supertract,
                              repeat_sales_pairs: pd.DataFrame,
                              start_period: int,
                              end_period: int) -> Optional[HPIIndex]:
        """Build index for a single geographic entity.
        
        Args:
            entity: Supertract to build index for
            repeat_sales_pairs: All repeat sales pairs
            start_period: Starting period for index
            end_period: Ending period for index
            
        Returns:
            HPIIndex or None if insufficient data
        """
        # Filter pairs for this entity
        entity_pairs = self._filter_pairs_for_entity(entity, repeat_sales_pairs)
        
        if entity_pairs.empty:
            logger.warning(f"No pairs found for entity {entity.supertract_id}")
            return None
        
        # Check if we have enough data
        unique_periods = pd.concat([
            entity_pairs['period_1'],
            entity_pairs['period_2']
        ]).unique()
        
        if len(unique_periods) < 2:
            logger.warning(f"Insufficient periods for entity {entity.supertract_id}")
            return None
        
        try:
            # Fit BMN regression
            bmn_results = self.bmn_regressor.fit(entity_pairs)
            
            # Build index from coefficients
            index_values = self._build_index_from_bmn(
                bmn_results, start_period, end_period
            )
            
            # Create HPIIndex object
            return HPIIndex(
                index_values=index_values,
                entity_id=entity.supertract_id,
                entity_type='supertract',
                base_period=self.base_year,
                metadata={
                    'n_tracts': len(entity.component_tracts),
                    'n_pairs': len(entity_pairs),
                    'r_squared': bmn_results.r_squared,
                    'periods_covered': len(unique_periods)
                }
            )
            
        except Exception as e:
            logger.error(f"Error building index for {entity.supertract_id}: {e}")
            return None
    
    def _build_index_from_bmn(self, 
                             bmn_results: 'BMNResults',
                             start_period: int,
                             end_period: int) -> Dict[int, float]:
        """Build index values from BMN regression results.
        
        Args:
            bmn_results: Results from BMN regression
            start_period: Start year
            end_period: End year
            
        Returns:
            Dictionary mapping periods to index values
        """
        # Get coefficients from results
        coefficients = bmn_results.coefficients
        
        # Create index values for requested period range
        index_values = {}
        
        # BMN coefficients represent log price levels
        # Convert to index form (exp of coefficients)
        for i, year in enumerate(range(start_period, end_period + 1)):
            if i < len(coefficients):
                # Index = exp(coefficient)
                index_values[year] = np.exp(coefficients[i])
            else:
                # If we don't have a coefficient for this year, use last available
                if coefficients.size > 0:
                    index_values[year] = np.exp(coefficients[-1])
                else:
                    index_values[year] = 1.0
        
        # Normalize to base year if present
        if self.base_year in index_values:
            base_value = index_values[self.base_year]
            index_values = {year: val / base_value for year, val in index_values.items()}
        
        return index_values
    
    def build_indices_for_cbsa(self,
                              supertracts: List[Supertract],
                              repeat_sales_pairs: pd.DataFrame,
                              start_period: int,
                              end_period: int) -> Dict[str, HPIIndex]:
        """Build indices for all supertracts in a CBSA.
        
        Args:
            supertracts: List of supertracts in the CBSA
            repeat_sales_pairs: All repeat sales pairs
            start_period: Starting period for indices
            end_period: Ending period for indices
            
        Returns:
            Dictionary mapping supertract IDs to indices
        """
        indices = {}
        
        logger.info(f"Building indices for {len(supertracts)} supertracts")
        
        for i, supertract in enumerate(supertracts):
            if i % 10 == 0 and i > 0:
                logger.debug(f"Processed {i}/{len(supertracts)} supertracts")
            
            index = self.build_index_for_entity(
                supertract, repeat_sales_pairs, start_period, end_period
            )
            
            if index:
                indices[supertract.supertract_id] = index
            else:
                logger.warning(f"Failed to build index for {supertract.supertract_id}")
        
        logger.info(f"Successfully built {len(indices)} indices")
        return indices
    
    def _filter_pairs_for_entity(self,
                                entity: Supertract,
                                repeat_sales_pairs: pd.DataFrame) -> pd.DataFrame:
        """Filter repeat sales pairs for a specific entity.
        
        Args:
            entity: Supertract to filter for
            repeat_sales_pairs: All repeat sales pairs
            
        Returns:
            Filtered DataFrame
        """
        # Get all tract codes in the supertract
        tract_codes = entity.tract_codes
        
        # Filter pairs where census tract is in the supertract
        return repeat_sales_pairs[
            repeat_sales_pairs['census_tract'].isin(tract_codes)
        ].copy()
    
    
    def calculate_chained_index(self,
                               period_indices: List[Tuple[int, int, float]],
                               base_period: int) -> Dict[int, float]:
        """Calculate chained index from period-to-period appreciation rates.
        
        Args:
            period_indices: List of (period1, period2, appreciation) tuples
            base_period: Base period for index
            
        Returns:
            Dictionary mapping periods to index values
        """
        # Sort by first period
        period_indices.sort(key=lambda x: x[0])
        
        # Build index dictionary
        index_values = {base_period: 1.0}
        
        for period1, period2, appreciation in period_indices:
            if period1 in index_values:
                # Calculate forward
                index_values[period2] = index_values[period1] * (1 + appreciation)
            elif period2 in index_values:
                # Calculate backward
                index_values[period1] = index_values[period2] / (1 + appreciation)
        
        return index_values
    
    def merge_indices(self,
                     indices: List[HPIIndex],
                     weights: Optional[List[float]] = None) -> HPIIndex:
        """Merge multiple indices with optional weights.
        
        Args:
            indices: List of HPIIndex objects
            weights: Optional weights for each index
            
        Returns:
            Merged HPIIndex
        """
        if not indices:
            raise ValueError("No indices to merge")
        
        # Validate weights first if provided
        if weights is not None:
            if len(weights) != len(indices):
                raise ValueError("Number of weights must match number of indices")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
        
        if len(indices) == 1:
            return indices[0]
        
        # Default to equal weights
        if weights is None:
            weights = [1.0 / len(indices)] * len(indices)
        
        # Get all unique periods
        all_periods = set()
        for index in indices:
            all_periods.update(index.index_values.keys())
        
        # Calculate weighted average for each period
        merged_values = {}
        for period in sorted(all_periods):
            weighted_sum = 0.0
            total_weight = 0.0
            
            for index, weight in zip(indices, weights):
                if period in index.index_values:
                    weighted_sum += index.index_values[period] * weight
                    total_weight += weight
            
            if total_weight > 0:
                merged_values[period] = weighted_sum / total_weight
        
        # Create merged index
        entity_ids = [idx.entity_id for idx in indices]
        return HPIIndex(
            index_values=merged_values,
            entity_id=f"merged_{len(indices)}",
            entity_type='merged',
            base_period=indices[0].base_period,
            metadata={
                'source_entities': entity_ids,
                'merge_weights': weights,
                'n_sources': len(indices)
            }
        )