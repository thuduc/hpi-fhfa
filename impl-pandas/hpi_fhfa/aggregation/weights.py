"""Weight calculation methods for index aggregation."""

from enum import Enum
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from ..geography.census_tract import CensusTract
from ..geography.supertract import Supertract
import logging

logger = logging.getLogger(__name__)


class WeightType(Enum):
    """Types of weights supported for aggregation."""
    SAMPLE = "sample"  # Share of half-pairs
    VALUE = "value"    # Share of aggregate housing value
    UNIT = "unit"      # Share of housing units
    UPB = "upb"        # Share of unpaid principal balance
    COLLEGE = "college"  # Share of college-educated population
    NONWHITE = "nonwhite"  # Share of non-white population


class WeightCalculator:
    """Factory for calculating different types of aggregation weights."""
    
    @staticmethod
    def calculate_weights(weight_type: Union[str, WeightType],
                         entities: Union[List[CensusTract], List[Supertract]],
                         repeat_sales_data: Optional[pd.DataFrame] = None,
                         additional_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Calculate weights for aggregation based on specified type.
        
        Args:
            weight_type: Type of weight to calculate
            entities: List of census tracts or supertracts
            repeat_sales_data: DataFrame with repeat sales pairs (for sample weights)
            additional_data: Additional data needed for certain weight types
                            (e.g., housing values for VALUE weights)
        
        Returns:
            Dictionary mapping entity IDs to weights (sum to 1.0)
        """
        if not entities:
            return {}
        
        # Convert string to enum if needed
        if isinstance(weight_type, str):
            try:
                weight_type = WeightType(weight_type.lower())
            except ValueError:
                raise ValueError(f"Unknown weight type: {weight_type}")
        
        logger.info(f"Calculating {weight_type.value} weights for {len(entities)} entities")
        
        # Dispatch to appropriate method
        if weight_type == WeightType.SAMPLE:
            return WeightCalculator._calculate_sample_weights(entities, repeat_sales_data)
        elif weight_type == WeightType.VALUE:
            return WeightCalculator._calculate_value_weights(entities, additional_data)
        elif weight_type == WeightType.UNIT:
            return WeightCalculator._calculate_unit_weights(entities)
        elif weight_type == WeightType.UPB:
            return WeightCalculator._calculate_upb_weights(entities, additional_data)
        elif weight_type == WeightType.COLLEGE:
            return WeightCalculator._calculate_demographic_weights(entities, 'college')
        elif weight_type == WeightType.NONWHITE:
            return WeightCalculator._calculate_demographic_weights(entities, 'nonwhite')
        else:
            raise ValueError(f"Unsupported weight type: {weight_type}")
    
    @staticmethod
    def _calculate_sample_weights(entities: Union[List[CensusTract], List[Supertract]],
                                 repeat_sales_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calculate weights based on share of half-pairs.
        
        Args:
            entities: List of tracts or supertracts
            repeat_sales_data: DataFrame with repeat sales pairs
            
        Returns:
            Dictionary of weights
        """
        weights = {}
        
        # Handle supertracts
        if entities and isinstance(entities[0], Supertract):
            total_half_pairs = sum(st.half_pairs_count for st in entities)
            if total_half_pairs == 0:
                # Equal weights if no data
                weight = 1.0 / len(entities)
                for st in entities:
                    weights[st.supertract_id] = weight
            else:
                for st in entities:
                    weights[st.supertract_id] = st.half_pairs_count / total_half_pairs
        
        # Handle census tracts
        elif entities and isinstance(entities[0], CensusTract):
            if repeat_sales_data is None or repeat_sales_data.empty:
                # Equal weights if no data
                weight = 1.0 / len(entities)
                for tract in entities:
                    weights[tract.tract_code] = weight
            else:
                # Count half-pairs per tract
                tract_counts = {}
                for tract in entities:
                    tract_pairs = repeat_sales_data[
                        repeat_sales_data['census_tract'] == tract.tract_code
                    ]
                    # Each pair contributes 2 half-pairs (one for each period)
                    tract_counts[tract.tract_code] = len(tract_pairs) * 2
                
                total_count = sum(tract_counts.values())
                if total_count == 0:
                    # Equal weights if no pairs
                    weight = 1.0 / len(entities)
                    for tract in entities:
                        weights[tract.tract_code] = weight
                else:
                    for tract_code, count in tract_counts.items():
                        weights[tract_code] = count / total_count
        
        return weights
    
    @staticmethod
    def _calculate_value_weights(entities: Union[List[CensusTract], List[Supertract]],
                                additional_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calculate weights based on aggregate housing value.
        
        Args:
            entities: List of tracts or supertracts
            additional_data: DataFrame with 'tract_code' and 'aggregate_value' columns
            
        Returns:
            Dictionary of weights
        """
        if additional_data is None or additional_data.empty:
            logger.warning("No value data provided, using equal weights")
            weight = 1.0 / len(entities)
            if isinstance(entities[0], Supertract):
                return {st.supertract_id: weight for st in entities}
            else:
                return {tract.tract_code: weight for tract in entities}
        
        weights = {}
        total_value = 0.0
        
        # Calculate total value
        if isinstance(entities[0], Supertract):
            for st in entities:
                st_value = 0.0
                for tract in st.component_tracts:
                    tract_data = additional_data[
                        additional_data['tract_code'] == tract.tract_code
                    ]
                    if not tract_data.empty:
                        st_value += tract_data['aggregate_value'].iloc[0]
                weights[st.supertract_id] = st_value
                total_value += st_value
        else:
            for tract in entities:
                tract_data = additional_data[
                    additional_data['tract_code'] == tract.tract_code
                ]
                if not tract_data.empty:
                    value = tract_data['aggregate_value'].iloc[0]
                    weights[tract.tract_code] = value
                    total_value += value
                else:
                    weights[tract.tract_code] = 0.0
        
        # Normalize to sum to 1
        if total_value > 0:
            for key in weights:
                weights[key] /= total_value
        else:
            # Equal weights if no value data
            weight = 1.0 / len(entities)
            for key in weights:
                weights[key] = weight
        
        return weights
    
    @staticmethod
    def _calculate_unit_weights(entities: Union[List[CensusTract], List[Supertract]]) -> Dict[str, float]:
        """Calculate weights based on housing units.
        
        Args:
            entities: List of tracts or supertracts
            
        Returns:
            Dictionary of weights
        """
        weights = {}
        total_units = 0
        
        if isinstance(entities[0], Supertract):
            for st in entities:
                st_units = sum(
                    tract.housing_units or 0 
                    for tract in st.component_tracts
                )
                weights[st.supertract_id] = st_units
                total_units += st_units
        else:
            for tract in entities:
                units = tract.housing_units or 0
                weights[tract.tract_code] = units
                total_units += units
        
        # Normalize
        if total_units > 0:
            for key in weights:
                weights[key] /= total_units
        else:
            # Equal weights if no unit data
            weight = 1.0 / len(entities)
            for key in weights:
                weights[key] = weight
        
        return weights
    
    @staticmethod
    def _calculate_upb_weights(entities: Union[List[CensusTract], List[Supertract]],
                              additional_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calculate weights based on unpaid principal balance.
        
        Args:
            entities: List of tracts or supertracts
            additional_data: DataFrame with 'tract_code' and 'upb' columns
            
        Returns:
            Dictionary of weights
        """
        if additional_data is None or additional_data.empty:
            logger.warning("No UPB data provided, using equal weights")
            weight = 1.0 / len(entities)
            if isinstance(entities[0], Supertract):
                return {st.supertract_id: weight for st in entities}
            else:
                return {tract.tract_code: weight for tract in entities}
        
        weights = {}
        total_upb = 0.0
        
        # Calculate total UPB
        if isinstance(entities[0], Supertract):
            for st in entities:
                st_upb = 0.0
                for tract in st.component_tracts:
                    tract_data = additional_data[
                        additional_data['tract_code'] == tract.tract_code
                    ]
                    if not tract_data.empty:
                        st_upb += tract_data['upb'].iloc[0]
                weights[st.supertract_id] = st_upb
                total_upb += st_upb
        else:
            for tract in entities:
                tract_data = additional_data[
                    additional_data['tract_code'] == tract.tract_code
                ]
                if not tract_data.empty:
                    upb = tract_data['upb'].iloc[0]
                    weights[tract.tract_code] = upb
                    total_upb += upb
                else:
                    weights[tract.tract_code] = 0.0
        
        # Normalize
        if total_upb > 0:
            for key in weights:
                weights[key] /= total_upb
        else:
            # Equal weights if no UPB data
            weight = 1.0 / len(entities)
            for key in weights:
                weights[key] = weight
        
        return weights
    
    @staticmethod
    def _calculate_demographic_weights(entities: Union[List[CensusTract], List[Supertract]],
                                     demographic_type: str) -> Dict[str, float]:
        """Calculate weights based on demographic characteristics.
        
        Args:
            entities: List of tracts or supertracts
            demographic_type: 'college' or 'nonwhite'
            
        Returns:
            Dictionary of weights
        """
        weights = {}
        total_weighted_pop = 0.0
        
        if isinstance(entities[0], Supertract):
            for st in entities:
                # Calculate population-weighted average
                total_pop = 0
                weighted_sum = 0.0
                
                for tract in st.component_tracts:
                    if tract.population:
                        share = tract.get_demographic_weight(demographic_type)
                        if share is not None:
                            weighted_sum += tract.population * share
                            total_pop += tract.population
                
                if total_pop > 0:
                    st_weighted = weighted_sum
                    weights[st.supertract_id] = st_weighted
                    total_weighted_pop += st_weighted
                else:
                    weights[st.supertract_id] = 0.0
        else:
            for tract in entities:
                if tract.population:
                    share = tract.get_demographic_weight(demographic_type)
                    if share is not None:
                        weighted = tract.population * share
                        weights[tract.tract_code] = weighted
                        total_weighted_pop += weighted
                    else:
                        weights[tract.tract_code] = 0.0
                else:
                    weights[tract.tract_code] = 0.0
        
        # Normalize
        if total_weighted_pop > 0:
            for key in weights:
                weights[key] /= total_weighted_pop
        else:
            # Equal weights if no demographic data
            weight = 1.0 / len(entities)
            for key in weights:
                weights[key] = weight
        
        return weights
    
    @staticmethod
    def validate_weights(weights: Dict[str, float], tolerance: float = 1e-6) -> bool:
        """Validate that weights sum to 1.0 within tolerance.
        
        Args:
            weights: Dictionary of weights
            tolerance: Acceptable deviation from 1.0
            
        Returns:
            True if weights are valid
        """
        if not weights:
            return False
        
        total = sum(weights.values())
        is_valid = abs(total - 1.0) < tolerance
        
        if not is_valid:
            logger.warning(f"Weights sum to {total}, not 1.0")
        
        return is_valid
    
    @staticmethod
    def combine_weights(weights_list: List[Dict[str, float]],
                       combination_weights: Optional[List[float]] = None) -> Dict[str, float]:
        """Combine multiple weight sets with optional meta-weights.
        
        Args:
            weights_list: List of weight dictionaries
            combination_weights: Weights for combining (must sum to 1.0)
            
        Returns:
            Combined weight dictionary
        """
        if not weights_list:
            return {}
        
        if len(weights_list) == 1:
            return weights_list[0].copy()
        
        # Default to equal combination weights
        if combination_weights is None:
            combination_weights = [1.0 / len(weights_list)] * len(weights_list)
        
        # Validate combination weights
        if len(combination_weights) != len(weights_list):
            raise ValueError("Combination weights must match number of weight sets")
        
        if abs(sum(combination_weights) - 1.0) > 1e-6:
            raise ValueError("Combination weights must sum to 1.0")
        
        # Get all unique keys
        all_keys = set()
        for weights in weights_list:
            all_keys.update(weights.keys())
        
        # Combine weights
        combined = {}
        for key in all_keys:
            combined_weight = 0.0
            for weights, comb_weight in zip(weights_list, combination_weights):
                combined_weight += weights.get(key, 0.0) * comb_weight
            combined[key] = combined_weight
        
        # Renormalize to ensure sum to 1
        total = sum(combined.values())
        if total > 0:
            for key in combined:
                combined[key] /= total
        
        return combined