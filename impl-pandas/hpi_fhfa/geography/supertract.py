"""Supertract aggregation for ensuring minimum data requirements."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from .census_tract import CensusTract
from .distance import find_nearest_neighbors, calculate_centroid_distance
from ..config.constants import MIN_HALF_PAIRS
import logging

logger = logging.getLogger(__name__)


@dataclass
class Supertract:
    """Dynamic aggregation of census tracts to meet minimum data requirements.
    
    Attributes:
        supertract_id: Unique identifier for the supertract
        component_tracts: List of census tracts in this supertract
        period: Time period for which this supertract is valid
        half_pairs_count: Number of half-pairs in this supertract
        centroid_lat: Weighted centroid latitude
        centroid_lon: Weighted centroid longitude
        metadata: Additional properties
    """
    
    supertract_id: str
    component_tracts: List[CensusTract]
    period: int
    half_pairs_count: int
    centroid_lat: float = 0.0
    centroid_lon: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived properties."""
        if not self.component_tracts:
            raise ValueError("Supertract must contain at least one tract")
        
        # Calculate weighted centroid if not provided
        if self.centroid_lat == 0.0 and self.centroid_lon == 0.0:
            self._calculate_centroid()
        
        # Ensure all tracts are from same CBSA
        cbsas = {tract.cbsa_code for tract in self.component_tracts}
        if len(cbsas) > 1:
            raise ValueError(f"Supertract contains tracts from multiple CBSAs: {cbsas}")
    
    def _calculate_centroid(self):
        """Calculate weighted centroid of component tracts."""
        # Weight by population if available, otherwise equal weights
        weights = []
        for tract in self.component_tracts:
            if tract.population:
                weights.append(tract.population)
            else:
                weights.append(1.0)
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate weighted centroid
        self.centroid_lat = sum(
            tract.centroid_lat * weight 
            for tract, weight in zip(self.component_tracts, weights)
        )
        self.centroid_lon = sum(
            tract.centroid_lon * weight 
            for tract, weight in zip(self.component_tracts, weights)
        )
    
    @property
    def tract_codes(self) -> List[str]:
        """Get list of component tract codes."""
        return [tract.tract_code for tract in self.component_tracts]
    
    @property
    def cbsa_code(self) -> str:
        """Get CBSA code (all component tracts must have same CBSA)."""
        return self.component_tracts[0].cbsa_code
    
    @property
    def is_single_tract(self) -> bool:
        """Check if this is a single tract (not aggregated)."""
        return len(self.component_tracts) == 1
    
    def get_aggregate_weight(self, weight_type: str) -> float:
        """Calculate aggregate weight for the supertract.
        
        Args:
            weight_type: Type of weight to aggregate
            
        Returns:
            Aggregated weight value
        """
        values = []
        for tract in self.component_tracts:
            value = tract.get_demographic_weight(weight_type)
            if value is not None:
                values.append(value)
        
        if not values:
            return 0.0
        
        # For share-based weights, use weighted average
        if weight_type in ['college', 'nonwhite']:
            # Weight by population if available
            populations = []
            for i, tract in enumerate(self.component_tracts):
                if i < len(values) and tract.population:
                    populations.append(tract.population)
            
            if populations and len(populations) == len(values):
                total_pop = sum(populations)
                return sum(v * p for v, p in zip(values, populations)) / total_pop
            else:
                return np.mean(values)
        else:
            # For count-based weights, sum
            return sum(values)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert supertract to dictionary representation."""
        return {
            'supertract_id': self.supertract_id,
            'tract_codes': self.tract_codes,
            'period': self.period,
            'half_pairs_count': self.half_pairs_count,
            'centroid_lat': self.centroid_lat,
            'centroid_lon': self.centroid_lon,
            'cbsa_code': self.cbsa_code,
            'n_tracts': len(self.component_tracts),
            **self.metadata
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (f"Supertract({self.supertract_id}, "
                f"tracts={len(self.component_tracts)}, "
                f"half_pairs={self.half_pairs_count})")
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"Supertract(id='{self.supertract_id}', "
                f"period={self.period}, "
                f"tracts={self.tract_codes}, "
                f"half_pairs={self.half_pairs_count})")


class SupertractAlgorithm:
    """Algorithm for dynamic tract aggregation to meet minimum data requirements."""
    
    def __init__(self, min_half_pairs: int = MIN_HALF_PAIRS,
                 max_merge_distance: float = 50.0,
                 prefer_adjacent: bool = True):
        """Initialize supertract algorithm.
        
        Args:
            min_half_pairs: Minimum number of half-pairs required
            max_merge_distance: Maximum distance for merging tracts (miles)
            prefer_adjacent: Whether to prefer adjacent tracts when merging
        """
        self.min_half_pairs = min_half_pairs
        self.max_merge_distance = max_merge_distance
        self.prefer_adjacent = prefer_adjacent
        logger.info(f"Initialized SupertractAlgorithm with min_half_pairs={min_half_pairs}")
    
    def build_supertracts(self, 
                         tracts: List[CensusTract],
                         repeat_sales_pairs: pd.DataFrame,
                         period: int) -> List[Supertract]:
        """Build supertracts for a specific time period.
        
        Args:
            tracts: List of census tracts in the CBSA
            repeat_sales_pairs: DataFrame of repeat sales pairs
            period: Time period (year) for supertract construction
            
        Returns:
            List of supertracts meeting minimum requirements
        """
        if not tracts:
            return []
        
        logger.info(f"Building supertracts for period {period} with {len(tracts)} tracts")
        
        # Calculate half-pairs per tract for this period
        tract_half_pairs = self._calculate_tract_half_pairs(
            tracts, repeat_sales_pairs, period
        )
        
        # Initialize each tract as its own supertract
        supertracts = []
        tract_to_supertract = {}  # Map tract code to current supertract
        
        for tract in tracts:
            half_pairs = tract_half_pairs.get(tract.tract_code, 0)
            
            st = Supertract(
                supertract_id=f"{tract.tract_code}_{period}",
                component_tracts=[tract],
                period=period,
                half_pairs_count=half_pairs
            )
            supertracts.append(st)
            tract_to_supertract[tract.tract_code] = st
        
        # Iteratively merge tracts that don't meet minimum
        changed = True
        iteration = 0
        while changed and iteration < 100:  # Prevent infinite loops
            changed = False
            iteration += 1
            
            # Find supertracts that need merging
            insufficient_supertracts = [
                st for st in supertracts 
                if st.half_pairs_count < self.min_half_pairs
            ]
            
            if not insufficient_supertracts:
                break
            
            logger.debug(f"Iteration {iteration}: {len(insufficient_supertracts)} "
                        f"supertracts below threshold")
            
            # Process each insufficient supertract
            for st in insufficient_supertracts:
                if st not in supertracts:  # May have been merged already
                    continue
                
                # Find best merge candidate
                merge_candidate = self._find_merge_candidate(
                    st, supertracts, tracts
                )
                
                if merge_candidate:
                    # Perform merge
                    logger.debug(f"Merging {st.supertract_id} with "
                               f"{merge_candidate.supertract_id}")
                    
                    merged = self._merge_supertracts(
                        st, merge_candidate, period, tract_half_pairs
                    )
                    
                    # Update structures
                    supertracts.remove(st)
                    supertracts.remove(merge_candidate)
                    supertracts.append(merged)
                    
                    # Update tract mapping
                    for tract_code in merged.tract_codes:
                        tract_to_supertract[tract_code] = merged
                    
                    changed = True
        
        # Log final statistics
        n_single = sum(1 for st in supertracts if st.is_single_tract)
        n_merged = len(supertracts) - n_single
        logger.info(f"Created {len(supertracts)} supertracts: "
                   f"{n_single} single tracts, {n_merged} merged")
        
        return supertracts
    
    def _calculate_tract_half_pairs(self,
                                   tracts: List[CensusTract],
                                   repeat_sales_pairs: pd.DataFrame,
                                   period: int) -> Dict[str, int]:
        """Calculate half-pairs per tract for a specific period.
        
        Args:
            tracts: List of census tracts
            repeat_sales_pairs: DataFrame of repeat sales pairs
            period: Time period (year)
            
        Returns:
            Dictionary mapping tract codes to half-pair counts
        """
        # Filter pairs involving this period
        period_pairs = repeat_sales_pairs[
            (repeat_sales_pairs['period_1'] == period) |
            (repeat_sales_pairs['period_2'] == period)
        ].copy()
        
        if period_pairs.empty:
            return {tract.tract_code: 0 for tract in tracts}
        
        # Count half-pairs per tract
        tract_codes = [tract.tract_code for tract in tracts]
        half_pairs_dict = {}
        
        for tract_code in tract_codes:
            # Count pairs where this tract appears in either period
            tract_pairs = period_pairs[
                period_pairs['census_tract'] == tract_code
            ]
            
            if not tract_pairs.empty:
                # Count half-pairs: for each pair, count 1 for each period involved
                # If period appears in period_1, count 1
                # If period appears in period_2, count 1
                count = 0
                count += (tract_pairs['period_1'] == period).sum()
                count += (tract_pairs['period_2'] == period).sum()
                half_pairs_dict[tract_code] = count
            else:
                half_pairs_dict[tract_code] = 0
        
        return half_pairs_dict
    
    def _find_merge_candidate(self,
                             target_supertract: Supertract,
                             all_supertracts: List[Supertract],
                             all_tracts: List[CensusTract]) -> Optional[Supertract]:
        """Find best candidate supertract to merge with.
        
        Args:
            target_supertract: Supertract needing merge
            all_supertracts: All current supertracts
            all_tracts: All census tracts (for distance calculations)
            
        Returns:
            Best merge candidate or None
        """
        # Get representative tract for distance calculations
        # Use the tract closest to supertract centroid
        target_centroid_tract = min(
            target_supertract.component_tracts,
            key=lambda t: calculate_centroid_distance(
                t,
                CensusTract(
                    tract_code="00000000000",  # Dummy
                    cbsa_code=t.cbsa_code,
                    state_code="00",
                    county_code="000",
                    tract_number="000000",
                    centroid_lat=target_supertract.centroid_lat,
                    centroid_lon=target_supertract.centroid_lon,
                    distance_to_cbd=0
                )
            )
        )
        
        # Find candidates
        candidates = []
        for st in all_supertracts:
            if st == target_supertract:
                continue
            
            # Must be in same CBSA
            if st.cbsa_code != target_supertract.cbsa_code:
                continue
            
            # Calculate minimum distance between any tracts
            min_distance = float('inf')
            for t1 in target_supertract.component_tracts:
                for t2 in st.component_tracts:
                    dist = calculate_centroid_distance(t1, t2)
                    min_distance = min(min_distance, dist)
            
            if min_distance <= self.max_merge_distance:
                # Score based on distance and resulting half-pairs
                combined_half_pairs = (target_supertract.half_pairs_count + 
                                     st.half_pairs_count)
                
                # Prefer merges that get us just above threshold
                if combined_half_pairs >= self.min_half_pairs:
                    excess = combined_half_pairs - self.min_half_pairs
                    score = 1.0 / (1.0 + excess) + 1.0 / (1.0 + min_distance)
                else:
                    score = combined_half_pairs / self.min_half_pairs
                
                candidates.append((st, min_distance, score))
        
        if not candidates:
            logger.warning(f"No merge candidates found for {target_supertract.supertract_id}")
            return None
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[0][0]
    
    def _merge_supertracts(self,
                          st1: Supertract,
                          st2: Supertract,
                          period: int,
                          tract_half_pairs: Dict[str, int]) -> Supertract:
        """Merge two supertracts into one.
        
        Args:
            st1: First supertract
            st2: Second supertract
            period: Time period
            tract_half_pairs: Half-pairs per tract
            
        Returns:
            Merged supertract
        """
        # Combine component tracts
        merged_tracts = st1.component_tracts + st2.component_tracts
        
        # Calculate total half-pairs
        total_half_pairs = sum(
            tract_half_pairs.get(tract.tract_code, 0)
            for tract in merged_tracts
        )
        
        # Create merged ID
        tract_codes = sorted([t.tract_code for t in merged_tracts])
        merged_id = f"ST_{tract_codes[0]}_{tract_codes[-1]}_{period}"
        
        return Supertract(
            supertract_id=merged_id,
            component_tracts=merged_tracts,
            period=period,
            half_pairs_count=total_half_pairs,
            metadata={
                'merged_from': [st1.supertract_id, st2.supertract_id],
                'n_merges': st1.metadata.get('n_merges', 0) + 
                           st2.metadata.get('n_merges', 0) + 1
            }
        )
    
    def build_supertracts_multi_period(self,
                                      tracts: List[CensusTract],
                                      repeat_sales_pairs: pd.DataFrame,
                                      periods: List[int]) -> Dict[int, List[Supertract]]:
        """Build supertracts for multiple time periods.
        
        Args:
            tracts: List of census tracts
            repeat_sales_pairs: DataFrame of repeat sales pairs
            periods: List of time periods
            
        Returns:
            Dictionary mapping periods to lists of supertracts
        """
        results = {}
        
        for period in periods:
            logger.info(f"Building supertracts for period {period}")
            supertracts = self.build_supertracts(tracts, repeat_sales_pairs, period)
            results[period] = supertracts
        
        return results