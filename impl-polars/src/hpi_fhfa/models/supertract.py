"""Supertract algorithm for dynamic aggregation of census tracts."""

import polars as pl
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import structlog

from ..config.constants import MIN_HALF_PAIRS
from ..utils.exceptions import ProcessingError

logger = structlog.get_logger()


@dataclass
class Supertract:
    """Represents a supertract (aggregation of census tracts)."""
    id: str
    period: int
    component_tracts: Set[str]
    total_half_pairs: int
    centroid_lat: float
    centroid_lon: float


class SupertractBuilder:
    """Build supertracts by dynamically aggregating census tracts.
    
    The algorithm ensures each supertract has at least MIN_HALF_PAIRS
    observations in both the current period and previous period by
    merging nearby tracts based on centroid distance.
    """
    
    def __init__(self, geographic_df: pl.DataFrame):
        """Initialize builder with geographic data.
        
        Args:
            geographic_df: DataFrame with tract geographic info
                Must contain: tract_id, centroid_lat, centroid_lon
        """
        self.geographic_df = geographic_df
        self.distance_matrix = self._calculate_distance_matrix()
        self.tract_locations = self._create_location_lookup()
        
    def _create_location_lookup(self) -> Dict[str, Tuple[float, float]]:
        """Create lookup of tract locations."""
        return {
            row["tract_id"]: (row["centroid_lat"], row["centroid_lon"])
            for row in self.geographic_df.iter_rows(named=True)
        }
    
    def _calculate_distance_matrix(self) -> Dict[Tuple[str, str], float]:
        """Pre-calculate distances between all tract pairs.
        
        Uses Haversine formula for geographic distance.
        """
        logger.info("Calculating distance matrix for tracts")
        
        tracts = self.geographic_df["tract_id"].to_list()
        lats = self.geographic_df["centroid_lat"].to_numpy()
        lons = self.geographic_df["centroid_lon"].to_numpy()
        
        distances = {}
        
        for i, tract1 in enumerate(tracts):
            for j, tract2 in enumerate(tracts):
                if i < j:  # Only calculate upper triangle
                    dist = self._haversine_distance(
                        lats[i], lons[i], lats[j], lons[j]
                    )
                    distances[(tract1, tract2)] = dist
                    distances[(tract2, tract1)] = dist  # Symmetric
        
        logger.info(f"Calculated {len(distances)} tract pair distances")
        return distances
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula.
        
        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth radius in km
        
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def build_supertracts(
        self,
        half_pairs_df: pl.DataFrame,
        period: int
    ) -> List[Supertract]:
        """Build supertracts for a given period.
        
        Args:
            half_pairs_df: DataFrame with half-pairs counts by tract and period
                Must contain: tract_id, period, half_pairs_count
            period: The period to build supertracts for
            
        Returns:
            List of Supertract objects
        """
        logger.info(f"Building supertracts for period {period}")
        
        # Get half-pairs for current and previous period
        current_counts = self._get_period_counts(half_pairs_df, period)
        prev_counts = self._get_period_counts(half_pairs_df, period - 1)
        
        # Initialize each tract as its own supertract
        supertracts = {}
        for tract_id in current_counts.keys() | prev_counts.keys():
            supertracts[tract_id] = Supertract(
                id=f"S{period}_{tract_id}",
                period=period,
                component_tracts={tract_id},
                total_half_pairs=min(
                    current_counts.get(tract_id, 0),
                    prev_counts.get(tract_id, 0)
                ),
                centroid_lat=self.tract_locations[tract_id][0],
                centroid_lon=self.tract_locations[tract_id][1]
            )
        
        # Merge tracts until all meet minimum threshold
        merged_count = 0
        max_iterations = len(supertracts) * 2  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Find supertracts below threshold
            below_threshold = [
                st for st in supertracts.values()
                if st.total_half_pairs < MIN_HALF_PAIRS
            ]
            
            if not below_threshold:
                break
            
            # If only one supertract and no others to merge with, break
            if len(supertracts) == 1:
                logger.warning(
                    f"Single supertract with {list(supertracts.values())[0].total_half_pairs} "
                    f"half-pairs (below threshold of {MIN_HALF_PAIRS})"
                )
                break
            
            merges_made = 0
            # Merge each below-threshold supertract with nearest neighbor
            for supertract in below_threshold:
                if supertract.id not in supertracts:
                    continue  # Already merged
                
                nearest_id = self._find_nearest_supertract(
                    supertract, supertracts, period
                )
                
                if nearest_id:
                    self._merge_supertracts(
                        supertracts, supertract.id, nearest_id,
                        current_counts, prev_counts, period
                    )
                    merged_count += 1
                    merges_made += 1
            
            # If no merges were made in this iteration, break
            if merges_made == 0:
                logger.warning(
                    f"No merges possible for {len(below_threshold)} "
                    f"supertracts below threshold"
                )
                break
        
        logger.info(
            f"Built {len(supertracts)} supertracts for period {period}",
            merged_count=merged_count
        )
        
        return list(supertracts.values())
    
    def _get_period_counts(
        self, 
        half_pairs_df: pl.DataFrame, 
        period: int
    ) -> Dict[str, int]:
        """Get half-pairs counts for a specific period."""
        period_df = half_pairs_df.filter(pl.col("period") == period)
        return {
            row["tract_id"]: row["half_pairs_count"]
            for row in period_df.iter_rows(named=True)
        }
    
    def _find_nearest_supertract(
        self,
        supertract: Supertract,
        all_supertracts: Dict[str, Supertract],
        period: int
    ) -> Optional[str]:
        """Find the nearest supertract to merge with."""
        min_distance = float('inf')
        nearest_id = None
        
        for other_id, other in all_supertracts.items():
            if other_id == supertract.id:
                continue
            
            # Calculate distance between supertract centroids
            distance = self._supertract_distance(supertract, other)
            
            if distance < min_distance:
                min_distance = distance
                nearest_id = other_id
        
        return nearest_id
    
    def _supertract_distance(self, st1: Supertract, st2: Supertract) -> float:
        """Calculate distance between two supertracts.
        
        Uses minimum distance between any pair of component tracts.
        """
        min_dist = float('inf')
        
        for tract1 in st1.component_tracts:
            for tract2 in st2.component_tracts:
                key = (tract1, tract2) if tract1 < tract2 else (tract2, tract1)
                if key in self.distance_matrix:
                    min_dist = min(min_dist, self.distance_matrix[key])
        
        return min_dist
    
    def _merge_supertracts(
        self,
        supertracts: Dict[str, Supertract],
        source_id: str,
        target_id: str,
        current_counts: Dict[str, int],
        prev_counts: Dict[str, int],
        period: int
    ) -> None:
        """Merge source supertract into target supertract."""
        source = supertracts[source_id]
        target = supertracts[target_id]
        
        # Combine component tracts
        target.component_tracts.update(source.component_tracts)
        
        # Recalculate total half-pairs
        current_total = sum(
            current_counts.get(tract, 0) for tract in target.component_tracts
        )
        prev_total = sum(
            prev_counts.get(tract, 0) for tract in target.component_tracts
        )
        target.total_half_pairs = min(current_total, prev_total)
        
        # Update centroid (weighted average by tract count)
        n_source = len(source.component_tracts)
        n_target = len(target.component_tracts) - n_source  # Before merge
        n_total = len(target.component_tracts)
        
        target.centroid_lat = (
            target.centroid_lat * n_target + source.centroid_lat * n_source
        ) / n_total
        target.centroid_lon = (
            target.centroid_lon * n_target + source.centroid_lon * n_source
        ) / n_total
        
        # Update ID to reflect merge
        target.id = f"S{period}_{'_'.join(sorted(target.component_tracts)[:3])}_etc"
        
        # Remove source supertract
        del supertracts[source_id]
    
    def create_supertract_mapping(
        self,
        supertracts: List[Supertract]
    ) -> pl.DataFrame:
        """Create mapping from tracts to supertracts.
        
        Args:
            supertracts: List of Supertract objects
            
        Returns:
            DataFrame with tract_id, supertract_id, period columns
        """
        mappings = []
        for supertract in supertracts:
            for tract_id in supertract.component_tracts:
                mappings.append({
                    "tract_id": tract_id,
                    "supertract_id": supertract.id,
                    "period": supertract.period,
                    "n_component_tracts": len(supertract.component_tracts),
                    "total_half_pairs": supertract.total_half_pairs
                })
        
        return pl.DataFrame(mappings)