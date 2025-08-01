"""Geographic processing module for HPI-FHFA implementation.

This module provides functionality for:
- Census tract data structures and boundaries
- Geographic distance calculations
- Dynamic supertract aggregation
- Spatial operations for index construction
"""

from .census_tract import CensusTract
from .distance import (
    calculate_centroid_distance,
    calculate_haversine_distance,
    find_nearest_neighbors
)
from .supertract import Supertract, SupertractAlgorithm

__all__ = [
    'CensusTract',
    'Supertract',
    'SupertractAlgorithm',
    'calculate_centroid_distance',
    'calculate_haversine_distance',
    'find_nearest_neighbors'
]