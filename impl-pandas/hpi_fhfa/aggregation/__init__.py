"""Index aggregation module for HPI-FHFA implementation.

This module provides functionality for:
- Weight calculation for different aggregation schemes
- Tract-level index construction
- City-level index aggregation
- Weighted index combination
"""

from .weights import WeightCalculator, WeightType
from .index_builder import IndexBuilder, HPIIndex
from .city_level import CityLevelIndexBuilder

__all__ = [
    'WeightCalculator',
    'WeightType',
    'IndexBuilder',
    'HPIIndex',
    'CityLevelIndexBuilder'
]