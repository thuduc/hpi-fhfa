"""Numba-accelerated functions for performance-critical operations."""

import numpy as np
from typing import Tuple, Optional

# Check if numba is available
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define dummy decorators if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, parallel=True, cache=True)
def fast_log_diff(prices1: np.ndarray, prices2: np.ndarray) -> np.ndarray:
    """Calculate log price differences efficiently.
    
    Args:
        prices1: First period prices
        prices2: Second period prices
        
    Returns:
        Array of log price differences
    """
    n = len(prices1)
    result = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        if prices1[i] > 0 and prices2[i] > 0:
            result[i] = np.log(prices2[i]) - np.log(prices1[i])
        else:
            result[i] = np.nan
            
    return result


@jit(nopython=True, cache=True)
def fast_design_matrix(period1: np.ndarray, 
                      period2: np.ndarray,
                      n_periods: int,
                      normalize_first: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build BMN regression design matrix efficiently.
    
    Args:
        period1: First period indices
        period2: Second period indices  
        n_periods: Total number of periods
        normalize_first: Whether to drop first period
        
    Returns:
        Tuple of (row_indices, col_indices, values) for sparse matrix
    """
    n_obs = len(period1)
    
    if normalize_first:
        n_cols = n_periods - 1
    else:
        n_cols = n_periods
        
    # Pre-allocate arrays (max size is 2 * n_obs)
    max_entries = 2 * n_obs
    row_indices = np.empty(max_entries, dtype=np.int32)
    col_indices = np.empty(max_entries, dtype=np.int32)
    values = np.empty(max_entries, dtype=np.float64)
    
    entry_count = 0
    
    for i in range(n_obs):
        p1 = period1[i]
        p2 = period2[i]
        
        # Add -1 for first period
        if normalize_first:
            if p1 > 0:  # Skip first period (index 0)
                row_indices[entry_count] = i
                col_indices[entry_count] = p1 - 1
                values[entry_count] = -1.0
                entry_count += 1
        else:
            row_indices[entry_count] = i
            col_indices[entry_count] = p1
            values[entry_count] = -1.0
            entry_count += 1
            
        # Add +1 for second period
        if normalize_first:
            if p2 > 0:  # Skip first period (index 0)
                row_indices[entry_count] = i
                col_indices[entry_count] = p2 - 1
                values[entry_count] = 1.0
                entry_count += 1
        else:
            row_indices[entry_count] = i
            col_indices[entry_count] = p2
            values[entry_count] = 1.0
            entry_count += 1
    
    # Return only the filled portion
    return (row_indices[:entry_count], 
            col_indices[:entry_count], 
            values[:entry_count])


@jit(nopython=True, parallel=True, cache=True)
def fast_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Calculate weighted mean efficiently.
    
    Args:
        values: Array of values
        weights: Array of weights
        
    Returns:
        Weighted mean
    """
    if len(values) == 0:
        return np.nan
        
    weighted_sum = 0.0
    weight_sum = 0.0
    
    for i in prange(len(values)):
        if not np.isnan(values[i]) and not np.isnan(weights[i]):
            weighted_sum += values[i] * weights[i]
            weight_sum += weights[i]
    
    if weight_sum > 0:
        return weighted_sum / weight_sum
    else:
        return np.nan


@jit(nopython=True, parallel=True, cache=True)
def fast_distance_calc(lat1: np.ndarray, lon1: np.ndarray,
                      lat2: float, lon2: float) -> np.ndarray:
    """Calculate haversine distances efficiently.
    
    Args:
        lat1: Array of latitudes
        lon1: Array of longitudes
        lat2: Single latitude
        lon2: Single longitude
        
    Returns:
        Array of distances in kilometers
    """
    n = len(lat1)
    distances = np.empty(n, dtype=np.float64)
    
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert to radians
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    for i in prange(n):
        lat1_rad = np.radians(lat1[i])
        lon1_rad = np.radians(lon1[i])
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distances[i] = R * c
        
    return distances


@jit(nopython=True, cache=True)
def fast_period_mapping(dates: np.ndarray, base_year: int = 1989) -> np.ndarray:
    """Map dates to period indices efficiently.
    
    Args:
        dates: Array of years
        base_year: Base year for indexing
        
    Returns:
        Array of period indices
    """
    n = len(dates)
    periods = np.empty(n, dtype=np.int32)
    
    for i in range(n):
        periods[i] = dates[i] - base_year
        
    return periods


@jit(nopython=True, parallel=True, cache=True)
def fast_zscore_filter(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Apply z-score filtering efficiently.
    
    Args:
        values: Array of values to filter
        threshold: Z-score threshold
        
    Returns:
        Boolean mask of valid values
    """
    n = len(values)
    mask = np.ones(n, dtype=np.bool_)
    
    # Calculate mean and std
    mean = np.mean(values)
    std = np.std(values)
    
    if std > 0:
        for i in prange(n):
            z_score = abs((values[i] - mean) / std)
            if z_score > threshold:
                mask[i] = False
                
    return mask


def is_numba_available() -> bool:
    """Check if Numba is available and working."""
    return NUMBA_AVAILABLE