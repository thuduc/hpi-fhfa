"""
Geographic distance calculations for spatial analysis.

This module provides functions for calculating distances between geographic points
and identifying spatial relationships using various distance metrics.
"""

import logging
from typing import Tuple, Optional, Union, List
from math import radians, cos, sin, asin, sqrt, atan2

import polars as pl
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

logger = logging.getLogger(__name__)


def haversine_distance(
    lat1: float, 
    lon1: float, 
    lat2: float, 
    lon2: float,
    unit: str = "km"
) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point
        unit: Unit for distance ('km' or 'miles')
        
    Returns:
        Distance between points
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Earth radius
    r = 6371 if unit == "km" else 3959  # km or miles
    
    return c * r


def vincenty_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    unit: str = "km"
) -> float:
    """
    Calculate distance using Vincenty's formulae (more accurate for short distances).
    
    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point
        unit: Unit for distance ('km' or 'miles')
        
    Returns:
        Distance between points
    """
    # WGS-84 ellipsoid parameters
    a = 6378137  # Semi-major axis in meters
    f = 1 / 298.257223563  # Flattening
    b = (1 - f) * a  # Semi-minor axis
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Calculate reduced latitudes
    U1 = atan2((1 - f) * sin(lat1), cos(lat1))
    U2 = atan2((1 - f) * sin(lat2), cos(lat2))
    
    L = lon2 - lon1
    Lambda = L
    
    sinU1 = sin(U1)
    cosU1 = cos(U1)
    sinU2 = sin(U2)
    cosU2 = cos(U2)
    
    # Iterate until convergence
    for _ in range(100):
        sinLambda = sin(Lambda)
        cosLambda = cos(Lambda)
        
        sinSigma = sqrt((cosU2 * sinLambda) ** 2 +
                       (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)
        
        if sinSigma == 0:
            return 0  # Points are identical
            
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = atan2(sinSigma, cosSigma)
        
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha ** 2
        
        if cosSqAlpha == 0:
            cos2SigmaM = 0
        else:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
            
        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        
        Lambda_prev = Lambda
        Lambda = L + (1 - C) * f * sinAlpha * (
            sigma + C * sinSigma * (
                cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM ** 2)
            )
        )
        
        # Check for convergence
        if abs(Lambda - Lambda_prev) < 1e-12:
            break
    
    uSq = cosSqAlpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    
    deltaSigma = B * sinSigma * (
        cos2SigmaM + B / 4 * (
            cosSigma * (-1 + 2 * cos2SigmaM ** 2) -
            B / 6 * cos2SigmaM * (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)
        )
    )
    
    s = b * A * (sigma - deltaSigma)  # Distance in meters
    
    # Convert to requested unit
    if unit == "km":
        return s / 1000
    elif unit == "miles":
        return s / 1609.344
    else:
        return s


def calculate_distance_matrix_polars(
    df: pl.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    method: str = "haversine",
    unit: str = "km"
) -> np.ndarray:
    """
    Calculate pairwise distance matrix for locations in a Polars DataFrame.
    
    Args:
        df: Polars DataFrame with coordinates
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        method: Distance calculation method ('haversine' or 'vincenty')
        unit: Unit for distance
        
    Returns:
        NumPy array with pairwise distances
    """
    # Extract coordinates
    coords = df.select([lat_col, lon_col]).to_numpy()
    n_points = len(coords)
    
    # Initialize distance matrix
    distances = np.zeros((n_points, n_points))
    
    # Calculate distances
    distance_func = haversine_distance if method == "haversine" else vincenty_distance
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = distance_func(
                coords[i, 0], coords[i, 1],
                coords[j, 0], coords[j, 1],
                unit
            )
            distances[i, j] = dist
            distances[j, i] = dist
            
    return distances


def find_neighbors_within_distance(
    df: pl.DataFrame,
    target_lat: float,
    target_lon: float,
    max_distance: float,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    id_col: str = "property_id",
    method: str = "ball_tree",
    unit: str = "km"
) -> pl.DataFrame:
    """
    Find all points within a specified distance of a target location.
    
    Args:
        df: Polars DataFrame with coordinates
        target_lat: Target latitude
        target_lon: Target longitude
        max_distance: Maximum distance to search
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        id_col: Name of ID column
        method: Search method ('ball_tree' or 'brute_force')
        unit: Unit for distance
        
    Returns:
        Polars DataFrame with neighbors and distances
    """
    # Filter out null coordinates
    valid_df = df.filter(
        pl.col(lat_col).is_not_null() & 
        pl.col(lon_col).is_not_null()
    )
    
    if len(valid_df) == 0:
        return pl.DataFrame()
    
    if method == "ball_tree":
        # Convert to radians for BallTree
        coords_rad = np.radians(valid_df.select([lat_col, lon_col]).to_numpy())
        
        # Create BallTree
        tree = BallTree(coords_rad, metric='haversine')
        
        # Convert max distance to radians
        earth_radius = 6371 if unit == "km" else 3959
        max_dist_rad = max_distance / earth_radius
        
        # Query neighbors
        target_rad = np.radians([[target_lat, target_lon]])
        indices, distances = tree.query_radius(
            target_rad,
            r=max_dist_rad,
            return_distance=True,
            sort_results=True
        )
        
        # Convert distances back to original unit
        distances = distances[0] * earth_radius
        indices = indices[0]
        
    else:  # brute_force
        # Calculate distances to all points
        distances = []
        indices = []
        
        coords = valid_df.select([lat_col, lon_col]).to_numpy()
        
        for i in range(len(coords)):
            dist = haversine_distance(
                target_lat, target_lon,
                coords[i, 0], coords[i, 1],
                unit
            )
            if dist <= max_distance:
                distances.append(dist)
                indices.append(i)
                
        distances = np.array(distances)
        indices = np.array(indices)
    
    # Create result DataFrame
    if len(indices) > 0:
        result_df = valid_df[indices].with_columns([
            pl.Series("distance", distances),
            pl.lit(target_lat).alias("target_lat"),
            pl.lit(target_lon).alias("target_lon")
        ])
        return result_df.sort("distance")
    else:
        return pl.DataFrame()


def calculate_spatial_weights(
    df: pl.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    id_col: str = "property_id",
    method: str = "inverse_distance",
    cutoff_distance: Optional[float] = None,
    power: float = 1.0,
    unit: str = "km"
) -> pl.DataFrame:
    """
    Calculate spatial weights based on geographic distances.
    
    Args:
        df: Polars DataFrame with coordinates
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        id_col: Name of ID column
        method: Weighting method ('inverse_distance', 'fixed_distance', 'k_nearest')
        cutoff_distance: Cutoff distance for weights
        power: Power parameter for inverse distance weights
        unit: Unit for distance
        
    Returns:
        Polars DataFrame with spatial weights
    """
    # Filter valid coordinates
    valid_df = df.filter(
        pl.col(lat_col).is_not_null() & 
        pl.col(lon_col).is_not_null()
    ).select([id_col, lat_col, lon_col])
    
    n_locations = len(valid_df)
    if n_locations == 0:
        return pl.DataFrame()
    
    # Calculate distance matrix
    distances = calculate_distance_matrix_polars(
        valid_df, lat_col, lon_col, "haversine", unit
    )
    
    # Calculate weights based on method
    if method == "inverse_distance":
        # Inverse distance weighting
        with np.errstate(divide='ignore'):
            weights = 1.0 / (distances ** power)
        weights[np.isinf(weights)] = 0  # Set diagonal to 0
        
        if cutoff_distance is not None:
            weights[distances > cutoff_distance] = 0
            
    elif method == "fixed_distance":
        # Binary weights within cutoff distance
        if cutoff_distance is None:
            raise ValueError("cutoff_distance required for fixed_distance method")
        weights = (distances <= cutoff_distance).astype(float)
        np.fill_diagonal(weights, 0)
        
    elif method == "k_nearest":
        # K-nearest neighbors (using cutoff_distance as k)
        k = int(cutoff_distance) if cutoff_distance else 5
        weights = np.zeros_like(distances)
        
        for i in range(n_locations):
            # Find k nearest neighbors
            nearest_idx = np.argpartition(distances[i], k+1)[:k+1]
            nearest_idx = nearest_idx[nearest_idx != i][:k]
            weights[i, nearest_idx] = 1.0
            
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights (row standardization)
    row_sums = weights.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = weights / row_sums
    weights[np.isnan(weights)] = 0
    
    # Convert to sparse format for efficiency
    from_ids = []
    to_ids = []
    weight_values = []
    
    ids = valid_df[id_col].to_list()
    
    for i in range(n_locations):
        for j in range(n_locations):
            if weights[i, j] > 0:
                from_ids.append(ids[i])
                to_ids.append(ids[j])
                weight_values.append(weights[i, j])
    
    # Create result DataFrame
    result_df = pl.DataFrame({
        f"{id_col}_from": from_ids,
        f"{id_col}_to": to_ids,
        "weight": weight_values,
        "weight_type": method
    })
    
    return result_df


def create_distance_bands(
    df: pl.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    bands: List[float] = [0, 1, 5, 10, 25, 50],
    unit: str = "km"
) -> pl.DataFrame:
    """
    Create distance bands for spatial analysis.
    
    Args:
        df: Polars DataFrame with coordinates
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        bands: List of distance band boundaries
        unit: Unit for distance
        
    Returns:
        Polars DataFrame with distance band assignments
    """
    # Ensure bands are sorted
    bands = sorted(bands)
    
    # Calculate pairwise distances
    distances = calculate_distance_matrix_polars(
        df, lat_col, lon_col, "haversine", unit
    )
    
    # Create band labels
    band_labels = []
    for i in range(len(bands) - 1):
        band_labels.append(f"{bands[i]}-{bands[i+1]}{unit}")
    
    # Assign distance bands
    n_locations = len(df)
    band_matrix = np.zeros((n_locations, n_locations), dtype=int)
    
    for i in range(len(bands) - 1):
        mask = (distances >= bands[i]) & (distances < bands[i + 1])
        band_matrix[mask] = i + 1
    
    # Convert to DataFrame format
    results = []
    for i in range(n_locations):
        for j in range(i + 1, n_locations):
            if band_matrix[i, j] > 0:
                results.append({
                    "location1_idx": i,
                    "location2_idx": j,
                    "distance": distances[i, j],
                    "band_index": band_matrix[i, j],
                    "band_label": band_labels[band_matrix[i, j] - 1]
                })
    
    if results:
        return pl.DataFrame(results)
    else:
        return pl.DataFrame()


def calculate_centroid(
    df: pl.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    weight_col: Optional[str] = None
) -> Tuple[float, float]:
    """
    Calculate the geographic centroid of points.
    
    Args:
        df: Polars DataFrame with coordinates
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        weight_col: Optional column for weighted centroid
        
    Returns:
        Tuple of (latitude, longitude) for centroid
    """
    # Filter valid coordinates
    valid_df = df.filter(
        pl.col(lat_col).is_not_null() & 
        pl.col(lon_col).is_not_null()
    )
    
    if len(valid_df) == 0:
        return (np.nan, np.nan)
    
    # Convert to radians
    lats_rad = np.radians(valid_df[lat_col].to_numpy())
    lons_rad = np.radians(valid_df[lon_col].to_numpy())
    
    # Get weights
    if weight_col and weight_col in valid_df.columns:
        weights = valid_df[weight_col].to_numpy()
        weights = weights / weights.sum()  # Normalize
    else:
        weights = np.ones(len(valid_df)) / len(valid_df)
    
    # Convert to Cartesian coordinates
    x = np.cos(lats_rad) * np.cos(lons_rad)
    y = np.cos(lats_rad) * np.sin(lons_rad)
    z = np.sin(lats_rad)
    
    # Calculate weighted average
    x_mean = np.sum(x * weights)
    y_mean = np.sum(y * weights)
    z_mean = np.sum(z * weights)
    
    # Convert back to geographic coordinates
    lon_rad = np.arctan2(y_mean, x_mean)
    hyp = np.sqrt(x_mean**2 + y_mean**2)
    lat_rad = np.arctan2(z_mean, hyp)
    
    return (np.degrees(lat_rad), np.degrees(lon_rad))