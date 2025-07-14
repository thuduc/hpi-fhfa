"""Geographic distance calculations for RSAI"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy.spatial import cKDTree


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the haversine distance between two points on Earth.
    
    Parameters:
    -----------
    lat1, lon1: float
        Latitude and longitude of first point in degrees
    lat2, lon2: float
        Latitude and longitude of second point in degrees
        
    Returns:
    --------
    float
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth's radius in kilometers
    r = 6371
    
    return r * c


def haversine_vectorized(lat1: np.ndarray, lon1: np.ndarray, 
                        lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Vectorized version of haversine distance calculation.
    
    Parameters:
    -----------
    lat1, lon1: array-like
        Arrays of latitudes and longitudes for first set of points
    lat2, lon2: array-like
        Arrays of latitudes and longitudes for second set of points
        
    Returns:
    --------
    np.ndarray
        Array of distances in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth's radius in kilometers
    r = 6371
    
    return r * c


class GeographicDistanceCalculator:
    """Efficient distance calculations for census tracts"""
    
    def __init__(self, geographic_df: pd.DataFrame):
        """
        Initialize with geographic data containing tract centroids.
        
        Parameters:
        -----------
        geographic_df: pd.DataFrame
            DataFrame with columns: census_tract_2010, centroid_lat, centroid_lon
        """
        self.geographic_df = geographic_df.copy()
        self._build_spatial_index()
    
    def _build_spatial_index(self):
        """Build spatial index for efficient nearest neighbor queries"""
        # Convert lat/lon to radians for more accurate distance calculations
        coords_rad = np.radians(self.geographic_df[['centroid_lat', 'centroid_lon']].values)
        
        # Build KDTree for efficient spatial queries
        self.tree = cKDTree(coords_rad)
        
        # Create tract ID to index mapping
        self.tract_to_idx = {tract: idx for idx, tract in 
                            enumerate(self.geographic_df['census_tract_2010'])}
        self.idx_to_tract = {idx: tract for tract, idx in self.tract_to_idx.items()}
    
    def get_nearest_neighbor(self, tract_id: str, exclude_tracts: set = None) -> Tuple[str, float]:
        """
        Find the nearest neighbor for a given census tract.
        
        Parameters:
        -----------
        tract_id: str
            Census tract ID to find neighbor for
        exclude_tracts: set, optional
            Set of tract IDs to exclude from search
            
        Returns:
        --------
        tuple
            (nearest_tract_id, distance_km)
        """
        if tract_id not in self.tract_to_idx:
            raise ValueError(f"Tract {tract_id} not found in geographic data")
        
        # Get coordinates for the query tract
        idx = self.tract_to_idx[tract_id]
        query_point = np.radians(self.geographic_df.iloc[idx][['centroid_lat', 'centroid_lon']].values)
        
        # Find k nearest neighbors (k=len to ensure we can skip excluded)
        k = min(len(self.geographic_df), 20)  # Limit search for efficiency
        distances, indices = self.tree.query([query_point], k=k)
        
        # Find first non-excluded neighbor
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            neighbor_tract = self.idx_to_tract[neighbor_idx]
            
            # Skip self and excluded tracts
            if neighbor_tract == tract_id:
                continue
            if exclude_tracts and neighbor_tract in exclude_tracts:
                continue
            
            # Convert distance to kilometers (approximate)
            distance_km = dist * 6371  # Earth's radius
            
            return neighbor_tract, distance_km
        
        raise ValueError(f"No valid neighbor found for tract {tract_id}")
    
    def get_distance_between_tracts(self, tract1: str, tract2: str) -> float:
        """
        Calculate distance between two census tracts.
        
        Parameters:
        -----------
        tract1, tract2: str
            Census tract IDs
            
        Returns:
        --------
        float
            Distance in kilometers
        """
        if tract1 not in self.tract_to_idx or tract2 not in self.tract_to_idx:
            raise ValueError("One or both tracts not found in geographic data")
        
        idx1 = self.tract_to_idx[tract1]
        idx2 = self.tract_to_idx[tract2]
        
        row1 = self.geographic_df.iloc[idx1]
        row2 = self.geographic_df.iloc[idx2]
        
        return haversine_distance(
            row1['centroid_lat'], row1['centroid_lon'],
            row2['centroid_lat'], row2['centroid_lon']
        )
    
    def get_all_distances_from_tract(self, tract_id: str) -> pd.DataFrame:
        """
        Calculate distances from one tract to all others.
        
        Parameters:
        -----------
        tract_id: str
            Source census tract ID
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: census_tract_2010, distance_km
        """
        if tract_id not in self.tract_to_idx:
            raise ValueError(f"Tract {tract_id} not found in geographic data")
        
        idx = self.tract_to_idx[tract_id]
        source = self.geographic_df.iloc[idx]
        
        # Calculate distances to all other tracts
        distances = haversine_vectorized(
            source['centroid_lat'], source['centroid_lon'],
            self.geographic_df['centroid_lat'].values,
            self.geographic_df['centroid_lon'].values
        )
        
        result_df = pd.DataFrame({
            'census_tract_2010': self.geographic_df['census_tract_2010'],
            'distance_km': distances
        })
        
        # Sort by distance
        result_df = result_df.sort_values('distance_km')
        
        return result_df[result_df['census_tract_2010'] != tract_id]