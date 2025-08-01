"""Geographic distance calculations for census tracts."""

import math
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from .census_tract import CensusTract
import logging

logger = logging.getLogger(__name__)

# Earth radius in miles
EARTH_RADIUS_MILES = 3959.0


def calculate_haversine_distance(lat1: float, lon1: float, 
                                lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points using Haversine formula.
    
    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees
        
    Returns:
        Distance in miles
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(dlon / 2) ** 2)
    
    c = 2 * math.asin(math.sqrt(a))
    
    return EARTH_RADIUS_MILES * c


def calculate_centroid_distance(tract1: CensusTract, tract2: CensusTract) -> float:
    """Calculate distance between census tract centroids.
    
    Args:
        tract1: First census tract
        tract2: Second census tract
        
    Returns:
        Distance in miles
    """
    return calculate_haversine_distance(
        tract1.centroid_lat, tract1.centroid_lon,
        tract2.centroid_lat, tract2.centroid_lon
    )


def find_nearest_neighbors(target_tract: CensusTract, 
                          candidate_tracts: List[CensusTract],
                          n_neighbors: int = 5,
                          max_distance: float = float('inf'),
                          same_cbsa_only: bool = True) -> List[Tuple[CensusTract, float]]:
    """Find nearest neighboring tracts to a target tract.
    
    Args:
        target_tract: The tract to find neighbors for
        candidate_tracts: List of potential neighbor tracts
        n_neighbors: Maximum number of neighbors to return
        max_distance: Maximum distance threshold in miles
        same_cbsa_only: Whether to restrict to same CBSA
        
    Returns:
        List of (tract, distance) tuples sorted by distance
    """
    if not candidate_tracts:
        return []
    
    # Filter candidates
    filtered_candidates = []
    for tract in candidate_tracts:
        # Skip the target tract itself
        if tract.tract_code == target_tract.tract_code:
            continue
        
        # Apply CBSA filter if requested
        if same_cbsa_only and tract.cbsa_code != target_tract.cbsa_code:
            continue
        
        filtered_candidates.append(tract)
    
    if not filtered_candidates:
        logger.warning(f"No valid neighbors found for tract {target_tract.tract_code}")
        return []
    
    # Calculate distances
    distances = []
    for tract in filtered_candidates:
        dist = calculate_centroid_distance(target_tract, tract)
        if dist <= max_distance:
            distances.append((tract, dist))
    
    # Sort by distance and return top n
    distances.sort(key=lambda x: x[1])
    return distances[:n_neighbors]


def calculate_distance_matrix(tracts: List[CensusTract]) -> pd.DataFrame:
    """Calculate pairwise distance matrix between all tracts.
    
    Args:
        tracts: List of census tracts
        
    Returns:
        DataFrame with tract codes as index/columns and distances as values
    """
    n_tracts = len(tracts)
    tract_codes = [t.tract_code for t in tracts]
    
    # Initialize distance matrix
    distances = np.zeros((n_tracts, n_tracts))
    
    # Calculate pairwise distances
    for i in range(n_tracts):
        for j in range(i + 1, n_tracts):
            dist = calculate_centroid_distance(tracts[i], tracts[j])
            distances[i, j] = dist
            distances[j, i] = dist  # Symmetric
    
    # Convert to DataFrame
    return pd.DataFrame(distances, index=tract_codes, columns=tract_codes)


def find_tracts_within_radius(center_tract: CensusTract,
                             all_tracts: List[CensusTract],
                             radius_miles: float) -> List[Tuple[CensusTract, float]]:
    """Find all tracts within a specified radius of a center tract.
    
    Args:
        center_tract: The center tract
        all_tracts: List of all tracts to search
        radius_miles: Search radius in miles
        
    Returns:
        List of (tract, distance) tuples within radius
    """
    nearby_tracts = []
    
    for tract in all_tracts:
        if tract.tract_code == center_tract.tract_code:
            continue
        
        dist = calculate_centroid_distance(center_tract, tract)
        if dist <= radius_miles:
            nearby_tracts.append((tract, dist))
    
    # Sort by distance
    nearby_tracts.sort(key=lambda x: x[1])
    return nearby_tracts


def group_tracts_by_proximity(tracts: List[CensusTract],
                             max_distance: float = 5.0) -> List[List[CensusTract]]:
    """Group tracts into clusters based on proximity.
    
    Simple clustering algorithm that groups tracts within max_distance
    of each other. Used for initial supertract formation.
    
    Args:
        tracts: List of census tracts
        max_distance: Maximum distance for grouping in miles
        
    Returns:
        List of tract groups
    """
    if not tracts:
        return []
    
    # Track which tracts have been assigned to groups
    unassigned = set(tracts)
    groups = []
    
    while unassigned:
        # Start new group with an unassigned tract
        seed_tract = unassigned.pop()
        current_group = [seed_tract]
        
        # Find all tracts within max_distance of any tract in current group
        changed = True
        while changed:
            changed = False
            to_add = []
            
            for candidate in list(unassigned):
                # Check if candidate is close to any tract in current group
                for group_tract in current_group:
                    dist = calculate_centroid_distance(candidate, group_tract)
                    if dist <= max_distance:
                        to_add.append(candidate)
                        break
            
            if to_add:
                changed = True
                for tract in to_add:
                    current_group.append(tract)
                    unassigned.remove(tract)
        
        groups.append(current_group)
    
    return groups


def calculate_geographic_weights(tracts: List[CensusTract],
                                weight_type: str = 'inverse_distance') -> Dict[str, float]:
    """Calculate geographic-based weights for tracts.
    
    Args:
        tracts: List of census tracts
        weight_type: Type of geographic weighting
            - 'inverse_distance': Weight by 1/distance from centroid
            - 'uniform': Equal weights
            
    Returns:
        Dictionary mapping tract codes to weights
    """
    if not tracts:
        return {}
    
    weights = {}
    
    if weight_type == 'uniform':
        # Equal weights
        weight = 1.0 / len(tracts)
        for tract in tracts:
            weights[tract.tract_code] = weight
    
    elif weight_type == 'inverse_distance':
        # Calculate geographic centroid of all tracts
        mean_lat = np.mean([t.centroid_lat for t in tracts])
        mean_lon = np.mean([t.centroid_lon for t in tracts])
        
        # Calculate inverse distances
        inv_distances = []
        for tract in tracts:
            dist = calculate_haversine_distance(
                mean_lat, mean_lon,
                tract.centroid_lat, tract.centroid_lon
            )
            # Add small epsilon to avoid division by zero
            inv_dist = 1.0 / (dist + 0.1)
            inv_distances.append(inv_dist)
        
        # Normalize to sum to 1
        total = sum(inv_distances)
        for tract, inv_dist in zip(tracts, inv_distances):
            weights[tract.tract_code] = inv_dist / total
    
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")
    
    return weights