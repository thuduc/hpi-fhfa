"""Unit tests for distance module."""

import pytest
import math
import pandas as pd
import numpy as np
from hpi_fhfa.geography.census_tract import CensusTract
from hpi_fhfa.geography.distance import (
    calculate_haversine_distance,
    calculate_centroid_distance,
    find_nearest_neighbors,
    calculate_distance_matrix,
    find_tracts_within_radius,
    group_tracts_by_proximity,
    calculate_geographic_weights
)


class TestDistanceCalculations:
    """Test distance calculation functions."""
    
    def test_haversine_distance_same_point(self):
        """Test distance between same point is zero."""
        dist = calculate_haversine_distance(40.7128, -74.0060, 40.7128, -74.0060)
        assert dist == pytest.approx(0.0, abs=1e-6)
    
    def test_haversine_distance_known_values(self):
        """Test distance calculation with known values."""
        # NYC to LA approximately 2451 miles
        dist = calculate_haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        assert dist == pytest.approx(2451, rel=0.01)
    
    def test_haversine_distance_antipodal(self):
        """Test distance between antipodal points."""
        # Half Earth circumference ≈ 12,437 miles
        dist = calculate_haversine_distance(0, 0, 0, 180)
        expected = math.pi * 3959.0  # π * Earth radius
        assert dist == pytest.approx(expected, rel=0.01)
    
    def test_calculate_centroid_distance(self):
        """Test distance calculation between census tracts."""
        tract1 = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        tract2 = CensusTract(
            tract_code="12345678902",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678902",
            centroid_lat=40.7580,
            centroid_lon=-73.9855,
            distance_to_cbd=4.0
        )
        
        dist = calculate_centroid_distance(tract1, tract2)
        assert dist > 0
        assert dist < 10  # Should be relatively close in same city
    
    def test_find_nearest_neighbors_basic(self):
        """Test finding nearest neighbors."""
        target = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        candidates = [
            CensusTract(
                tract_code="12345678902",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678902",
                centroid_lat=40.7130,  # Very close
                centroid_lon=-74.0062,
                distance_to_cbd=5.4
            ),
            CensusTract(
                tract_code="12345678903",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678903",
                centroid_lat=40.7580,  # Farther
                centroid_lon=-73.9855,
                distance_to_cbd=4.0
            ),
            CensusTract(
                tract_code="12345678904",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678904",
                centroid_lat=40.7135,  # Medium distance
                centroid_lon=-74.0070,
                distance_to_cbd=5.3
            )
        ]
        
        neighbors = find_nearest_neighbors(target, candidates, n_neighbors=2)
        
        assert len(neighbors) == 2
        # First neighbor should be the closest
        assert neighbors[0][0].tract_code == "12345678902"
        assert neighbors[1][0].tract_code == "12345678904"
        # Distances should be ordered
        assert neighbors[0][1] < neighbors[1][1]
    
    def test_find_nearest_neighbors_same_cbsa_filter(self):
        """Test filtering by same CBSA."""
        target = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        candidates = [
            CensusTract(
                tract_code="12345678902",
                cbsa_code="12345",  # Same CBSA
                state_code="12",
                county_code="345",
                tract_number="678902",
                centroid_lat=40.7130,
                centroid_lon=-74.0062,
                distance_to_cbd=5.4
            ),
            CensusTract(
                tract_code="98765432101",
                cbsa_code="98765",  # Different CBSA
                state_code="98",
                county_code="765",
                tract_number="432101",
                centroid_lat=40.7125,  # Actually closer
                centroid_lon=-74.0058,
                distance_to_cbd=5.6
            )
        ]
        
        neighbors = find_nearest_neighbors(target, candidates, same_cbsa_only=True)
        
        assert len(neighbors) == 1
        assert neighbors[0][0].cbsa_code == "12345"
    
    def test_find_nearest_neighbors_max_distance(self):
        """Test maximum distance threshold."""
        target = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        candidates = [
            CensusTract(
                tract_code="12345678902",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678902",
                centroid_lat=40.7130,
                centroid_lon=-74.0062,
                distance_to_cbd=5.4
            ),
            CensusTract(
                tract_code="12345678903",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678903",
                centroid_lat=41.0,  # Far away
                centroid_lon=-74.5,
                distance_to_cbd=20.0
            )
        ]
        
        neighbors = find_nearest_neighbors(target, candidates, max_distance=1.0)
        
        # Only the close tract should be included
        assert len(neighbors) == 1
        assert neighbors[0][0].tract_code == "12345678902"
    
    def test_find_nearest_neighbors_empty_candidates(self):
        """Test with empty candidate list."""
        target = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        neighbors = find_nearest_neighbors(target, [])
        assert neighbors == []
    
    def test_calculate_distance_matrix(self):
        """Test distance matrix calculation."""
        tracts = [
            CensusTract(
                tract_code="12345678901",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678901",
                centroid_lat=40.7128,
                centroid_lon=-74.0060,
                distance_to_cbd=5.5
            ),
            CensusTract(
                tract_code="12345678902",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678902",
                centroid_lat=40.7130,
                centroid_lon=-74.0062,
                distance_to_cbd=5.4
            ),
            CensusTract(
                tract_code="12345678903",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678903",
                centroid_lat=40.7135,
                centroid_lon=-74.0070,
                distance_to_cbd=5.3
            )
        ]
        
        matrix = calculate_distance_matrix(tracts)
        
        # Check shape
        assert matrix.shape == (3, 3)
        
        # Check diagonal is zero
        assert all(matrix.iloc[i, i] == 0 for i in range(3))
        
        # Check symmetry
        assert matrix.iloc[0, 1] == matrix.iloc[1, 0]
        assert matrix.iloc[0, 2] == matrix.iloc[2, 0]
        assert matrix.iloc[1, 2] == matrix.iloc[2, 1]
        
        # Check all distances are non-negative
        assert (matrix >= 0).all().all()
    
    def test_find_tracts_within_radius(self):
        """Test finding tracts within radius."""
        center = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        all_tracts = [
            center,  # Should be excluded
            CensusTract(
                tract_code="12345678902",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678902",
                centroid_lat=40.7130,  # Very close
                centroid_lon=-74.0062,
                distance_to_cbd=5.4
            ),
            CensusTract(
                tract_code="12345678903",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678903",
                centroid_lat=40.75,  # Medium distance
                centroid_lon=-74.01,
                distance_to_cbd=4.0
            ),
            CensusTract(
                tract_code="12345678904",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678904",
                centroid_lat=41.0,  # Far
                centroid_lon=-74.5,
                distance_to_cbd=20.0
            )
        ]
        
        nearby = find_tracts_within_radius(center, all_tracts, radius_miles=5.0)
        
        # Should find 2 tracts (excluding center and far tract)
        assert len(nearby) == 2
        # Should be sorted by distance
        assert nearby[0][1] < nearby[1][1]
    
    def test_group_tracts_by_proximity(self):
        """Test grouping tracts by proximity."""
        tracts = [
            # Group 1 - close together
            CensusTract(
                tract_code="12345678901",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678901",
                centroid_lat=40.71,
                centroid_lon=-74.00,
                distance_to_cbd=5.5
            ),
            CensusTract(
                tract_code="12345678902",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678902",
                centroid_lat=40.72,
                centroid_lon=-74.01,
                distance_to_cbd=5.4
            ),
            # Group 2 - far from group 1
            CensusTract(
                tract_code="12345678903",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678903",
                centroid_lat=41.0,
                centroid_lon=-74.5,
                distance_to_cbd=20.0
            ),
            CensusTract(
                tract_code="12345678904",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678904",
                centroid_lat=41.01,
                centroid_lon=-74.51,
                distance_to_cbd=20.5
            )
        ]
        
        groups = group_tracts_by_proximity(tracts, max_distance=5.0)
        
        # Should create 2 groups
        assert len(groups) == 2
        
        # Each group should have 2 tracts
        assert all(len(group) == 2 for group in groups)
        
        # Tracts in same group should be close
        for group in groups:
            if len(group) > 1:
                dist = calculate_centroid_distance(group[0], group[1])
                assert dist <= 5.0
    
    def test_group_tracts_by_proximity_empty(self):
        """Test grouping with empty tract list."""
        groups = group_tracts_by_proximity([])
        assert groups == []
    
    def test_calculate_geographic_weights_uniform(self):
        """Test uniform geographic weights."""
        tracts = [
            CensusTract(
                tract_code="12345678901",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678901",
                centroid_lat=40.71,
                centroid_lon=-74.00,
                distance_to_cbd=5.5
            ),
            CensusTract(
                tract_code="12345678902",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678902",
                centroid_lat=40.72,
                centroid_lon=-74.01,
                distance_to_cbd=5.4
            ),
            CensusTract(
                tract_code="12345678903",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678903",
                centroid_lat=40.73,
                centroid_lon=-74.02,
                distance_to_cbd=5.3
            )
        ]
        
        weights = calculate_geographic_weights(tracts, 'uniform')
        
        # All weights should be equal
        assert len(weights) == 3
        assert all(w == pytest.approx(1/3) for w in weights.values())
        
        # Should sum to 1
        assert sum(weights.values()) == pytest.approx(1.0)
    
    def test_calculate_geographic_weights_inverse_distance(self):
        """Test inverse distance weights."""
        tracts = [
            CensusTract(
                tract_code="12345678901",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678901",
                centroid_lat=40.71,
                centroid_lon=-74.00,
                distance_to_cbd=5.5
            ),
            CensusTract(
                tract_code="12345678902",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678902",
                centroid_lat=40.72,
                centroid_lon=-74.01,
                distance_to_cbd=5.4
            ),
            CensusTract(
                tract_code="12345678903",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678903",
                centroid_lat=40.80,  # Farther from centroid
                centroid_lon=-74.10,
                distance_to_cbd=8.0
            )
        ]
        
        weights = calculate_geographic_weights(tracts, 'inverse_distance')
        
        # Should have 3 weights
        assert len(weights) == 3
        
        # Should sum to 1
        assert sum(weights.values()) == pytest.approx(1.0)
        
        # Farther tract should have lower weight
        assert weights["12345678903"] < weights["12345678901"]
        assert weights["12345678903"] < weights["12345678902"]
    
    def test_calculate_geographic_weights_empty(self):
        """Test weights with empty tract list."""
        weights = calculate_geographic_weights([], 'uniform')
        assert weights == {}
    
    def test_calculate_geographic_weights_invalid_type(self):
        """Test invalid weight type."""
        tracts = [
            CensusTract(
                tract_code="12345678901",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678901",
                centroid_lat=40.71,
                centroid_lon=-74.00,
                distance_to_cbd=5.5
            )
        ]
        
        with pytest.raises(ValueError, match="Unknown weight type"):
            calculate_geographic_weights(tracts, 'invalid')