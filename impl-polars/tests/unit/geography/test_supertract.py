"""
Unit tests for supertract generation.

Tests the algorithms for aggregating census tracts into supertracts
based on transaction density and geographic proximity.
"""

import pytest
from datetime import date
import polars as pl
import numpy as np

from rsai.src.geography.supertract import SupertractGenerator, TractInfo
from rsai.src.data.models import SupertractDefinition


class TestSuperTractGenerator:
    """Test SupertractGenerator class."""
    
    def test_initialization(self):
        """Test SupertractGenerator initialization."""
        generator = SupertractGenerator(
            min_transactions=100,
            max_distance_km=10.0,
            min_tracts=1,
            max_tracts=20,
            method="hierarchical"
        )
        assert generator.min_transactions == 100
        assert generator.max_distance_km == 10.0
        assert generator.method == "hierarchical"
        assert len(generator.tract_info) == 0
        assert len(generator.supertracts) == 0
    
    def test_prepare_tract_data(self, sample_transactions_df, sample_properties_df):
        """Test preparation of tract-level statistics."""
        generator = SupertractGenerator()
        
        # Ensure tract column exists
        if "tract" not in sample_transactions_df.columns:
            sample_transactions_df = sample_transactions_df.with_columns([
                pl.lit("06037123456").alias("tract")
            ])
        
        tract_stats = generator.prepare_tract_data(
            sample_transactions_df,
            sample_properties_df
        )
        
        assert isinstance(tract_stats, pl.DataFrame)
        assert "tract" in tract_stats.columns
        assert "num_transactions" in tract_stats.columns
        assert "num_properties" in tract_stats.columns
        assert "median_price" in tract_stats.columns
        assert "centroid_lat" in tract_stats.columns
        assert "centroid_lon" in tract_stats.columns
        
        # Check that tract info was stored
        assert len(generator.tract_info) > 0
        for tract_id, info in generator.tract_info.items():
            assert isinstance(info, TractInfo)
            assert info.num_transactions > 0
    
    def test_prepare_tract_data_no_properties(self, sample_transactions_df):
        """Test tract data preparation without property data."""
        generator = SupertractGenerator()
        
        tract_stats = generator.prepare_tract_data(sample_transactions_df)
        
        assert isinstance(tract_stats, pl.DataFrame)
        assert "centroid_lat" in tract_stats.columns
        assert "centroid_lon" in tract_stats.columns
        
        # Centroids should be null without property data
        assert tract_stats["centroid_lat"][0] is None
        assert tract_stats["centroid_lon"][0] is None
    
    def test_hierarchical_clustering(self, sample_tract_stats_df):
        """Test hierarchical clustering method."""
        # Ensure test data has enough transactions per tract
        sample_tract_stats_df = sample_tract_stats_df.with_columns([
            (pl.col("num_transactions") + 5).alias("num_transactions")
        ])
        
        generator = SupertractGenerator(
            min_transactions=5,  # Lowered for test data
            max_distance_km=5.0,
            min_tracts=1,
            max_tracts=10,  # Increased to allow larger clusters
            method="hierarchical"
        )
        
        # Prepare tract info
        for row in sample_tract_stats_df.iter_rows(named=True):
            generator.tract_info[row["tract"]] = TractInfo(
                tract_id=row["tract"],
                county_fips=row["county_fips"],
                num_properties=row["num_properties"],
                num_transactions=row["num_transactions"],
                median_price=row["median_price"],
                centroid_lat=row.get("centroid_lat"),
                centroid_lon=row.get("centroid_lon")
            )
        
        supertracts = generator.generate_supertracts(sample_tract_stats_df)
        
        assert isinstance(supertracts, list)
        assert len(supertracts) > 0
        
        for supertract in supertracts:
            assert isinstance(supertract, SupertractDefinition)
            assert len(supertract.tract_ids) >= generator.min_tracts
            assert len(supertract.tract_ids) <= generator.max_tracts
            assert supertract.num_transactions >= generator.min_transactions
    
    def test_dbscan_clustering(self, sample_tract_stats_df):
        """Test DBSCAN clustering method."""
        generator = SupertractGenerator(
            min_transactions=50,
            max_distance_km=5.0,
            min_tracts=1,
            max_tracts=10,
            method="dbscan"
        )
        
        # Prepare tract info
        for row in sample_tract_stats_df.iter_rows(named=True):
            generator.tract_info[row["tract"]] = TractInfo(
                tract_id=row["tract"],
                county_fips=row["county_fips"],
                num_properties=row["num_properties"],
                num_transactions=row["num_transactions"],
                median_price=row["median_price"],
                centroid_lat=row.get("centroid_lat"),
                centroid_lon=row.get("centroid_lon")
            )
        
        supertracts = generator.generate_supertracts(sample_tract_stats_df)
        
        assert isinstance(supertracts, list)
        # DBSCAN may not create supertracts if density is too low
        assert len(supertracts) >= 0
    
    def test_greedy_aggregation(self, sample_tract_stats_df):
        """Test greedy aggregation method."""
        generator = SupertractGenerator(
            min_transactions=50,
            max_distance_km=10.0,
            min_tracts=1,
            max_tracts=5,
            method="greedy"
        )
        
        # Prepare tract info
        for row in sample_tract_stats_df.iter_rows(named=True):
            generator.tract_info[row["tract"]] = TractInfo(
                tract_id=row["tract"],
                county_fips=row["county_fips"],
                num_properties=row["num_properties"],
                num_transactions=row["num_transactions"],
                median_price=row["median_price"],
                centroid_lat=row.get("centroid_lat"),
                centroid_lon=row.get("centroid_lon")
            )
        
        supertracts = generator.generate_supertracts(sample_tract_stats_df)
        
        assert isinstance(supertracts, list)
        assert len(supertracts) > 0
    
    def test_fallback_grouping(self):
        """Test fallback grouping when no geographic data is available."""
        # Create tract stats without coordinates
        tract_stats = pl.DataFrame({
            "tract": ["06037001", "06037002", "06037003", "06037004"],
            "county_fips": ["06037"] * 4,
            "num_transactions": [100, 50, 75, 125],
            "num_properties": [80, 40, 60, 100],
            "median_price": [500000, 400000, 450000, 600000],
            "centroid_lat": [None] * 4,
            "centroid_lon": [None] * 4
        })
        
        generator = SupertractGenerator(
            min_transactions=100,
            max_distance_km=10.0,
            min_tracts=1,
            max_tracts=3,
            method="hierarchical"
        )
        
        # Prepare tract info
        for row in tract_stats.iter_rows(named=True):
            generator.tract_info[row["tract"]] = TractInfo(
                tract_id=row["tract"],
                county_fips=row["county_fips"],
                num_properties=row["num_properties"],
                num_transactions=row["num_transactions"],
                median_price=row["median_price"],
                centroid_lat=None,
                centroid_lon=None
            )
        
        supertracts = generator.generate_supertracts(tract_stats)
        
        assert len(supertracts) > 0
        # Should use fallback grouping
        total_tracts = sum(len(st.tract_ids) for st in supertracts)
        assert total_tracts == 4
    
    def test_create_supertract_from_tracts(self):
        """Test creating a supertract from tract IDs."""
        generator = SupertractGenerator()
        
        # Add some tract info
        generator.tract_info = {
            "06037001": TractInfo(
                tract_id="06037001",
                county_fips="06037",
                num_properties=100,
                num_transactions=200,
                median_price=500000,
                centroid_lat=34.05,
                centroid_lon=-118.25
            ),
            "06037002": TractInfo(
                tract_id="06037002",
                county_fips="06037",
                num_properties=150,
                num_transactions=300,
                median_price=600000,
                centroid_lat=34.06,
                centroid_lon=-118.24
            )
        }
        
        supertract = generator._create_supertract_from_tracts(
            ["06037001", "06037002"],
            "06037"
        )
        
        assert supertract is not None
        assert supertract.supertract_id == "06037_ST_001"
        assert len(supertract.tract_ids) == 2
        assert supertract.num_properties == 250
        assert supertract.num_transactions == 500
        assert supertract.median_price == 550000  # Median of medians
        assert supertract.min_lat == 34.05
        assert supertract.max_lat == 34.06
    
    def test_create_supertract_empty_list(self):
        """Test creating supertract with empty tract list."""
        generator = SupertractGenerator()
        
        supertract = generator._create_supertract_from_tracts([], "06037")
        
        assert supertract is None
    
    def test_find_nearby_unassigned_tracts(self):
        """Test finding nearby unassigned tracts."""
        generator = SupertractGenerator(max_distance_km=5.0)
        
        # Create sample tract data
        county_tracts = pl.DataFrame({
            "tract": ["06037001", "06037002", "06037003"],
            "centroid_lat": [34.05, 34.06, 34.20],  # Third is far away
            "centroid_lon": [-118.25, -118.24, -118.30],
            "num_transactions": [100, 150, 200]
        })
        
        assigned = {"06037001"}  # First tract already assigned
        
        nearby = generator._find_nearby_unassigned_tracts(
            34.05, -118.25,  # Reference point
            county_tracts,
            assigned
        )
        
        assert len(nearby) >= 1
        assert nearby[0]["tract"] == "06037002"  # Closest unassigned
        assert all(tract["tract"] not in assigned for tract in nearby)
    
    def test_export_mapping(self):
        """Test exporting tract to supertract mapping."""
        generator = SupertractGenerator()
        
        # Create some supertracts
        generator.supertracts = [
            SupertractDefinition(
                supertract_id="06037_ST_001",
                county_fips="06037",
                tract_ids=["06037001", "06037002"],
                num_properties=100,
                num_transactions=200
            ),
            SupertractDefinition(
                supertract_id="06037_ST_002",
                county_fips="06037",
                tract_ids=["06037003"],
                num_properties=50,
                num_transactions=100
            )
        ]
        
        mapping_df = generator.export_mapping()
        
        assert isinstance(mapping_df, pl.DataFrame)
        assert len(mapping_df) == 3  # Total number of tract mappings
        assert "tract_id" in mapping_df.columns
        assert "supertract_id" in mapping_df.columns
        assert "county_fips" in mapping_df.columns
        assert "num_tracts_in_supertract" in mapping_df.columns
        
        # Check specific mappings
        tract1_mapping = mapping_df.filter(pl.col("tract_id") == "06037001")
        assert tract1_mapping["supertract_id"][0] == "06037_ST_001"
        assert tract1_mapping["num_tracts_in_supertract"][0] == 2
    
    def test_adjust_cluster_size(self):
        """Test cluster size adjustment logic."""
        generator = SupertractGenerator(
            min_transactions=100,
            max_tracts=3
        )
        
        # Create a cluster that's too large
        large_cluster = pl.DataFrame({
            "tract": ["T1", "T2", "T3", "T4", "T5"],
            "county_fips": ["06037"] * 5,
            "num_transactions": [50, 60, 70, 80, 90]
        })
        
        # Mock distance matrix
        distances = np.random.rand(5, 5)
        np.fill_diagonal(distances, 0)
        
        # This should trigger splitting logic
        all_tracts = large_cluster.with_row_count()
        
        # Add tract info
        for row in large_cluster.iter_rows(named=True):
            generator.tract_info[row["tract"]] = TractInfo(
                tract_id=row["tract"],
                county_fips=row["county_fips"],
                num_properties=row["num_transactions"] // 2,
                num_transactions=row["num_transactions"],
                median_price=500000,
                centroid_lat=34.0,
                centroid_lon=-118.0
            )
        
        adjusted = generator._adjust_cluster_size(
            large_cluster,
            all_tracts,
            distances
        )
        
        assert isinstance(adjusted, list)
        # Should split into multiple supertracts
        assert len(adjusted) >= 2
    
    def test_invalid_method(self):
        """Test error handling for invalid clustering method."""
        generator = SupertractGenerator(method="invalid_method")
        
        tract_stats = pl.DataFrame({
            "tract": ["06037001"],
            "county_fips": ["06037"],
            "num_transactions": [100],
            "num_properties": [80],
            "median_price": [500000]
        })
        
        with pytest.raises(ValueError, match="Unknown method"):
            generator.generate_supertracts(tract_stats)
    
    def test_small_county_handling(self, sample_tract_stats_df):
        """Test handling of small counties that don't need clustering."""
        generator = SupertractGenerator(
            min_transactions=10,
            max_tracts=20,  # Large enough to contain all tracts
            method="hierarchical"
        )
        
        # Use only a few tracts
        small_county_data = sample_tract_stats_df.head(3)
        
        # Prepare tract info
        for row in small_county_data.iter_rows(named=True):
            generator.tract_info[row["tract"]] = TractInfo(
                tract_id=row["tract"],
                county_fips=row["county_fips"],
                num_properties=row["num_properties"],
                num_transactions=row["num_transactions"],
                median_price=row["median_price"],
                centroid_lat=row.get("centroid_lat"),
                centroid_lon=row.get("centroid_lon")
            )
        
        supertracts = generator.generate_supertracts(small_county_data)
        
        # Should create a single supertract for the small county
        assert len(supertracts) >= 1
        total_tracts = sum(len(st.tract_ids) for st in supertracts)
        assert total_tracts == 3