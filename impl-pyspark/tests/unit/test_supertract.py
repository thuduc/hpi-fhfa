"""Unit tests for Supertract algorithm"""

import pytest
from pyspark.sql import SparkSession
from datetime import date

from hpi_fhfa.core.supertract import SupertractAlgorithm
from hpi_fhfa.schemas.data_schemas import DataSchemas


class TestSupertractAlgorithm:
    """Test suite for Supertract algorithm"""
    
    @pytest.fixture
    def spark(self):
        return SparkSession.builder \
            .master("local[2]") \
            .appName("test-supertract") \
            .config("spark.sql.shuffle.partitions", "2") \
            .getOrCreate()
    
    @pytest.fixture
    def supertract_algo(self, spark):
        return SupertractAlgorithm(spark, min_half_pairs=40)
    
    @pytest.fixture
    def sample_half_pairs(self, spark):
        """Create sample half-pairs data"""
        data = [
            # Tract with sufficient half-pairs
            ("12345", "19100", 2019, 100),
            ("12345", "19100", 2020, 95),
            
            # Tract below threshold
            ("12346", "19100", 2019, 20),
            ("12346", "19100", 2020, 25),
            
            # Another tract below threshold
            ("12347", "19100", 2019, 15),
            ("12347", "19100", 2020, 18),
            
            # Tract in different CBSA
            ("12348", "19200", 2019, 30),
            ("12348", "19200", 2020, 35),
        ]
        
        schema = ["census_tract", "cbsa_code", "year", "total_half_pairs"]
        return spark.createDataFrame(data, schema)
    
    @pytest.fixture
    def sample_geographic(self, spark):
        """Create sample geographic data"""
        data = [
            ("12345", "19100", 40.0, -74.0, ["12346"]),
            ("12346", "19100", 40.1, -74.1, ["12345", "12347"]),
            ("12347", "19100", 40.2, -74.2, ["12346"]),
            ("12348", "19200", 41.0, -75.0, []),
        ]
        
        return spark.createDataFrame(data, DataSchemas.GEOGRAPHIC_SCHEMA)
    
    def test_create_supertracts_basic(self, supertract_algo, sample_half_pairs, sample_geographic):
        """Test basic supertract creation"""
        supertracts = supertract_algo.create_supertracts(
            sample_half_pairs,
            sample_geographic,
            year=2020
        )
        
        # Should create supertracts
        assert supertracts.count() > 0
        
        # Check schema
        assert "supertract_id" in supertracts.columns
        assert "tract_list" in supertracts.columns
        assert "min_half_pairs" in supertracts.columns
        
        # Tract 12345 should remain independent (has enough half-pairs)
        tract_12345 = supertracts.filter(
            supertracts.supertract_id == "12345"
        ).first()
        
        if tract_12345:
            assert tract_12345.min_half_pairs >= 40
            assert len(tract_12345.tract_list) == 1
        
        # Tracts 12346 and 12347 should be merged
        merged_count = supertracts.filter(
            supertracts.num_tracts > 1
        ).count()
        assert merged_count > 0
    
    def test_create_supertracts_different_cbsa(self, supertract_algo, sample_half_pairs, sample_geographic):
        """Test that tracts from different CBSAs are not merged"""
        supertracts = supertract_algo.create_supertracts(
            sample_half_pairs,
            sample_geographic,
            year=2020
        )
        
        # Get all supertracts
        all_supertracts = supertracts.collect()
        
        # Check that no supertract contains tracts from multiple CBSAs
        for st in all_supertracts:
            cbsas = set()
            for tract in st.tract_list:
                # Find CBSA for this tract
                geo_row = sample_geographic.filter(
                    sample_geographic.census_tract == tract
                ).first()
                if geo_row:
                    cbsas.add(geo_row.cbsa_code)
            
            # Should only have one CBSA per supertract
            assert len(cbsas) == 1
    
    def test_create_tract_to_supertract_mapping(self, supertract_algo, spark):
        """Test tract to supertract mapping creation"""
        # Create sample supertracts
        supertract_data = [
            {
                "supertract_id": "ST1",
                "cbsa_code": "19100",
                "tract_list": ["12345", "12346"],
                "min_half_pairs": 60,
                "num_tracts": 2
            },
            {
                "supertract_id": "ST2",
                "cbsa_code": "19100",
                "tract_list": ["12347"],
                "min_half_pairs": 45,
                "num_tracts": 1
            }
        ]
        
        supertracts = spark.createDataFrame(supertract_data)
        
        # Create mapping
        mapping = supertract_algo.create_tract_to_supertract_mapping(supertracts)
        
        # Check mapping
        assert mapping.count() == 3  # 3 total tracts
        
        # Check specific mappings
        tract_12345_mapping = mapping.filter(
            mapping.census_tract == "12345"
        ).first()
        assert tract_12345_mapping.supertract_id == "ST1"
        
        tract_12347_mapping = mapping.filter(
            mapping.census_tract == "12347"
        ).first()
        assert tract_12347_mapping.supertract_id == "ST2"
    
    def test_min_half_pairs_threshold(self, spark):
        """Test different minimum half-pairs thresholds"""
        # Test with different thresholds
        algo_20 = SupertractAlgorithm(spark, min_half_pairs=20)
        algo_50 = SupertractAlgorithm(spark, min_half_pairs=50)
        
        # Create test data
        data = [
            ("12345", "19100", 2020, 30),
            ("12346", "19100", 2020, 35),
            ("12347", "19100", 2020, 40),
        ]
        
        half_pairs = spark.createDataFrame(
            data, ["census_tract", "cbsa_code", "year", "total_half_pairs"]
        )
        
        geographic = spark.createDataFrame([
            ("12345", "19100", 40.0, -74.0, ["12346"]),
            ("12346", "19100", 40.1, -74.1, ["12345", "12347"]),
            ("12347", "19100", 40.2, -74.2, ["12346"]),
        ], DataSchemas.GEOGRAPHIC_SCHEMA)
        
        # With threshold 20, all tracts should be independent
        supertracts_20 = algo_20.create_supertracts(half_pairs, geographic, 2020)
        assert supertracts_20.count() == 3
        
        # With threshold 50, all tracts should merge
        supertracts_50 = algo_50.create_supertracts(half_pairs, geographic, 2020)
        assert supertracts_50.count() < 3