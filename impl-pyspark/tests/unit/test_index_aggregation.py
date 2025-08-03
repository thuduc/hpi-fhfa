"""Unit tests for index aggregation module"""

import pytest
from datetime import date
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from hpi_fhfa.core.index_aggregation import IndexAggregator


class TestIndexAggregator:
    
    @pytest.fixture
    def index_aggregator(self, spark):
        """Create IndexAggregator instance"""
        return IndexAggregator(spark)
    
    @pytest.fixture
    def sample_supertract_indices(self, spark):
        """Create sample supertract BMN results"""
        data = [
            # CBSA 19100 - Year 2020
            ("ST1", "19100", 2020, 0.05, 0.92, 0.02, 50),
            ("ST2", "19100", 2020, 0.03, 0.88, 0.03, 40),
            ("ST3", "19100", 2020, 0.06, 0.95, 0.01, 60),
            
            # CBSA 19200 - Year 2020
            ("ST4", "19200", 2020, 0.04, 0.90, 0.02, 45),
            ("ST5", "19200", 2020, 0.02, 0.85, 0.04, 35),
        ]
        
        schema = ["supertract_id", "cbsa_code", "year", "appreciation_rate", 
                  "r2", "rmse", "num_observations"]
        
        return spark.createDataFrame(data, schema)
    
    @pytest.fixture
    def sample_supertracts(self, spark):
        """Create sample supertract data"""
        data = [
            ("ST1", "19100", ["12345", "12346"], 100, 2),
            ("ST2", "19100", ["12347"], 40, 1),
            ("ST3", "19100", ["12348", "12349"], 120, 2),
            ("ST4", "19200", ["12350", "12351"], 90, 2),
            ("ST5", "19200", ["12352"], 35, 1),
        ]
        
        schema = ["supertract_id", "cbsa_code", "tract_list", "min_half_pairs", "num_tracts"]
        
        return spark.createDataFrame(data, schema)
    
    @pytest.fixture
    def sample_weight_data(self, spark):
        """Create sample weight data"""
        data = []
        
        # Time-varying weights for 2020
        for tract in ["12345", "12346", "12347", "12348", "12349", "12350", "12351", "12352"]:
            cbsa = "19100" if int(tract) < 12350 else "19200"
            data.append((
                tract, cbsa, 2020,
                1000000.0 + int(tract) * 1000,  # value_measure
                100.0 + int(tract) % 10,        # unit_measure
                800000.0 + int(tract) * 800,    # upb_measure
                None, None  # college/nonwhite only for 2010
            ))
        
        # Static weights for 2010
        for tract in ["12345", "12346", "12347", "12348", "12349", "12350", "12351", "12352"]:
            cbsa = "19100" if int(tract) < 12350 else "19200"
            data.append((
                tract, cbsa, 2010,
                None, None, None,  # No time-varying measures for 2010
                0.3 + (int(tract) % 10) * 0.01,  # college_share
                0.2 + (int(tract) % 10) * 0.01   # nonwhite_share
            ))
        
        schema = ["census_tract", "cbsa_code", "year", "value_measure", 
                  "unit_measure", "upb_measure", "college_share", "nonwhite_share"]
        
        return spark.createDataFrame(data, schema)
    
    def test_calculate_weights_sample(self, index_aggregator, sample_supertracts):
        """Test sample weight calculation"""
        weights = index_aggregator.calculate_weights(
            sample_supertracts, 
            None,  # No weight data needed for sample weights
            "sample", 
            2020
        )
        
        # Check structure
        assert "supertract_id" in weights.columns
        assert "weight" in weights.columns
        
        # Check normalization within CBSA
        cbsa_19100_weights = weights.filter(
            F.col("cbsa_code") == "19100"
        ).agg(F.sum("weight")).collect()[0][0]
        
        assert abs(cbsa_19100_weights - 1.0) < 0.001
    
    def test_calculate_weights_value(self, index_aggregator, sample_supertracts, sample_weight_data):
        """Test value-based weight calculation"""
        weights = index_aggregator.calculate_weights(
            sample_supertracts,
            sample_weight_data,
            "value",
            2020
        )
        
        # Should have weights for all supertracts
        assert weights.count() == sample_supertracts.count()
        
        # Weights should sum to 1 within each CBSA
        for cbsa in ["19100", "19200"]:
            total = weights.filter(
                F.col("cbsa_code") == cbsa
            ).agg(F.sum("weight")).collect()[0][0]
            assert abs(total - 1.0) < 0.001
    
    def test_aggregate_city_index(self, index_aggregator, sample_supertract_indices, spark):
        """Test city-level aggregation"""
        # Create simple equal weights
        weights_data = [
            ("ST1", "19100", 0.33),
            ("ST2", "19100", 0.33),
            ("ST3", "19100", 0.34),
        ]
        weights = spark.createDataFrame(weights_data, ["supertract_id", "cbsa_code", "weight"])
        
        result = index_aggregator.aggregate_city_index(
            sample_supertract_indices,
            weights,
            "19100",
            2020
        )
        
        # Check result structure
        assert "appreciation_rate" in result
        assert "num_supertracts" in result
        assert "coverage" in result
        
        # Check values
        assert result["num_supertracts"] == 3
        assert abs(result["coverage"] - 1.0) < 0.001
        
        # Weighted average should be around 0.047
        expected = 0.05 * 0.33 + 0.03 * 0.33 + 0.06 * 0.34
        assert abs(result["appreciation_rate"] - expected) < 0.001
    
    def test_construct_index_series(self, index_aggregator):
        """Test index series construction"""
        appreciations = {
            2018: 0.00,  # Base year
            2019: 0.05,
            2020: 0.03,
            2021: 0.04
        }
        
        series = index_aggregator.construct_index_series(
            appreciations,
            "19100",
            "sample",
            base_year=2018
        )
        
        # Check structure
        assert "year" in series.columns
        assert "index_value" in series.columns
        assert "yoy_change" in series.columns
        
        # Check values
        rows = series.orderBy("year").collect()
        
        # 2018 (base)
        assert rows[0]["index_value"] == 100.0
        assert rows[0]["yoy_change"] == 0.0
        
        # 2019
        assert abs(rows[1]["index_value"] - 100 * np.exp(0.05)) < 0.01
        
        # 2020
        assert abs(rows[2]["index_value"] - 100 * np.exp(0.05 + 0.03)) < 0.01
        
        # 2021
        assert abs(rows[3]["index_value"] - 100 * np.exp(0.05 + 0.03 + 0.04)) < 0.01
    
    def test_process_all_weights(
        self, 
        index_aggregator, 
        sample_supertract_indices,
        sample_supertracts,
        sample_weight_data
    ):
        """Test processing all weight types"""
        results = index_aggregator.process_all_weights(
            sample_supertract_indices,
            sample_supertracts,
            sample_weight_data,
            "19100",
            2020
        )
        
        # Should have results for all weight types
        assert results.count() > 0
        
        # Check that we have multiple weight types
        weight_types = results.select("weight_type").distinct().collect()
        assert len(weight_types) >= 1
        
        # All results should be for the correct CBSA and year
        assert results.filter(F.col("cbsa_code") != "19100").count() == 0
        assert results.filter(F.col("year") != 2020).count() == 0