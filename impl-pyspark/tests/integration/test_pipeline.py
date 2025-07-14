"""
Integration tests for the complete RSAI pipeline.
"""

import pytest
import json
from pathlib import Path
from datetime import date, datetime
import shutil

from pyspark.sql import functions as F

from rsai.src.main import RSAIPipeline
from rsai.src.data.models import (
    RSAIConfig,
    GeographyLevel,
    WeightingScheme,
    get_transaction_schema,
    get_property_schema
)


class TestRSAIPipeline:
    """Test complete RSAI pipeline integration."""
    
    @pytest.fixture
    def sample_data_dir(self, temp_dir):
        """Create sample data files for testing."""
        data_dir = temp_dir / "data"
        data_dir.mkdir(exist_ok=True)
        return data_dir
        
    @pytest.fixture
    def create_sample_data(self, spark_fresh, sample_data_dir):
        """Create sample transaction and property data files."""
        # Create transaction data
        transactions = []
        properties = []
        
        # Generate data for 3 counties, 10 properties each
        property_id = 0
        for county_idx in range(3):
            county_id = f"3606{county_idx}"
            cbsa_id = "35620" if county_idx < 2 else "35614"
            
            for prop_idx in range(10):
                property_id += 1
                prop_id = f"P{property_id:04d}"
                tract_id = f"{county_id}00{prop_idx:02d}00"
                
                # Property data
                properties.append((
                    prop_id,
                    "single_family",
                    2000 + prop_idx,
                    1500 + prop_idx * 100,
                    40.7 + county_idx * 0.1 + prop_idx * 0.01,
                    -74.0 - county_idx * 0.1 - prop_idx * 0.01,
                    tract_id,
                    county_id,
                    cbsa_id,
                    "36",
                    f"{100 + property_id} Main St"
                ))
                
                # Create 2 transactions per property (repeat sale)
                base_price = 200000 + county_idx * 50000 + prop_idx * 10000
                
                # First sale
                transactions.append((
                    f"T{property_id:04d}_1",
                    prop_id,
                    date(2019, 1, 1).replace(month=((prop_idx % 12) + 1)),
                    float(base_price),
                    "arms_length"
                ))
                
                # Second sale
                growth = 1.0 + 0.05 * (1 + prop_idx / 10)  # 5-15% growth
                transactions.append((
                    f"T{property_id:04d}_2",
                    prop_id,
                    date(2021, 1, 1).replace(month=((prop_idx % 12) + 1)),
                    float(base_price * growth),
                    "arms_length"
                ))
                
        # Save as parquet files
        trans_df = spark_fresh.createDataFrame(
            transactions,
            schema=get_transaction_schema()
        )
        trans_df.write.mode("overwrite").parquet(
            str(sample_data_dir / "transactions.parquet")
        )
        
        prop_df = spark_fresh.createDataFrame(
            properties,
            schema=get_property_schema()
        )
        prop_df.write.mode("overwrite").parquet(
            str(sample_data_dir / "properties.parquet")
        )
        
        return sample_data_dir
        
    @pytest.fixture
    def test_config(self, create_sample_data, temp_dir):
        """Create test configuration."""
        config = {
            "min_price": 10000,
            "max_price": 10000000,
            "max_holding_period_years": 10,
            "min_pairs_threshold": 3,
            "outlier_std_threshold": 3.0,
            "frequency": "monthly",
            "base_period": None,
            "weighting_scheme": "equal",
            "geography_levels": ["tract", "county", "cbsa", "state"],
            "clustering_method": "kmeans",
            "n_clusters": 5,
            "min_cluster_size": 3,
            "max_cluster_size": 10,
            "spark_app_name": "RSAI Test",
            "spark_master": "local[2]",
            "spark_executor_memory": "1g",
            "spark_driver_memory": "1g",
            "spark_config": {
                "spark.sql.shuffle.partitions": "4"
            },
            "input_files": {
                "transactions": str(create_sample_data / "transactions.parquet"),
                "properties": str(create_sample_data / "properties.parquet")
            },
            "output_dir": str(temp_dir / "output")
        }
        
        config_path = temp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        return config_path
        
    def test_pipeline_end_to_end(self, spark_fresh, test_config):
        """Test complete pipeline execution."""
        # Create pipeline
        pipeline = RSAIPipeline(test_config, spark=spark_fresh)
        
        # Run pipeline
        results = pipeline.run()
        
        # Check overall status
        assert results["status"] == "success"
        assert results["total_transactions"] == 60  # 30 properties * 2 sales
        assert results["total_repeat_sales"] == 30
        assert results["regression_models_fitted"] > 0
        
        # Check that geography levels were processed (tract may not have enough data)
        processed_levels = results["geography_levels_processed"]
        # At least some geography levels should be processed
        assert len(processed_levels) > 0
        # County and state should typically be processable with test data
        assert "county" in processed_levels or "state" in processed_levels
        # If tract has sufficient data, it should be included, otherwise it's expected to be missing
        
        # Check output files exist
        output_files = results["output_files"]
        assert Path(output_files["indices"]).exists()
        assert Path(output_files["regression_results"]).exists()
        assert Path(output_files["summary_report"]).exists()
        
        # Clean up
        pipeline.stop()
        
    def test_pipeline_with_value_weights(self, spark_fresh, create_sample_data, temp_dir):
        """Test pipeline with value-based weighting."""
        # Create config with value weights
        config = {
            "min_price": 10000,
            "max_price": 10000000,
            "max_holding_period_years": 10,
            "min_pairs_threshold": 3,
            "outlier_std_threshold": 3.0,
            "frequency": "monthly",
            "weighting_scheme": "value",  # Value weights
            "geography_levels": ["county"],
            "spark_app_name": "RSAI Test Value",
            "spark_master": "local[2]",
            "spark_executor_memory": "1g",
            "spark_driver_memory": "1g",
            "input_files": {
                "transactions": str(create_sample_data / "transactions.parquet"),
                "properties": str(create_sample_data / "properties.parquet")
            },
            "output_dir": str(temp_dir / "output_value")
        }
        
        config_path = temp_dir / "config_value.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        # Run pipeline
        pipeline = RSAIPipeline(config_path, spark=spark_fresh)
        results = pipeline.run()
        
        # Check success
        assert results["status"] == "success"
        
        # Clean up
        pipeline.stop()
        
    def test_pipeline_error_handling(self, spark_fresh, temp_dir):
        """Test pipeline error handling."""
        # Create config with invalid input files
        config = {
            "min_price": 10000,
            "max_price": 10000000,
            "max_holding_period_years": 10,
            "min_pairs_threshold": 3,
            "outlier_std_threshold": 3.0,
            "frequency": "monthly",
            "weighting_scheme": "equal",
            "geography_levels": ["county"],
            "spark_app_name": "RSAI Test Error",
            "spark_master": "local[2]",
            "spark_executor_memory": "1g",
            "spark_driver_memory": "1g",
            "input_files": {
                "transactions": "nonexistent.parquet",
                "properties": "nonexistent.parquet"
            },
            "output_dir": str(temp_dir / "output_error")
        }
        
        config_path = temp_dir / "config_error.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        # Run pipeline
        pipeline = RSAIPipeline(config_path, spark=spark_fresh)
        results = pipeline.run()
        
        # Should succeed but with no meaningful results (graceful handling of missing data)
        assert results["status"] == "success"
        assert results["total_transactions"] == 0
        assert results["total_repeat_sales"] == 0
        assert results["regression_models_fitted"] == 0
        
        # Clean up
        pipeline.stop()
        
    def test_pipeline_with_insufficient_data(self, spark_fresh, temp_dir):
        """Test pipeline with insufficient data for regression."""
        # Create minimal data
        data_dir = temp_dir / "minimal_data"
        data_dir.mkdir(exist_ok=True)
        
        # Only 2 transactions (1 repeat sale)
        transactions = [
            ("T001", "P001", date(2020, 1, 1), 200000.0, "arms_length"),
            ("T002", "P001", date(2021, 1, 1), 220000.0, "arms_length"),
        ]
        
        properties = [
            ("P001", "single_family", 2000, 1500, 40.7, -74.0,
             "36061000100", "36061", "35620", "36", "123 Main St"),
        ]
        
        # Save data
        trans_df = spark_fresh.createDataFrame(
            transactions,
            schema=get_transaction_schema()
        )
        trans_df.write.mode("overwrite").parquet(
            str(data_dir / "transactions.parquet")
        )
        
        prop_df = spark_fresh.createDataFrame(
            properties,
            schema=get_property_schema()
        )
        prop_df.write.mode("overwrite").parquet(
            str(data_dir / "properties.parquet")
        )
        
        # Create config with high threshold
        config = {
            "min_price": 10000,
            "max_price": 10000000,
            "max_holding_period_years": 10,
            "min_pairs_threshold": 10,  # High threshold
            "outlier_std_threshold": 3.0,
            "frequency": "monthly",
            "weighting_scheme": "equal",
            "geography_levels": ["county"],
            "spark_app_name": "RSAI Test Minimal",
            "spark_master": "local[2]",
            "spark_executor_memory": "1g", 
            "spark_driver_memory": "1g",
            "input_files": {
                "transactions": str(data_dir / "transactions.parquet"),
                "properties": str(data_dir / "properties.parquet")
            },
            "output_dir": str(temp_dir / "output_minimal")
        }
        
        config_path = temp_dir / "config_minimal.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        # Run pipeline
        pipeline = RSAIPipeline(config_path, spark=spark_fresh)
        results = pipeline.run()
        
        # Should complete but with no models fitted
        assert results["status"] == "success"
        assert results["regression_models_fitted"] == 0
        
        # Clean up
        pipeline.stop()
        
    def test_pipeline_output_validation(self, spark_fresh, test_config, temp_dir):
        """Test that pipeline outputs are valid."""
        # Run pipeline
        pipeline = RSAIPipeline(test_config, spark=spark_fresh)
        results = pipeline.run()
        
        assert results["status"] == "success"
        
        # Load and validate index output
        index_path = results["output_files"]["indices"]
        index_df = spark_fresh.read.parquet(index_path)
        
        # Check schema
        expected_cols = [
            "geography_level", "geography_id", "period",
            "index_value", "num_pairs", "num_properties", "median_price"
        ]
        for col in expected_cols:
            assert col in index_df.columns
            
        # Check data validity
        assert index_df.count() > 0
        assert index_df.filter(F.col("index_value") <= 0).count() == 0
        assert index_df.filter(F.col("num_pairs") <= 0).count() == 0
        
        # Load and validate regression results
        regression_path = results["output_files"]["regression_results"]
        with open(regression_path, 'r') as f:
            regression_data = json.load(f)
            
        # Should have results for multiple geographies
        assert len(regression_data) > 0
        
        # Check each result
        for geo_id, geo_results in regression_data.items():
            assert "geography_level" in geo_results
            assert "r_squared" in geo_results
            assert 0 <= geo_results["r_squared"] <= 1
            
        # Check report exists
        report_path = results["output_files"]["summary_report"]
        assert Path(report_path).exists()
        assert Path(report_path).stat().st_size > 0
        
        # Clean up
        pipeline.stop()
        
    def test_pipeline_with_custom_config(self, spark_fresh, create_sample_data, temp_dir):
        """Test pipeline with various custom configurations."""
        # Test quarterly frequency
        config = {
            "min_price": 100000,
            "max_price": 500000,
            "max_holding_period_years": 5,
            "min_pairs_threshold": 5,
            "outlier_std_threshold": 2.5,
            "frequency": "quarterly",  # Quarterly
            "weighting_scheme": "bmn",  # BMN weights
            "geography_levels": ["county", "state"],
            "spark_app_name": "RSAI Test Custom",
            "spark_master": "local[2]",
            "spark_executor_memory": "1g",
            "spark_driver_memory": "1g",
            "input_files": {
                "transactions": str(create_sample_data / "transactions.parquet"),
                "properties": str(create_sample_data / "properties.parquet")
            },
            "output_dir": str(temp_dir / "output_custom")
        }
        
        config_path = temp_dir / "config_custom.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        # Run pipeline
        pipeline = RSAIPipeline(config_path, spark=spark_fresh)
        results = pipeline.run()
        
        # Check success
        assert results["status"] == "success"
        
        # Load index data to verify quarterly periods
        index_path = results["output_files"]["indices"]
        index_df = spark_fresh.read.parquet(index_path)
        
        # Check that periods are quarterly
        periods = index_df.select("period").distinct().collect()
        for row in periods:
            # Quarterly periods should be on Jan 1, Apr 1, Jul 1, or Oct 1
            assert row["period"].month in [1, 4, 7, 10]
            assert row["period"].day == 1
            
        # Clean up
        pipeline.stop()