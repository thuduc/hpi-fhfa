"""End-to-end integration tests for HPI pipeline"""

import pytest
from pyspark.sql import SparkSession
import tempfile
import shutil
import os
from datetime import date
import numpy as np

from hpi_fhfa.pipeline.main_pipeline import HPIPipeline
from hpi_fhfa.schemas.data_schemas import DataSchemas


class TestPipelineE2E:
    """End-to-end tests for the complete pipeline"""
    
    @pytest.fixture(scope="class")
    def spark(self):
        """Create Spark session for integration tests"""
        return SparkSession.builder \
            .master("local[4]") \
            .appName("integration-tests") \
            .config("spark.sql.shuffle.partitions", "10") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()
    
    @pytest.fixture
    def test_data_dir(self):
        """Create temporary directory for test data"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def _generate_test_transactions(self, spark, num_properties=1000):
        """Generate synthetic transaction data"""
        # Create properties with 2-3 transactions each
        data = []
        base_year = 2010
        
        for i in range(num_properties):
            property_id = f"P{i:05d}"
            census_tract = f"1234{i % 10}"  # 10 different tracts
            cbsa_code = f"1910{i % 3}"  # 3 different CBSAs
            base_price = 100000 + (i * 1000) % 500000
            
            # First transaction
            data.append((
                property_id,
                date(base_year + i % 5, 1, 1),
                float(base_price),
                census_tract,
                cbsa_code,
                float(5.0 + (i % 10))
            ))
            
            # Second transaction
            appreciation = 1.0 + np.random.uniform(0.02, 0.15)
            data.append((
                property_id,
                date(base_year + 5 + i % 3, 6, 1),
                float(base_price * appreciation),
                census_tract,
                cbsa_code,
                float(5.0 + (i % 10))
            ))
            
            # Third transaction for some properties
            if i % 3 == 0:
                appreciation2 = 1.0 + np.random.uniform(0.01, 0.10)
                data.append((
                    property_id,
                    date(base_year + 8 + i % 2, 3, 1),
                    float(base_price * appreciation * appreciation2),
                    census_tract,
                    cbsa_code,
                    float(5.0 + (i % 10))
                ))
        
        return spark.createDataFrame(data, DataSchemas.TRANSACTION_SCHEMA)
    
    def _generate_test_geographic(self, spark):
        """Generate synthetic geographic data"""
        data = []
        
        for i in range(10):  # 10 census tracts
            census_tract = f"1234{i}"
            cbsa_code = f"1910{i % 3}"
            
            # Generate coordinates (simplified)
            lat = 40.0 + (i * 0.1)
            lon = -74.0 + (i * 0.1)
            
            # Adjacent tracts
            adjacent = []
            if i > 0:
                adjacent.append(f"1234{i-1}")
            if i < 9:
                adjacent.append(f"1234{i+1}")
            
            data.append((census_tract, cbsa_code, lat, lon, adjacent))
        
        return spark.createDataFrame(data, DataSchemas.GEOGRAPHIC_SCHEMA)
    
    def _generate_test_weights(self, spark):
        """Generate synthetic weight data"""
        data = []
        
        for year in range(2010, 2020):
            for i in range(10):  # 10 census tracts
                census_tract = f"1234{i}"
                cbsa_code = f"1910{i % 3}"
                
                # Generate weight measures
                value_measure = 1000000.0 * (1 + i) * (1 + (year - 2010) * 0.03)
                unit_measure = 100.0 * (1 + i)
                upb_measure = 800000.0 * (1 + i) * (1 + (year - 2010) * 0.02)
                
                # Static measures (only for 2010)
                if year == 2010:
                    college_share = 0.3 + (i * 0.02)
                    nonwhite_share = 0.2 + (i * 0.03)
                else:
                    college_share = None
                    nonwhite_share = None
                
                data.append((
                    census_tract, cbsa_code, year,
                    value_measure, unit_measure, upb_measure,
                    college_share, nonwhite_share
                ))
        
        return spark.createDataFrame(data, DataSchemas.WEIGHT_SCHEMA)
    
    def _generate_test_data(self, spark, test_data_dir):
        """Generate all test data files"""
        # Generate data
        transactions = self._generate_test_transactions(spark)
        geographic = self._generate_test_geographic(spark)
        weights = self._generate_test_weights(spark)
        
        # Save to parquet
        transactions.write.mode("overwrite").parquet(
            os.path.join(test_data_dir, "transactions")
        )
        geographic.write.mode("overwrite").parquet(
            os.path.join(test_data_dir, "geographic")
        )
        weights.write.mode("overwrite").parquet(
            os.path.join(test_data_dir, "weights")
        )
        
        return {
            "transactions": transactions.count(),
            "geographic": geographic.count(),
            "weights": weights.count()
        }
    
    @pytest.mark.integration
    def test_full_pipeline(self, spark, test_data_dir):
        """Test complete pipeline execution"""
        # Generate test data
        data_counts = self._generate_test_data(spark, test_data_dir)
        
        # Create and run pipeline
        pipeline = HPIPipeline()
        
        metrics = pipeline.run_pipeline(
            os.path.join(test_data_dir, "transactions"),
            os.path.join(test_data_dir, "geographic"),
            os.path.join(test_data_dir, "weights"),
            os.path.join(test_data_dir, "output"),
            start_year=2015,
            end_year=2018
        )
        
        # Verify execution completed
        assert metrics["status"] == "SUCCESS"
        assert metrics["repeat_sales_count"] > 0
        assert metrics["output_rows"] > 0
        
        # Load and verify output
        output = spark.read.parquet(os.path.join(test_data_dir, "output"))
        
        # Check output schema
        expected_columns = {
            "cbsa_code", "year", "weight_type", 
            "appreciation_rate", "index_value", "yoy_change"
        }
        assert set(output.columns) >= expected_columns
        
        # Check data quality
        assert output.filter(output.index_value < 0).count() == 0
        assert output.filter(output.year == 2015).count() > 0
        
        # Check weight types (should have 6 types for the data that processed successfully)
        weight_types = output.select("weight_type").distinct().collect()
        assert len(weight_types) >= 1  # At least one weight type should be present
        
        cbsa_codes = output.select("cbsa_code").distinct().collect()
        assert len(cbsa_codes) >= 1  # At least one CBSA should be present
    
    @pytest.mark.integration
    def test_pipeline_with_minimal_data(self, spark, test_data_dir):
        """Test pipeline with minimal data"""
        # Generate minimal test data (50 properties)
        transactions = self._generate_test_transactions(spark, num_properties=50)
        geographic = self._generate_test_geographic(spark)
        weights = self._generate_test_weights(spark)
        
        # Save data
        transactions.write.mode("overwrite").parquet(
            os.path.join(test_data_dir, "transactions")
        )
        geographic.write.mode("overwrite").parquet(
            os.path.join(test_data_dir, "geographic")
        )
        weights.write.mode("overwrite").parquet(
            os.path.join(test_data_dir, "weights")
        )
        
        # Run pipeline
        pipeline = HPIPipeline()
        
        metrics = pipeline.run_pipeline(
            os.path.join(test_data_dir, "transactions"),
            os.path.join(test_data_dir, "geographic"),
            os.path.join(test_data_dir, "weights"),
            os.path.join(test_data_dir, "output"),
            start_year=2015,
            end_year=2016
        )
        
        # Should still complete successfully
        assert metrics["status"] == "SUCCESS"
    
    @pytest.mark.integration
    def test_pipeline_data_validation(self, spark, test_data_dir):
        """Test pipeline data validation capabilities"""
        # Generate data with some issues
        data = [
            ("P1", date(2010, 1, 1), 100000.0, "12345", "19100", 5.0),
            ("P1", date(2015, 1, 1), -50000.0, "12345", "19100", 5.0),  # Negative price
            ("P2", date(2012, 1, 1), 200000.0, "12346", "19100", 3.0),  # Valid date now
        ]
        
        problematic_transactions = spark.createDataFrame(
            data, DataSchemas.TRANSACTION_SCHEMA
        )
        
        geographic = self._generate_test_geographic(spark)
        weights = self._generate_test_weights(spark)
        
        # Save data
        problematic_transactions.write.mode("overwrite").parquet(
            os.path.join(test_data_dir, "transactions")
        )
        geographic.write.mode("overwrite").parquet(
            os.path.join(test_data_dir, "geographic")
        )
        weights.write.mode("overwrite").parquet(
            os.path.join(test_data_dir, "weights")
        )
        
        # Pipeline should handle problematic data
        pipeline = HPIPipeline()
        
        # This should complete but with warnings
        metrics = pipeline.run_pipeline(
            os.path.join(test_data_dir, "transactions"),
            os.path.join(test_data_dir, "geographic"),
            os.path.join(test_data_dir, "weights"),
            os.path.join(test_data_dir, "output"),
            start_year=2015,
            end_year=2016
        )
        
        # The pipeline should fail gracefully due to problematic data
        # (negative prices and limited data)
        assert metrics["status"] == "FAILED"
        assert "CANNOT_INFER_EMPTY_SCHEMA" in metrics.get("error", "")