"""Performance tests for HPI-FHFA PySpark implementation"""

import pytest
import time
from datetime import datetime, date
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from hpi_fhfa.pipeline.main_pipeline import HPIPipeline
from hpi_fhfa.etl.data_processor import DataProcessor
from hpi_fhfa.core.bmn_regression import BMNRegression
from hpi_fhfa.core.supertract import SupertractAlgorithm
from hpi_fhfa.schemas.data_schemas import DataSchemas


class TestPerformance:
    """Performance tests to verify scalability and efficiency"""
    
    @pytest.fixture(scope="class")
    def spark_performance(self):
        """Create Spark session optimized for performance testing"""
        spark = SparkSession.builder \
            .appName("HPI-Performance-Tests") \
            .master("local[4]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        yield spark
        spark.stop()
    
    def _generate_large_dataset(self, spark, num_properties=100000, num_years=10):
        """Generate large synthetic dataset for performance testing"""
        print(f"Generating {num_properties:,} properties over {num_years} years...")
        
        # Generate transactions
        transactions = []
        np.random.seed(42)
        
        base_year = 2010
        num_tracts = 100
        num_cbsas = 10
        
        for i in range(num_properties):
            property_id = f"P{i:08d}"
            census_tract = f"{12345 + (i % num_tracts)}"
            cbsa_code = f"{19100 + (i % num_cbsas)}"
            base_price = 100000 + np.random.uniform(-50000, 200000)
            
            # Generate 2-3 transactions per property
            num_transactions = np.random.choice([2, 3], p=[0.7, 0.3])
            
            for j in range(num_transactions):
                year = base_year + j * np.random.randint(2, 5)
                if year > base_year + num_years:
                    break
                    
                price = base_price * (1 + np.random.uniform(-0.1, 0.2)) ** j
                
                transactions.append((
                    property_id,
                    date(year, np.random.randint(1, 13), np.random.randint(1, 28)),
                    float(price),
                    census_tract,
                    cbsa_code,
                    float(5.0 + np.random.uniform(0, 20))
                ))
        
        return spark.createDataFrame(transactions, DataSchemas.TRANSACTION_SCHEMA)
    
    @pytest.mark.performance
    def test_repeat_sales_creation_performance(self, spark_performance):
        """Test performance of repeat-sales pair creation"""
        # Generate test data
        transactions = self._generate_large_dataset(spark_performance, num_properties=50000)
        
        processor = DataProcessor(spark_performance)
        
        # Measure performance
        start_time = time.time()
        repeat_sales = processor.create_repeat_sales_pairs(transactions)
        count = repeat_sales.count()  # Force evaluation
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\nRepeat-sales creation: {count:,} pairs in {duration:.2f} seconds")
        print(f"Rate: {count/duration:.0f} pairs/second")
        
        # Performance assertions
        assert duration < 60  # Should complete in under 1 minute
        assert count > 10000  # Should create substantial number of pairs
    
    @pytest.mark.performance
    def test_filtering_performance(self, spark_performance):
        """Test performance of data quality filters"""
        # Generate test data with some extreme values
        transactions = self._generate_large_dataset(spark_performance, num_properties=30000)
        processor = DataProcessor(spark_performance)
        
        repeat_sales = processor.create_repeat_sales_pairs(transactions)
        initial_count = repeat_sales.count()
        
        # Measure filter performance
        start_time = time.time()
        filtered = processor.apply_filters(repeat_sales)
        final_count = filtered.count()
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\nFiltering: {initial_count:,} → {final_count:,} pairs in {duration:.2f} seconds")
        print(f"Rate: {initial_count/duration:.0f} pairs/second")
        
        # Performance assertions
        assert duration < 30  # Should complete quickly
        assert final_count > initial_count * 0.8  # Most data should pass filters
    
    @pytest.mark.performance
    def test_supertract_creation_performance(self, spark_performance):
        """Test performance of supertract algorithm"""
        # Generate half-pairs data
        half_pairs_data = []
        
        for tract_id in range(100):
            for year in range(2015, 2020):
                census_tract = f"{12345 + tract_id}"
                cbsa_code = f"{19100 + (tract_id % 10)}"
                half_pairs = np.random.poisson(30) + 10
                
                half_pairs_data.append((census_tract, cbsa_code, year, half_pairs))
        
        half_pairs = spark_performance.createDataFrame(
            half_pairs_data, 
            ["census_tract", "cbsa_code", "year", "total_half_pairs"]
        )
        
        # Generate geographic data
        geo_data = []
        for tract_id in range(100):
            census_tract = f"{12345 + tract_id}"
            cbsa_code = f"{19100 + (tract_id % 10)}"
            lat = 40.0 + tract_id * 0.01
            lon = -74.0 + tract_id * 0.01
            adjacent = [f"{12345 + tract_id - 1}", f"{12345 + tract_id + 1}"]
            
            geo_data.append((census_tract, cbsa_code, lat, lon, adjacent))
        
        geographic = spark_performance.createDataFrame(
            geo_data, DataSchemas.GEOGRAPHIC_SCHEMA
        )
        
        algo = SupertractAlgorithm(spark_performance)
        
        # Measure performance
        start_time = time.time()
        supertracts = algo.create_supertracts(half_pairs, geographic, 2018)
        count = supertracts.count()
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\nSupertract creation: {count} supertracts in {duration:.2f} seconds")
        
        # Performance assertions
        assert duration < 10  # Should complete quickly
        assert count > 0  # Should create some supertracts
    
    @pytest.mark.performance
    def test_bmn_regression_performance(self, spark_performance):
        """Test performance of BMN regression"""
        # Generate regression-ready data
        reg_data = []
        from pyspark.ml.linalg import Vectors
        
        # Create 10000 observations across 100 time periods
        for i in range(10000):
            label = np.random.uniform(0, 0.1)
            # Create sparse vector for time dummies
            period1 = np.random.randint(0, 50)
            period2 = period1 + np.random.randint(1, 50)
            features = Vectors.sparse(100, [period1, period2], [-1.0, 1.0])
            reg_data.append((label, features))
        
        regression_df = spark_performance.createDataFrame(reg_data, ["label", "features"])
        
        bmn = BMNRegression(spark_performance)
        
        # Measure performance
        start_time = time.time()
        results = bmn.estimate_bmn(regression_df)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\nBMN regression: 10,000 observations in {duration:.2f} seconds")
        
        # Performance assertions
        assert duration < 30  # Should complete in reasonable time
        assert results is not None
        assert "coefficients" in results
    
    @pytest.mark.performance
    @pytest.mark.integration
    def test_end_to_end_pipeline_performance(self, spark_performance, tmp_path):
        """Test end-to-end pipeline performance"""
        # Generate moderate-sized dataset
        print("\nGenerating test data for pipeline...")
        transactions = self._generate_large_dataset(
            spark_performance, 
            num_properties=10000,
            num_years=10
        )
        
        # Save test data
        trans_path = str(tmp_path / "transactions")
        transactions.write.mode("overwrite").parquet(trans_path)
        
        # Generate geographic data
        geo_data = []
        for i in range(100):
            census_tract = f"{12345 + i}"
            cbsa_code = f"{19100 + (i % 10)}"
            geo_data.append((census_tract, cbsa_code, 40.0 + i*0.01, -74.0 + i*0.01, []))
        
        geographic = spark_performance.createDataFrame(geo_data, DataSchemas.GEOGRAPHIC_SCHEMA)
        geo_path = str(tmp_path / "geographic")
        geographic.write.mode("overwrite").parquet(geo_path)
        
        # Generate weight data
        weight_data = []
        for year in range(2015, 2020):
            for i in range(100):
                census_tract = f"{12345 + i}"
                cbsa_code = f"{19100 + (i % 10)}"
                weight_data.append((
                    census_tract, cbsa_code, year,
                    1000000.0 * (1 + i*0.1), 100.0 + i, 800000.0 * (1 + i*0.1),
                    0.3 if year == 2010 else None,
                    0.2 if year == 2010 else None
                ))
        
        weights = spark_performance.createDataFrame(weight_data, DataSchemas.WEIGHT_SCHEMA)
        weight_path = str(tmp_path / "weights")
        weights.write.mode("overwrite").parquet(weight_path)
        
        # Run pipeline
        pipeline = HPIPipeline()
        
        start_time = time.time()
        metrics = pipeline.run_pipeline(
            trans_path,
            geo_path,
            weight_path,
            str(tmp_path / "output"),
            start_year=2016,
            end_year=2018
        )
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\nEnd-to-end pipeline: {duration:.2f} seconds")
        print(f"Repeat-sales processed: {metrics.get('repeat_sales_count', 0):,}")
        print(f"Output rows generated: {metrics.get('output_rows', 0):,}")
        
        # Performance assertions
        assert metrics["status"] == "SUCCESS"
        assert duration < 300  # Should complete in under 5 minutes
        
    def test_memory_usage(self, spark_performance):
        """Test memory usage patterns"""
        # This is a simple test to ensure no memory leaks
        # In production, you'd want more sophisticated memory profiling
        
        processor = DataProcessor(spark_performance)
        
        # Run multiple iterations
        for i in range(3):
            transactions = self._generate_large_dataset(
                spark_performance, 
                num_properties=5000
            )
            
            repeat_sales = processor.create_repeat_sales_pairs(transactions)
            filtered = processor.apply_filters(repeat_sales)
            half_pairs = processor.calculate_half_pairs(filtered)
            
            # Force evaluation
            count = half_pairs.count()
            print(f"Iteration {i+1}: {count} half-pairs")
            
            # Clear cache
            spark_performance.catalog.clearCache()
        
        # If we get here without OOM, test passes
        assert True