"""Benchmark tests comparing PySpark implementation performance"""

import pytest
import time
import numpy as np
from datetime import date
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from hpi_fhfa.etl.data_processor import DataProcessor
from hpi_fhfa.core.bmn_regression import BMNRegression
from hpi_fhfa.core.supertract import SupertractAlgorithm
from hpi_fhfa.schemas.data_schemas import DataSchemas


class TestBenchmarks:
    """Benchmarks to compare different implementation approaches"""
    
    @pytest.fixture(scope="class")
    def spark_benchmark(self):
        """Create Spark session for benchmarking"""
        spark = SparkSession.builder \
            .appName("HPI-Benchmarks") \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "100") \
            .getOrCreate()
        
        yield spark
        spark.stop()
    
    def _generate_benchmark_data(self, spark, size="small"):
        """Generate benchmark datasets of different sizes"""
        if size == "small":
            num_properties = 1000
            num_tracts = 10
        elif size == "medium":
            num_properties = 10000
            num_tracts = 50
        else:  # large
            num_properties = 100000
            num_tracts = 200
            
        transactions = []
        np.random.seed(42)
        
        for i in range(num_properties):
            property_id = f"P{i:08d}"
            census_tract = f"{12345 + (i % num_tracts)}"
            cbsa_code = f"{19100 + (i % 5)}"
            
            # Generate 2-3 transactions
            base_price = 100000 + np.random.uniform(-50000, 200000)
            for j in range(np.random.choice([2, 3])):
                year = 2010 + j * np.random.randint(2, 5)
                if year > 2020:
                    break
                    
                price = base_price * (1 + np.random.uniform(-0.1, 0.2)) ** j
                transactions.append((
                    property_id,
                    date(year, np.random.randint(1, 13), 1),
                    float(price),
                    census_tract,
                    cbsa_code,
                    float(5.0 + np.random.uniform(0, 20))
                ))
        
        return spark.createDataFrame(transactions, DataSchemas.TRANSACTION_SCHEMA)
    
    @pytest.mark.benchmark
    def test_join_vs_broadcast_join(self, spark_benchmark):
        """Compare regular join vs broadcast join performance"""
        # Generate main dataset
        transactions = self._generate_benchmark_data(spark_benchmark, "medium")
        
        # Generate small dimension table (suitable for broadcast)
        tract_info = []
        for i in range(50):
            tract_info.append((
                f"{12345 + i}",
                f"Region_{i % 5}",
                f"Type_{i % 3}"
            ))
        
        dim_table = spark_benchmark.createDataFrame(
            tract_info, 
            ["census_tract", "region", "tract_type"]
        )
        
        # Test regular join
        start_time = time.time()
        regular_join = transactions.join(
            dim_table,
            on="census_tract",
            how="inner"
        )
        regular_count = regular_join.count()
        regular_time = time.time() - start_time
        
        # Test broadcast join
        start_time = time.time()
        broadcast_join = transactions.join(
            F.broadcast(dim_table),
            on="census_tract",
            how="inner"
        )
        broadcast_count = broadcast_join.count()
        broadcast_time = time.time() - start_time
        
        print(f"\nJoin Performance Comparison:")
        print(f"Regular join: {regular_time:.2f} seconds")
        print(f"Broadcast join: {broadcast_time:.2f} seconds")
        print(f"Speedup: {regular_time/broadcast_time:.2f}x")
        
        assert regular_count == broadcast_count
        # Broadcast should be faster for small dimension tables
        assert broadcast_time < regular_time
    
    @pytest.mark.benchmark
    def test_window_vs_groupby_performance(self, spark_benchmark):
        """Compare window functions vs group by for running calculations"""
        # Generate time series data
        data = []
        for tract_id in range(20):
            for year in range(2010, 2021):
                for month in range(1, 13):
                    data.append((
                        f"{12345 + tract_id}",
                        year,
                        month,
                        np.random.uniform(0, 0.02)  # Monthly appreciation
                    ))
        
        df = spark_benchmark.createDataFrame(
            data,
            ["census_tract", "year", "month", "appreciation"]
        )
        
        # Method 1: Window functions for cumulative calculation
        from pyspark.sql import Window
        
        start_time = time.time()
        window_spec = Window.partitionBy("census_tract").orderBy("year", "month")
        window_result = df.withColumn(
            "cumulative_appreciation",
            F.sum("appreciation").over(window_spec)
        )
        window_count = window_result.count()
        window_time = time.time() - start_time
        
        # Method 2: Self-join approach (less efficient)
        start_time = time.time()
        df1 = df.alias("df1")
        df2 = df.alias("df2")
        
        join_result = df1.join(
            df2,
            (F.col("df1.census_tract") == F.col("df2.census_tract")) &
            ((F.col("df2.year") < F.col("df1.year")) |
             ((F.col("df2.year") == F.col("df1.year")) & 
              (F.col("df2.month") <= F.col("df1.month")))),
            how="left"
        ).groupBy(
            F.col("df1.census_tract"),
            F.col("df1.year"),
            F.col("df1.month")
        ).agg(
            F.sum(F.col("df2.appreciation")).alias("cumulative_appreciation")
        )
        join_count = join_result.count()
        join_time = time.time() - start_time
        
        print(f"\nCumulative Calculation Performance:")
        print(f"Window function: {window_time:.2f} seconds")
        print(f"Self-join: {join_time:.2f} seconds")
        print(f"Speedup: {join_time/window_time:.2f}x")
        
        # Window functions should be significantly faster
        assert window_time < join_time
    
    @pytest.mark.benchmark
    def test_cache_effectiveness(self, spark_benchmark):
        """Test effectiveness of caching for iterative operations"""
        # Generate dataset
        transactions = self._generate_benchmark_data(spark_benchmark, "medium")
        processor = DataProcessor(spark_benchmark)
        
        repeat_sales = processor.create_repeat_sales_pairs(transactions)
        
        # Test without cache
        start_time = time.time()
        for year in range(2015, 2018):
            year_data = repeat_sales.filter(
                F.year("sale_date_2") == year
            )
            count = year_data.count()
            print(f"Year {year}: {count} pairs")
        no_cache_time = time.time() - start_time
        
        # Test with cache
        repeat_sales_cached = repeat_sales.cache()
        repeat_sales_cached.count()  # Force cache
        
        start_time = time.time()
        for year in range(2015, 2018):
            year_data = repeat_sales_cached.filter(
                F.year("sale_date_2") == year
            )
            count = year_data.count()
        cache_time = time.time() - start_time
        
        print(f"\nCache Performance:")
        print(f"Without cache: {no_cache_time:.2f} seconds")
        print(f"With cache: {cache_time:.2f} seconds")
        print(f"Speedup: {no_cache_time/cache_time:.2f}x")
        
        # Cached version should be faster for multiple iterations
        assert cache_time < no_cache_time
        
        # Cleanup
        repeat_sales_cached.unpersist()
    
    @pytest.mark.benchmark
    def test_partitioning_strategies(self, spark_benchmark):
        """Compare different partitioning strategies"""
        # Generate large dataset
        transactions = self._generate_benchmark_data(spark_benchmark, "large")
        
        # Strategy 1: Default partitioning
        start_time = time.time()
        default_df = transactions
        default_groups = default_df.groupBy("cbsa_code", "census_tract").count()
        default_count = default_groups.count()
        default_time = time.time() - start_time
        
        # Strategy 2: Repartition by key columns
        start_time = time.time()
        repartitioned_df = transactions.repartition(100, "cbsa_code", "census_tract")
        repartitioned_groups = repartitioned_df.groupBy("cbsa_code", "census_tract").count()
        repartitioned_count = repartitioned_groups.count()
        repartitioned_time = time.time() - start_time
        
        # Strategy 3: Coalesce for small result sets
        start_time = time.time()
        coalesced_df = transactions.coalesce(50)
        coalesced_groups = coalesced_df.groupBy("cbsa_code", "census_tract").count()
        coalesced_count = coalesced_groups.count()
        coalesced_time = time.time() - start_time
        
        print(f"\nPartitioning Strategy Comparison:")
        print(f"Default: {default_time:.2f} seconds")
        print(f"Repartitioned: {repartitioned_time:.2f} seconds")
        print(f"Coalesced: {coalesced_time:.2f} seconds")
        
        assert default_count == repartitioned_count == coalesced_count
    
    @pytest.mark.benchmark
    def test_udf_vs_native_functions(self, spark_benchmark):
        """Compare UDF performance vs native Spark functions"""
        # Generate data
        data = [(i, float(100000 + i * 1000), float(110000 + i * 1100)) 
                for i in range(100000)]
        df = spark_benchmark.createDataFrame(
            data, 
            ["id", "price1", "price2"]
        )
        
        # Method 1: UDF
        from pyspark.sql.functions import udf
        from pyspark.sql.types import DoubleType
        
        def calculate_appreciation_udf(price1, price2):
            if price1 > 0 and price2 > 0:
                import math
                return math.log(price2 / price1)
            return 0.0
        
        appreciation_udf = udf(calculate_appreciation_udf, DoubleType())
        
        start_time = time.time()
        udf_result = df.withColumn(
            "appreciation",
            appreciation_udf(F.col("price1"), F.col("price2"))
        )
        udf_count = udf_result.filter(F.col("appreciation") > 0).count()
        udf_time = time.time() - start_time
        
        # Method 2: Native Spark functions
        start_time = time.time()
        native_result = df.withColumn(
            "appreciation",
            F.when(
                (F.col("price1") > 0) & (F.col("price2") > 0),
                F.log(F.col("price2") / F.col("price1"))
            ).otherwise(0.0)
        )
        native_count = native_result.filter(F.col("appreciation") > 0).count()
        native_time = time.time() - start_time
        
        print(f"\nUDF vs Native Function Performance:")
        print(f"UDF: {udf_time:.2f} seconds")
        print(f"Native: {native_time:.2f} seconds")
        print(f"Speedup: {udf_time/native_time:.2f}x")
        
        assert udf_count == native_count
        # Native functions should be faster
        assert native_time < udf_time  # Native should be faster
        # In practice, native functions are typically 1.5-3x faster than UDFs