# House Price Index (HPI) - FHFA Method Implementation Plan
## PySpark and MLlib Technical Stack

## 1. Executive Summary

This implementation plan details the construction of the FHFA Repeat-Sales Aggregation Index (RSAI) using PySpark and MLlib. The solution processes 63.3 million repeat-sales pairs across 63,122 census tracts and 581 CBSAs from 1975-2021, producing balanced panel indices with dynamic aggregation and flexible weighting schemes.

### Key Technologies
- **Apache Spark 3.5+**: Distributed data processing
- **PySpark SQL**: DataFrame operations and window functions
- **MLlib**: Distributed linear regression for BMN estimation
- **Delta Lake**: ACID transactions and time travel for data versioning
- **pytest-spark**: Unit testing framework
- **Great Expectations**: Data quality validation

## 2. Architecture Overview

### 2.1 System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                       Data Sources                          │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │Transaction │  │  Geographic  │  │  Census/Market   │  │
│  │   Data     │  │    Data      │  │     Weights      │  │
│  └────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PySpark ETL Layer                        │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Data     │  │    Data      │  │     Data         │  │
│  │  Ingestion │  │  Validation  │  │  Transformation  │  │
│  └────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Core Processing Engine                     │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Price    │  │  Supertract  │  │      BMN         │  │
│  │ Relatives  │  │  Algorithm   │  │   Regression     │  │
│  └────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Weight    │  │    Index     │  │   Validation     │  │
│  │Calculation │  │ Aggregation  │  │   & Quality      │  │
│  └────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Output Layer                           │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │Tract-Level │  │  City-Level  │  │   Analytics      │  │
│  │  Indices   │  │   Indices    │  │   Dashboard      │  │
│  └────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Architecture
```python
# Distributed Processing Pipeline
Raw Data → Spark DataFrames → Partitioned by CBSA/Year
         → Window Functions for Time Series
         → Broadcast Variables for Geographic Data
         → Custom UDAFs for Aggregation
         → Delta Tables for Output
```

## 3. Core Components Implementation

### 3.1 Data Models

```python
# schemas/transaction_schema.py
from pyspark.sql.types import *

class DataSchemas:
    TRANSACTION_SCHEMA = StructType([
        StructField("property_id", StringType(), False),
        StructField("transaction_date", DateType(), False),
        StructField("transaction_price", DoubleType(), False),
        StructField("census_tract", StringType(), False),
        StructField("cbsa_code", StringType(), False),
        StructField("distance_to_cbd", DoubleType(), False)
    ])
    
    GEOGRAPHIC_SCHEMA = StructType([
        StructField("census_tract", StringType(), False),
        StructField("cbsa_code", StringType(), False),
        StructField("centroid_lat", DoubleType(), False),
        StructField("centroid_lon", DoubleType(), False),
        StructField("adjacent_tracts", ArrayType(StringType()), False)
    ])
    
    REPEAT_SALES_SCHEMA = StructType([
        StructField("property_id", StringType(), False),
        StructField("sale_date_1", DateType(), False),
        StructField("sale_price_1", DoubleType(), False),
        StructField("sale_date_2", DateType(), False),
        StructField("sale_price_2", DoubleType(), False),
        StructField("census_tract", StringType(), False),
        StructField("cbsa_code", StringType(), False),
        StructField("price_relative", DoubleType(), False),
        StructField("time_diff_years", DoubleType(), False),
        StructField("cagr", DoubleType(), False)
    ])
```

### 3.2 Data Processing Pipeline

```python
# etl/data_processor.py
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import Dict, List, Tuple
import logging

class DataProcessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(__name__)
        
    def create_repeat_sales_pairs(self, transactions: DataFrame) -> DataFrame:
        """Create repeat-sales pairs using self-join with window functions"""
        
        # Add transaction sequence numbers per property
        window_spec = Window.partitionBy("property_id").orderBy("transaction_date")
        
        transactions_numbered = transactions.withColumn(
            "transaction_seq", F.row_number().over(window_spec)
        )
        
        # Self-join to create pairs
        t1 = transactions_numbered.alias("t1")
        t2 = transactions_numbered.alias("t2")
        
        repeat_sales = t1.join(
            t2,
            (F.col("t1.property_id") == F.col("t2.property_id")) &
            (F.col("t1.transaction_seq") == F.col("t2.transaction_seq") - 1),
            "inner"
        ).select(
            F.col("t1.property_id").alias("property_id"),
            F.col("t1.transaction_date").alias("sale_date_1"),
            F.col("t1.transaction_price").alias("sale_price_1"),
            F.col("t2.transaction_date").alias("sale_date_2"),
            F.col("t2.transaction_price").alias("sale_price_2"),
            F.col("t1.census_tract").alias("census_tract"),
            F.col("t1.cbsa_code").alias("cbsa_code"),
            F.col("t1.distance_to_cbd").alias("distance_to_cbd")
        )
        
        # Calculate price relatives and time differences
        repeat_sales = repeat_sales.withColumn(
            "price_relative", 
            F.log(F.col("sale_price_2")) - F.log(F.col("sale_price_1"))
        ).withColumn(
            "time_diff_years",
            F.datediff(F.col("sale_date_2"), F.col("sale_date_1")) / 365.25
        ).withColumn(
            "cagr",
            F.pow(F.col("sale_price_2") / F.col("sale_price_1"), 
                  1.0 / F.col("time_diff_years")) - 1
        )
        
        return repeat_sales
    
    def apply_filters(self, repeat_sales: DataFrame) -> DataFrame:
        """Apply data quality filters as per PRD specifications"""
        
        filtered = repeat_sales.filter(
            # Not in same 12-month period
            (F.year("sale_date_2") != F.year("sale_date_1")) |
            (F.month("sale_date_2") != F.month("sale_date_1"))
        ).filter(
            # CAGR filter
            (F.abs(F.col("cagr")) <= 0.30)
        ).filter(
            # Cumulative appreciation filter
            (F.col("sale_price_2") / F.col("sale_price_1")).between(0.25, 10.0)
        )
        
        self.logger.info(f"Filtered pairs: {repeat_sales.count()} → {filtered.count()}")
        return filtered
    
    def calculate_half_pairs(self, repeat_sales: DataFrame) -> DataFrame:
        """Calculate half-pairs for each tract-period combination"""
        
        # Create half-pairs for first sale
        half_pairs_1 = repeat_sales.select(
            F.col("census_tract"),
            F.col("cbsa_code"),
            F.year("sale_date_1").alias("year"),
            F.lit(1).alias("half_pairs")
        )
        
        # Create half-pairs for second sale
        half_pairs_2 = repeat_sales.select(
            F.col("census_tract"),
            F.col("cbsa_code"),
            F.year("sale_date_2").alias("year"),
            F.lit(1).alias("half_pairs")
        )
        
        # Union and aggregate
        half_pairs = half_pairs_1.union(half_pairs_2).groupBy(
            "census_tract", "cbsa_code", "year"
        ).agg(
            F.sum("half_pairs").alias("total_half_pairs")
        )
        
        return half_pairs
```

### 3.3 Supertract Algorithm

```python
# core/supertract.py
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.broadcast import Broadcast
from typing import Dict, Set, List
import numpy as np

class SupertractAlgorithm:
    def __init__(self, spark: SparkSession, min_half_pairs: int = 40):
        self.spark = spark
        self.min_half_pairs = min_half_pairs
        
    def create_supertracts(
        self, 
        half_pairs: DataFrame,
        geographic_data: DataFrame,
        year: int
    ) -> DataFrame:
        """Dynamic aggregation of census tracts into supertracts"""
        
        # Filter for specific year and adjacent year
        year_data = half_pairs.filter(
            F.col("year").isin([year, year - 1])
        ).groupBy("census_tract", "cbsa_code").agg(
            F.min(F.col("total_half_pairs")).alias("min_half_pairs")
        )
        
        # Broadcast geographic data for efficient joins
        geo_broadcast = self.spark.sparkContext.broadcast(
            geographic_data.collect()
        )
        
        # Initialize supertracts
        supertracts = year_data.withColumn(
            "supertract_id", F.col("census_tract")
        ).withColumn(
            "tract_list", F.array(F.col("census_tract"))
        )
        
        # Iterative aggregation
        iteration = 0
        while True:
            # Find tracts below threshold
            below_threshold = supertracts.filter(
                F.col("min_half_pairs") < self.min_half_pairs
            )
            
            if below_threshold.count() == 0:
                break
                
            # Custom UDAF for nearest neighbor aggregation
            merged = self._merge_with_nearest(
                below_threshold, supertracts, geo_broadcast
            )
            
            # Update supertracts
            supertracts = merged
            iteration += 1
            
            if iteration > 100:  # Safety check
                self.logger.warning("Max iterations reached in supertract creation")
                break
        
        return supertracts
    
    def _merge_with_nearest(
        self, 
        below_threshold: DataFrame,
        all_supertracts: DataFrame,
        geo_broadcast: Broadcast
    ) -> DataFrame:
        """Merge tracts with nearest neighbors using broadcast variable"""
        
        # UDF for finding nearest neighbor
        @F.udf(returnType=StringType())
        def find_nearest_neighbor(tract_id: str, cbsa: str, excluded: List[str]):
            geo_data = geo_broadcast.value
            
            # Get centroid of current tract
            current = next((g for g in geo_data if g["census_tract"] == tract_id), None)
            if not current:
                return None
                
            min_distance = float('inf')
            nearest = None
            
            # Find nearest tract in same CBSA
            for geo in geo_data:
                if (geo["cbsa_code"] == cbsa and 
                    geo["census_tract"] not in excluded):
                    
                    distance = haversine_distance(
                        current["centroid_lat"], current["centroid_lon"],
                        geo["centroid_lat"], geo["centroid_lon"]
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest = geo["census_tract"]
            
            return nearest
        
        # Apply merging logic
        merged = below_threshold.withColumn(
            "merge_target",
            find_nearest_neighbor(
                F.col("supertract_id"), 
                F.col("cbsa_code"),
                F.col("tract_list")
            )
        )
        
        # Aggregate merged tracts
        return self._aggregate_merged_tracts(merged, all_supertracts)
```

### 3.4 BMN Regression Implementation

```python
# core/bmn_regression.py
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from scipy.sparse import csr_matrix
import numpy as np

class BMNRegression:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        
    def prepare_regression_data(
        self, 
        repeat_sales: DataFrame,
        supertract: str,
        start_year: int,
        end_year: int
    ) -> DataFrame:
        """Prepare data for BMN regression with dummy variables"""
        
        # Filter for supertract
        supertract_data = repeat_sales.filter(
            F.col("supertract_id") == supertract
        )
        
        # Create time period columns
        periods = list(range(start_year, end_year + 1))
        
        # UDF to create dummy variable vector
        @F.udf(returnType=VectorUDT())
        def create_dummy_vector(year1: int, year2: int):
            # Create sparse vector for time dummies
            indices = []
            values = []
            
            # Find indices for sale years
            if year1 in periods:
                idx1 = periods.index(year1)
                indices.append(idx1)
                values.append(-1.0)
                
            if year2 in periods:
                idx2 = periods.index(year2)
                indices.append(idx2)
                values.append(1.0)
            
            return Vectors.sparse(len(periods), indices, values)
        
        # Prepare regression dataset
        regression_data = supertract_data.withColumn(
            "features",
            create_dummy_vector(
                F.year("sale_date_1"),
                F.year("sale_date_2")
            )
        ).select(
            "price_relative",
            "features"
        ).withColumnRenamed(
            "price_relative", "label"
        )
        
        return regression_data
    
    def estimate_bmn(
        self, 
        regression_data: DataFrame,
        elastic_net_param: float = 0.0,
        reg_param: float = 0.0
    ) -> Dict[str, float]:
        """Estimate BMN regression using MLlib LinearRegression"""
        
        # Configure regression
        lr = LinearRegression(
            elasticNetParam=elastic_net_param,
            regParam=reg_param,
            standardization=False,
            fitIntercept=False  # No intercept in BMN
        )
        
        # Fit model
        model = lr.fit(regression_data)
        
        # Extract coefficients
        coefficients = model.coefficients.toArray()
        
        # Calculate standard errors if needed
        summary = model.summary
        std_errors = np.sqrt(np.diag(summary.coefficientStandardErrors))
        
        return {
            "coefficients": coefficients,
            "std_errors": std_errors,
            "r2": summary.r2,
            "rmse": summary.rootMeanSquaredError
        }
    
    def calculate_appreciation_rates(
        self, 
        bmn_results: Dict[str, float],
        periods: List[int]
    ) -> DataFrame:
        """Calculate period-to-period appreciation rates"""
        
        coefficients = bmn_results["coefficients"]
        
        # Create appreciation rates
        appreciation_data = []
        for i in range(1, len(periods)):
            appreciation = coefficients[i] - coefficients[i-1]
            appreciation_data.append({
                "year": periods[i],
                "appreciation_rate": appreciation,
                "cumulative_index": np.exp(coefficients[i])
            })
        
        return self.spark.createDataFrame(appreciation_data)
```

### 3.5 Index Aggregation

```python
# core/index_aggregation.py
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from typing import Dict, List

class IndexAggregator:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.weight_types = [
            "sample", "value", "unit", "upb", "college", "nonwhite"
        ]
        
    def calculate_weights(
        self,
        supertract_data: DataFrame,
        weight_data: DataFrame,
        weight_type: str,
        year: int
    ) -> DataFrame:
        """Calculate normalized weights for aggregation"""
        
        if weight_type == "sample":
            # Sample weights based on half-pairs
            weights = supertract_data.groupBy("supertract_id").agg(
                F.sum("half_pairs").alias("weight_raw")
            )
        
        elif weight_type in ["value", "unit", "upb"]:
            # Time-varying weights
            weights = weight_data.filter(
                F.col("year") == year
            ).groupBy("supertract_id").agg(
                F.sum(F.col(f"{weight_type}_measure")).alias("weight_raw")
            )
            
        else:  # college, nonwhite
            # Static weights from 2010
            weights = weight_data.filter(
                F.col("year") == 2010
            ).groupBy("supertract_id").agg(
                F.sum(F.col(f"{weight_type}_share")).alias("weight_raw")
            )
        
        # Normalize weights
        total_weight = weights.agg(F.sum("weight_raw")).collect()[0][0]
        
        weights = weights.withColumn(
            "weight", F.col("weight_raw") / total_weight
        )
        
        return weights
    
    def aggregate_city_index(
        self,
        supertract_indices: DataFrame,
        weights: DataFrame,
        cbsa: str,
        year: int
    ) -> float:
        """Aggregate supertract indices to city level"""
        
        # Join indices with weights
        weighted_indices = supertract_indices.join(
            weights,
            on="supertract_id",
            how="inner"
        ).withColumn(
            "weighted_appreciation",
            F.col("appreciation_rate") * F.col("weight")
        )
        
        # Sum weighted appreciation
        city_appreciation = weighted_indices.agg(
            F.sum("weighted_appreciation").alias("city_appreciation")
        ).collect()[0]["city_appreciation"]
        
        return city_appreciation
    
    def construct_index_series(
        self,
        appreciations: Dict[int, float],
        base_year: int = 1989
    ) -> DataFrame:
        """Construct cumulative index series"""
        
        index_data = []
        cumulative_index = 100.0  # Base = 100
        
        for year in sorted(appreciations.keys()):
            if year > base_year:
                cumulative_index *= np.exp(appreciations[year])
                
            index_data.append({
                "year": year,
                "appreciation_rate": appreciations[year],
                "index_value": cumulative_index,
                "yoy_change": appreciations[year]
            })
        
        return self.spark.createDataFrame(index_data)
```

### 3.6 Main Pipeline Orchestrator

```python
# pipeline/main_pipeline.py
from pyspark.sql import SparkSession
from typing import Dict, List
import logging

class HPIPipeline:
    def __init__(self, spark_config: Dict[str, str]):
        self.spark = SparkSession.builder \
            .appName("HPI-FHFA-Pipeline") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "2000") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
            
        self.data_processor = DataProcessor(self.spark)
        self.supertract_algo = SupertractAlgorithm(self.spark)
        self.bmn_regression = BMNRegression(self.spark)
        self.aggregator = IndexAggregator(self.spark)
        
    def run_pipeline(
        self,
        transaction_path: str,
        geographic_path: str,
        weight_data_path: str,
        output_path: str,
        start_year: int = 1989,
        end_year: int = 2021
    ):
        """Execute full HPI pipeline"""
        
        # Load data with partitioning
        transactions = self.spark.read.parquet(transaction_path) \
            .repartition(200, "cbsa_code", "transaction_date")
            
        geographic_data = self.spark.read.parquet(geographic_path) \
            .cache()  # Small dataset, cache for broadcast
            
        weight_data = self.spark.read.parquet(weight_data_path) \
            .repartition(50, "year")
        
        # Create repeat-sales pairs
        repeat_sales = self.data_processor.create_repeat_sales_pairs(transactions)
        repeat_sales = self.data_processor.apply_filters(repeat_sales)
        
        # Calculate half-pairs
        half_pairs = self.data_processor.calculate_half_pairs(repeat_sales)
        
        # Process by year
        all_indices = {}
        
        for year in range(start_year, end_year + 1):
            logging.info(f"Processing year {year}")
            
            # Create supertracts
            supertracts = self.supertract_algo.create_supertracts(
                half_pairs, geographic_data, year
            )
            
            # Process each CBSA
            cbsa_list = supertracts.select("cbsa_code").distinct().collect()
            
            for cbsa_row in cbsa_list:
                cbsa = cbsa_row["cbsa_code"]
                
                # Get supertracts for CBSA
                cbsa_supertracts = supertracts.filter(
                    F.col("cbsa_code") == cbsa
                )
                
                # Run BMN for each supertract
                supertract_results = []
                
                for st_row in cbsa_supertracts.collect():
                    st_id = st_row["supertract_id"]
                    
                    # Prepare regression data
                    reg_data = self.bmn_regression.prepare_regression_data(
                        repeat_sales.filter(
                            F.col("census_tract").isin(st_row["tract_list"])
                        ),
                        st_id, year - 1, year
                    )
                    
                    # Estimate BMN
                    if reg_data.count() > 10:  # Minimum observations
                        bmn_results = self.bmn_regression.estimate_bmn(reg_data)
                        appreciation = bmn_results["coefficients"][-1] - \
                                     bmn_results["coefficients"][-2]
                        
                        supertract_results.append({
                            "supertract_id": st_id,
                            "appreciation_rate": appreciation
                        })
                
                # Create DataFrame of results
                st_indices = self.spark.createDataFrame(supertract_results)
                
                # Calculate all weight types
                for weight_type in self.aggregator.weight_types:
                    weights = self.aggregator.calculate_weights(
                        cbsa_supertracts, weight_data, weight_type, year
                    )
                    
                    city_appreciation = self.aggregator.aggregate_city_index(
                        st_indices, weights, cbsa, year
                    )
                    
                    key = f"{cbsa}_{weight_type}"
                    if key not in all_indices:
                        all_indices[key] = {}
                    all_indices[key][year] = city_appreciation
        
        # Construct final indices
        final_indices = []
        for key, appreciations in all_indices.items():
            cbsa, weight_type = key.rsplit("_", 1)
            index_series = self.aggregator.construct_index_series(appreciations)
            
            index_series = index_series.withColumn("cbsa_code", F.lit(cbsa)) \
                .withColumn("weight_type", F.lit(weight_type))
                
            final_indices.append(index_series)
        
        # Union all indices and save
        all_index_df = final_indices[0]
        for df in final_indices[1:]:
            all_index_df = all_index_df.union(df)
            
        all_index_df.write.partitionBy("cbsa_code", "year") \
            .mode("overwrite") \
            .parquet(output_path)
```

## 4. Testing Strategy

### 4.1 Test Architecture
```
tests/
├── unit/
│   ├── test_data_processor.py
│   ├── test_supertract.py
│   ├── test_bmn_regression.py
│   ├── test_index_aggregation.py
│   └── test_utils.py
├── integration/
│   ├── test_pipeline_e2e.py
│   ├── test_data_quality.py
│   └── test_performance.py
├── fixtures/
│   ├── sample_transactions.parquet
│   ├── sample_geographic.parquet
│   └── expected_outputs.parquet
└── conftest.py
```

### 4.2 Unit Testing Examples

```python
# tests/unit/test_data_processor.py
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
from datetime import datetime

class TestDataProcessor:
    @pytest.fixture(scope="session")
    def spark(self):
        return SparkSession.builder \
            .master("local[2]") \
            .appName("unit-tests") \
            .config("spark.sql.shuffle.partitions", "2") \
            .getOrCreate()
    
    @pytest.fixture
    def sample_transactions(self, spark):
        data = [
            ("P1", datetime(2010, 1, 1), 100000.0, "12345", "19100", 5.0),
            ("P1", datetime(2015, 6, 1), 150000.0, "12345", "19100", 5.0),
            ("P2", datetime(2012, 3, 1), 200000.0, "12346", "19100", 3.0),
            ("P2", datetime(2018, 9, 1), 180000.0, "12346", "19100", 3.0),
        ]
        schema = DataSchemas.TRANSACTION_SCHEMA
        return spark.createDataFrame(data, schema)
    
    def test_create_repeat_sales_pairs(self, spark, sample_transactions):
        processor = DataProcessor(spark)
        pairs = processor.create_repeat_sales_pairs(sample_transactions)
        
        # Test pair creation
        assert pairs.count() == 2
        
        # Test price relative calculation
        row1 = pairs.filter(pairs.property_id == "P1").first()
        expected_relative = np.log(150000) - np.log(100000)
        assert abs(row1.price_relative - expected_relative) < 0.0001
        
        # Test CAGR calculation
        expected_cagr = (150000/100000)**(1/5.42) - 1
        assert abs(row1.cagr - expected_cagr) < 0.01
    
    def test_apply_filters(self, spark, sample_transactions):
        processor = DataProcessor(spark)
        pairs = processor.create_repeat_sales_pairs(sample_transactions)
        
        # Add extreme CAGR case
        extreme_pair = spark.createDataFrame([
            ("P3", datetime(2010, 1, 1), 100000.0, datetime(2011, 1, 1), 
             200000.0, "12347", "19100", 5.0, np.log(2), 1.0, 1.0)
        ], pairs.schema)
        
        all_pairs = pairs.union(extreme_pair)
        filtered = processor.apply_filters(all_pairs)
        
        # Extreme CAGR should be filtered out
        assert filtered.count() == 2
        assert filtered.filter(filtered.property_id == "P3").count() == 0
    
    def test_calculate_half_pairs(self, spark):
        processor = DataProcessor(spark)
        
        # Create test data
        repeat_sales_data = [
            ("P1", datetime(2010, 1, 1), datetime(2015, 1, 1), 
             "12345", "19100"),
            ("P2", datetime(2010, 1, 1), datetime(2015, 1, 1), 
             "12345", "19100"),
            ("P3", datetime(2011, 1, 1), datetime(2016, 1, 1), 
             "12346", "19100"),
        ]
        
        repeat_sales = spark.createDataFrame(repeat_sales_data, 
            ["property_id", "sale_date_1", "sale_date_2", 
             "census_tract", "cbsa_code"])
        
        half_pairs = processor.calculate_half_pairs(repeat_sales)
        
        # Check aggregation
        tract_12345_2010 = half_pairs.filter(
            (half_pairs.census_tract == "12345") & 
            (half_pairs.year == 2010)
        ).first()
        
        assert tract_12345_2010.total_half_pairs == 2  # 2 properties sold
```

### 4.3 Integration Testing

```python
# tests/integration/test_pipeline_e2e.py
import pytest
from pyspark.sql import SparkSession
import tempfile
import shutil

class TestPipelineE2E:
    @pytest.fixture(scope="class")
    def spark(self):
        return SparkSession.builder \
            .master("local[4]") \
            .appName("integration-tests") \
            .config("spark.sql.shuffle.partitions", "10") \
            .getOrCreate()
    
    @pytest.fixture
    def test_data_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_pipeline(self, spark, test_data_dir):
        # Generate synthetic test data
        self._generate_test_data(spark, test_data_dir)
        
        # Run pipeline
        pipeline = HPIPipeline({"spark.app.name": "test"})
        pipeline.run_pipeline(
            f"{test_data_dir}/transactions",
            f"{test_data_dir}/geographic",
            f"{test_data_dir}/weights",
            f"{test_data_dir}/output",
            start_year=2015,
            end_year=2018
        )
        
        # Validate outputs
        output = spark.read.parquet(f"{test_data_dir}/output")
        
        # Check structure
        assert "year" in output.columns
        assert "index_value" in output.columns
        assert "cbsa_code" in output.columns
        assert "weight_type" in output.columns
        
        # Check data quality
        assert output.filter(output.index_value < 0).count() == 0
        assert output.filter(output.year == 2015).count() > 0
        
        # Check weight types
        weight_types = output.select("weight_type").distinct().collect()
        assert len(weight_types) == 6
    
    def test_data_quality_validation(self, spark, test_data_dir):
        from great_expectations.dataset import SparkDFDataset
        
        # Load output
        output = spark.read.parquet(f"{test_data_dir}/output")
        ge_df = SparkDFDataset(output)
        
        # Define expectations
        ge_df.expect_column_values_to_not_be_null("year")
        ge_df.expect_column_values_to_not_be_null("index_value")
        ge_df.expect_column_values_to_be_between("index_value", 50, 200)
        ge_df.expect_column_values_to_be_between("appreciation_rate", -0.5, 0.5)
        
        # Validate
        results = ge_df.validate()
        assert results["success"] == True
```

### 4.4 Performance Testing

```python
# tests/integration/test_performance.py
import time
import pytest
from pyspark.sql import SparkSession

class TestPerformance:
    @pytest.mark.performance
    def test_large_scale_processing(self, spark):
        # Generate large dataset
        num_properties = 1_000_000
        num_transactions = 3_000_000
        
        start_time = time.time()
        
        # Create synthetic data at scale
        transactions = self._generate_large_transaction_dataset(
            spark, num_properties, num_transactions
        )
        
        processor = DataProcessor(spark)
        repeat_sales = processor.create_repeat_sales_pairs(transactions)
        
        # Force computation
        count = repeat_sales.count()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance assertions
        assert processing_time < 300  # Should complete in 5 minutes
        assert count > 0
        
        # Log metrics
        print(f"Processed {count} repeat-sales pairs in {processing_time:.2f}s")
        print(f"Throughput: {count/processing_time:.0f} pairs/second")
    
    @pytest.mark.benchmark
    def test_bmn_regression_performance(self, spark):
        # Test regression performance with varying sizes
        sizes = [1000, 10000, 100000]
        
        for size in sizes:
            data = self._generate_regression_data(spark, size)
            
            start_time = time.time()
            bmn = BMNRegression(spark)
            results = bmn.estimate_bmn(data)
            end_time = time.time()
            
            print(f"BMN regression for {size} observations: {end_time-start_time:.2f}s")
            assert end_time - start_time < size / 1000  # Linear scaling
```

### 4.5 Test Coverage Configuration

```python
# pytest.ini
[pytest]
addopts = 
    --cov=hpi_fhfa
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
    -v
    --log-cli-level=INFO

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    benchmark: Benchmark tests
```

```python
# .coveragerc
[run]
source = hpi_fhfa
omit = 
    */tests/*
    */test_*.py
    setup.py
    */venv/*

[report]
precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov
```

## 5. Performance Optimization

### 5.1 Spark Configuration
```python
# config/spark_config.py
SPARK_CONFIG = {
    # Memory Management
    "spark.driver.memory": "8g",
    "spark.executor.memory": "16g",
    "spark.executor.memoryOverhead": "4g",
    
    # Parallelism
    "spark.default.parallelism": "400",
    "spark.sql.shuffle.partitions": "2000",
    
    # Adaptive Query Execution
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    
    # Broadcast Joins
    "spark.sql.autoBroadcastJoinThreshold": "100MB",
    
    # Serialization
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    "spark.kryo.registrationRequired": "false",
    
    # Delta Lake
    "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
    "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog"
}
```

### 5.2 Data Partitioning Strategy
```python
# Optimal partitioning for different stages
PARTITIONING_STRATEGY = {
    "transactions": {
        "partition_by": ["cbsa_code", "year"],
        "num_partitions": 2000
    },
    "repeat_sales": {
        "partition_by": ["cbsa_code", "sale_year"],
        "num_partitions": 1000
    },
    "indices": {
        "partition_by": ["cbsa_code", "weight_type"],
        "num_partitions": 500
    }
}
```

## 6. Deployment Architecture

### 6.1 Container Configuration
```dockerfile
# Dockerfile
FROM apache/spark:3.5.0-python3

# Install dependencies
RUN pip install \
    pyspark==3.5.0 \
    delta-spark==3.0.0 \
    numpy==1.24.3 \
    scipy==1.10.1 \
    pandas==2.0.3 \
    pytest==7.4.0 \
    pytest-spark==0.6.0 \
    pytest-cov==4.1.0 \
    great-expectations==0.17.23

# Copy application
COPY hpi_fhfa /app/hpi_fhfa
COPY tests /app/tests
COPY config /app/config

WORKDIR /app

# Set environment
ENV PYTHONPATH=/app
ENV SPARK_HOME=/opt/spark

# Entry point
ENTRYPOINT ["spark-submit"]
```

### 6.2 Orchestration with Airflow
```python
# dags/hpi_pipeline_dag.py
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=30)
}

dag = DAG(
    'hpi_fhfa_pipeline',
    default_args=default_args,
    description='FHFA House Price Index Pipeline',
    schedule_interval='@monthly',
    catchup=False
)

# Data validation task
validate_data = PythonOperator(
    task_id='validate_input_data',
    python_callable=validate_input_data,
    dag=dag
)

# Main pipeline
run_pipeline = SparkSubmitOperator(
    task_id='run_hpi_pipeline',
    application='/app/pipeline/main_pipeline.py',
    conn_id='spark_default',
    conf=SPARK_CONFIG,
    application_args=[
        '--transaction-path', '{{ var.value.transaction_path }}',
        '--geographic-path', '{{ var.value.geographic_path }}',
        '--weight-path', '{{ var.value.weight_path }}',
        '--output-path', '{{ var.value.output_path }}',
        '--start-year', '{{ ds | year }}',
        '--end-year', '{{ ds | year }}'
    ],
    dag=dag
)

# Quality checks
quality_checks = PythonOperator(
    task_id='run_quality_checks',
    python_callable=run_quality_validation,
    dag=dag
)

validate_data >> run_pipeline >> quality_checks
```

## 7. Monitoring and Logging

### 7.1 Metrics Collection
```python
# monitoring/metrics.py
from pyspark.sql import SparkSession
import time
from typing import Dict

class PipelineMetrics:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.metrics = {}
        
    def record_stage_metrics(self, stage_name: str, func):
        """Decorator to record stage execution metrics"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Get initial metrics
            initial_metrics = self._get_spark_metrics()
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Calculate metrics
            end_time = time.time()
            final_metrics = self._get_spark_metrics()
            
            self.metrics[stage_name] = {
                "duration_seconds": end_time - start_time,
                "records_processed": final_metrics["records"] - initial_metrics["records"],
                "shuffle_write_bytes": final_metrics["shuffle_write"] - initial_metrics["shuffle_write"],
                "shuffle_read_bytes": final_metrics["shuffle_read"] - initial_metrics["shuffle_read"]
            }
            
            return result
        return wrapper
    
    def _get_spark_metrics(self) -> Dict:
        """Extract current Spark metrics"""
        status = self.spark.sparkContext.statusTracker()
        return {
            "records": sum(stage.numTasks for stage in status.getActiveStageInfo()),
            "shuffle_write": sum(stage.shuffleWriteBytes for stage in status.getActiveStageInfo()),
            "shuffle_read": sum(stage.shuffleReadBytes for stage in status.getActiveStageInfo())
        }
```

## 8. Error Handling and Recovery

### 8.1 Checkpoint Strategy
```python
# utils/checkpoint.py
class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        
    def checkpoint_dataframe(self, df: DataFrame, name: str):
        """Save DataFrame checkpoint with versioning"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{self.checkpoint_dir}/{name}_{timestamp}"
        
        df.write.mode("overwrite").parquet(path)
        
        # Keep only last 3 checkpoints
        self._cleanup_old_checkpoints(name)
        
    def recover_checkpoint(self, spark: SparkSession, name: str) -> DataFrame:
        """Recover from latest checkpoint"""
        checkpoints = self._list_checkpoints(name)
        if checkpoints:
            latest = sorted(checkpoints)[-1]
            return spark.read.parquet(f"{self.checkpoint_dir}/{latest}")
        return None
```

## 9. Project Structure

```
hpi-fhfa-pyspark/
├── README.md
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── test.yml
│       └── deploy.yml
├── config/
│   ├── spark_config.py
│   └── pipeline_config.yaml
├── hpi_fhfa/
│   ├── __init__.py
│   ├── schemas/
│   │   └── data_schemas.py
│   ├── etl/
│   │   ├── __init__.py
│   │   ├── data_processor.py
│   │   └── data_validator.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── supertract.py
│   │   ├── bmn_regression.py
│   │   └── index_aggregation.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── main_pipeline.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── spark_utils.py
│   │   ├── checkpoint.py
│   │   └── logging_config.py
│   └── monitoring/
│       ├── __init__.py
│       └── metrics.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── validation_results.ipynb
└── scripts/
    ├── run_pipeline.sh
    └── validate_output.py
```

## 10. Development Timeline

### Phase 1: Foundation (2 weeks)
- Set up PySpark development environment
- Implement data schemas and models
- Create data ingestion pipeline
- Basic unit tests

### Phase 2: Core Algorithm (3 weeks)
- Implement repeat-sales pair generation
- Develop supertract algorithm
- Build BMN regression with MLlib
- Integration tests

### Phase 3: Aggregation & Indexing (2 weeks)
- Implement weight calculations
- Build index aggregation logic
- Create output generation
- Performance optimization

### Phase 4: Testing & Validation (2 weeks)
- Complete unit test suite (>80% coverage)
- Integration testing
- Performance benchmarking
- Data quality validation

### Phase 5: Deployment (1 week)
- Containerization
- CI/CD pipeline
- Documentation
- Production deployment

Total Timeline: 10 weeks

## 11. Success Metrics

1. **Functional Requirements**
   - All 6 index variants generated successfully
   - Balanced panel output for all census tracts
   - Accurate BMN regression implementation

2. **Performance Requirements**
   - Process 63.3M repeat-sales pairs in < 4 hours
   - Scale linearly with data volume
   - Memory usage < 128GB total cluster memory

3. **Quality Requirements**
   - Test coverage > 80% for core modules
   - All integration tests passing
   - Output validation against reference implementation

4. **Operational Requirements**
   - Automated monthly pipeline execution
   - Error recovery and checkpoint capability
   - Comprehensive monitoring and alerting