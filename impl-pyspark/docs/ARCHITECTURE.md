# Architecture Documentation

## System Overview

The HPI-FHFA PySpark implementation is designed as a distributed data processing pipeline that leverages Apache Spark's capabilities for handling large-scale repeat-sales data.

## Architecture Principles

1. **Scalability**: Horizontal scaling through Spark's distributed computing
2. **Fault Tolerance**: Checkpoint mechanisms and retry logic
3. **Performance**: Optimized data partitioning and caching strategies
4. **Modularity**: Clear separation of concerns with distinct components
5. **Testability**: Comprehensive test coverage at all levels

## Component Architecture

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

## Key Components

### 1. Data Processing Layer (ETL)

**DataProcessor** (`etl/data_processor.py`)
- Creates repeat-sales pairs from transaction data
- Applies business logic filters (CAGR, appreciation bounds)
- Calculates half-pairs for tract aggregation

**DataValidator** (`etl/data_validator.py`)
- Validates input data quality
- Checks for schema compliance
- Identifies data anomalies

### 2. Core Algorithms

**SupertractAlgorithm** (`core/supertract.py`)
- Dynamic aggregation of census tracts
- Ensures minimum sample size (40 half-pairs)
- Uses geographic proximity for merging

**BMNRegression** (`core/bmn_regression.py`)
- Implements Bailey-Muth-Nourse regression
- Uses MLlib LinearRegression
- Handles sparse dummy variable matrices

**IndexAggregator** (`core/index_aggregation.py`)
- Aggregates tract-level indices to city level
- Implements 6 different weighting schemes
- Constructs cumulative index series

### 3. Pipeline Orchestration

**HPIPipeline** (`pipeline/main_pipeline.py`)
- Orchestrates the entire processing flow
- Manages Spark session configuration
- Handles error recovery and logging

## Data Flow

1. **Input Stage**
   - Transaction data partitioned by CBSA and date
   - Geographic data broadcast for efficient joins
   - Weight data partitioned by year

2. **Processing Stage**
   - Repeat-sales pairs created using self-join
   - Filters applied to ensure data quality
   - Half-pairs calculated for aggregation

3. **Aggregation Stage**
   - Supertracts created dynamically per year
   - BMN regression run for each supertract
   - Indices aggregated using various weights

4. **Output Stage**
   - Results partitioned by CBSA and year
   - Saved in Parquet format for efficiency
   - Metrics exported for monitoring

## Performance Optimizations

### Partitioning Strategy
```python
# Optimal partitioning for different stages
transactions: partition by ["cbsa_code", "year"] (2000 partitions)
repeat_sales: partition by ["cbsa_code", "sale_year"] (1000 partitions)
indices: partition by ["cbsa_code", "weight_type"] (500 partitions)
```

### Caching Strategy
- Geographic data cached (small, frequently accessed)
- Intermediate results checkpointed every 10 iterations
- Broadcast joins for dimension tables

### Spark Configuration
```python
{
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.sql.shuffle.partitions": "2000",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer"
}
```

## Scalability Considerations

1. **Horizontal Scaling**
   - Add more Spark executors for increased parallelism
   - Scales linearly with data volume

2. **Memory Management**
   - Executor memory: 16GB recommended
   - Driver memory: 8GB for coordination
   - Memory overhead: 4GB for off-heap storage

3. **Data Skew Handling**
   - Adaptive query execution for skewed joins
   - Salt key technique for heavily skewed CBSAs

## Fault Tolerance

1. **Checkpointing**
   - Intermediate results saved to reliable storage
   - Configurable checkpoint intervals

2. **Retry Logic**
   - Automatic retry for transient failures
   - Exponential backoff for external services

3. **Error Recovery**
   - Pipeline state persisted for recovery
   - Partial reprocessing from last checkpoint

## Security Considerations

1. **Data Encryption**
   - Encryption at rest for S3/HDFS storage
   - SSL/TLS for data in transit

2. **Access Control**
   - IAM roles for AWS resources
   - Kerberos authentication for Hadoop clusters

3. **Audit Logging**
   - All data access logged
   - Pipeline execution audit trail

## Monitoring and Observability

1. **Metrics Collection**
   - Processing time per stage
   - Record counts and data volume
   - Resource utilization

2. **Logging**
   - Structured JSON logging
   - Correlation IDs for request tracking
   - Log aggregation to centralized system

3. **Alerting**
   - SLA breach notifications
   - Data quality alerts
   - Resource utilization warnings