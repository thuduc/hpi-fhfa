# API Documentation

## Core Modules

### hpi_fhfa.etl.DataProcessor

Main class for ETL operations on transaction data.

```python
class DataProcessor:
    def __init__(self, spark: SparkSession)
```

#### Methods

##### create_repeat_sales_pairs
```python
def create_repeat_sales_pairs(self, transactions: DataFrame) -> DataFrame
```
Creates repeat-sales pairs from transaction data.

**Parameters:**
- `transactions`: DataFrame with transaction data

**Returns:**
- DataFrame with repeat-sales pairs

**Example:**
```python
processor = DataProcessor(spark)
repeat_sales = processor.create_repeat_sales_pairs(transactions)
```

##### apply_filters
```python
def apply_filters(self, repeat_sales: DataFrame) -> DataFrame
```
Applies data quality filters to repeat-sales pairs.

**Filters Applied:**
- Same-year filter: Removes transactions in the same year
- CAGR filter: |CAGR| ≤ 30%
- Appreciation filter: 0.25 ≤ price ratio ≤ 10.0

##### calculate_half_pairs
```python
def calculate_half_pairs(self, repeat_sales: DataFrame) -> DataFrame
```
Calculates half-pairs for each tract-year combination.

---

### hpi_fhfa.core.SupertractAlgorithm

Implements dynamic aggregation of census tracts.

```python
class SupertractAlgorithm:
    def __init__(self, spark: SparkSession, min_half_pairs: int = 40)
```

#### Methods

##### create_supertracts
```python
def create_supertracts(
    self, 
    half_pairs: DataFrame,
    geographic_data: DataFrame,
    year: int
) -> DataFrame
```
Creates supertracts by aggregating census tracts with insufficient data.

**Parameters:**
- `half_pairs`: DataFrame with half-pair counts
- `geographic_data`: DataFrame with geographic information
- `year`: Year to process

**Returns:**
- DataFrame with supertract mappings

**Algorithm:**
1. Identifies tracts with < min_half_pairs
2. Merges with nearest geographic neighbor
3. Continues until all supertracts meet threshold

---

### hpi_fhfa.core.BMNRegression

Implements Bailey-Muth-Nourse regression using MLlib.

```python
class BMNRegression:
    def __init__(self, spark: SparkSession)
```

#### Methods

##### prepare_regression_data
```python
def prepare_regression_data(
    self, 
    repeat_sales: DataFrame,
    supertract: str,
    start_year: int,
    end_year: int
) -> DataFrame
```
Prepares data for BMN regression with time dummy variables.

##### estimate_bmn
```python
def estimate_bmn(
    self, 
    regression_data: DataFrame,
    elastic_net_param: float = 0.0,
    reg_param: float = 0.0
) -> Dict[str, Any]
```
Estimates BMN regression coefficients.

**Returns:**
```python
{
    "coefficients": np.array,
    "std_errors": np.array,
    "r2": float,
    "rmse": float
}
```

##### calculate_appreciation_rates
```python
def calculate_appreciation_rates(
    self, 
    bmn_results: Dict[str, Any],
    periods: List[int]
) -> DataFrame
```
Calculates period-to-period appreciation rates from BMN results.

---

### hpi_fhfa.core.IndexAggregator

Aggregates tract-level indices to city level with various weighting schemes.

```python
class IndexAggregator:
    def __init__(self, spark: SparkSession)
```

#### Weight Types
- `sample`: Based on number of half-pairs
- `value`: Property value weights
- `unit`: Number of housing units
- `upb`: Unpaid principal balance
- `college`: College-educated population share
- `nonwhite`: Non-white population share

#### Methods

##### calculate_weights
```python
def calculate_weights(
    self,
    supertract_data: DataFrame,
    weight_data: DataFrame,
    weight_type: str,
    year: int
) -> DataFrame
```
Calculates normalized weights for aggregation.

##### aggregate_city_index
```python
def aggregate_city_index(
    self,
    supertract_indices: DataFrame,
    weights: DataFrame,
    cbsa: str,
    year: int
) -> float
```
Aggregates supertract indices to city level using specified weights.

##### construct_index_series
```python
def construct_index_series(
    self,
    appreciations: Dict[int, float],
    base_year: int = 1989
) -> DataFrame
```
Constructs cumulative index series from appreciation rates.

---

### hpi_fhfa.pipeline.HPIPipeline

Main pipeline orchestrator for the entire HPI calculation process.

```python
class HPIPipeline:
    def __init__(self, config_path: Optional[str] = None, spark_config: Optional[Dict] = None)
```

#### Methods

##### run_pipeline
```python
def run_pipeline(
    self,
    transaction_path: str,
    geographic_path: str,
    weight_data_path: str,
    output_path: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> Dict[str, Any]
```
Executes the full HPI pipeline.

**Parameters:**
- `transaction_path`: Path to transaction data
- `geographic_path`: Path to geographic data
- `weight_data_path`: Path to weight data
- `output_path`: Path for output indices
- `start_year`: Override config start year
- `end_year`: Override config end year

**Returns:**
```python
{
    "status": "SUCCESS" | "FAILED",
    "start_time": datetime,
    "end_time": datetime,
    "duration_seconds": float,
    "repeat_sales_count": int,
    "output_rows": int,
    "error": str  # Only if failed
}
```

---

## Data Schemas

### Transaction Schema
```python
TRANSACTION_SCHEMA = StructType([
    StructField("property_id", StringType(), False),
    StructField("transaction_date", DateType(), False),
    StructField("transaction_price", DoubleType(), False),
    StructField("census_tract", StringType(), False),
    StructField("cbsa_code", StringType(), False),
    StructField("distance_to_cbd", DoubleType(), False)
])
```

### Geographic Schema
```python
GEOGRAPHIC_SCHEMA = StructType([
    StructField("census_tract", StringType(), False),
    StructField("cbsa_code", StringType(), False),
    StructField("centroid_lat", DoubleType(), False),
    StructField("centroid_lon", DoubleType(), False),
    StructField("adjacent_tracts", ArrayType(StringType()), False)
])
```

### Weight Schema
```python
WEIGHT_SCHEMA = StructType([
    StructField("census_tract", StringType(), False),
    StructField("cbsa_code", StringType(), False),
    StructField("year", IntegerType(), False),
    StructField("value_measure", DoubleType(), False),
    StructField("unit_measure", DoubleType(), False),
    StructField("upb_measure", DoubleType(), False),
    StructField("college_share", DoubleType(), True),
    StructField("nonwhite_share", DoubleType(), True)
])
```

### Output Index Schema
```python
INDEX_SCHEMA = StructType([
    StructField("cbsa_code", StringType(), False),
    StructField("year", IntegerType(), False),
    StructField("weight_type", StringType(), False),
    StructField("appreciation_rate", DoubleType(), False),
    StructField("index_value", DoubleType(), False),
    StructField("yoy_change", DoubleType(), False)
])
```

---

## Utility Functions

### hpi_fhfa.utils.spark_utils

##### create_spark_session
```python
def create_spark_session(
    app_name: str,
    mode: str = "local",
    additional_config: Optional[Dict[str, str]] = None
) -> SparkSession
```
Creates optimized Spark session.

### hpi_fhfa.utils.geo_utils

##### haversine_distance
```python
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float
```
Calculates distance between two geographic points in kilometers.

##### find_adjacent_tracts
```python
def find_adjacent_tracts(
    tract_id: str,
    geographic_data: List[Dict[str, Any]],
    max_distance_km: float = 5.0
) -> List[str]
```
Finds geographically adjacent census tracts.