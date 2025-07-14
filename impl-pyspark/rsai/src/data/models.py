"""
Data models and schema definitions for PySpark implementation.

This module defines the schemas and data models used throughout
the RSAI model implementation using PySpark types.
"""

from datetime import date
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    DateType, IntegerType, BooleanType, ArrayType,
    LongType
)


class PropertyType(str, Enum):
    """Property type enumeration."""
    SINGLE_FAMILY = "single_family"
    CONDO = "condo"
    TOWNHOUSE = "townhouse"
    MULTI_FAMILY = "multi_family"
    OTHER = "other"


class TransactionType(str, Enum):
    """Transaction type enumeration."""
    ARMS_LENGTH = "arms_length"
    NON_ARMS_LENGTH = "non_arms_length"
    FORECLOSURE = "foreclosure"
    SHORT_SALE = "short_sale"
    OTHER = "other"


class GeographyLevel(str, Enum):
    """Geographic aggregation levels."""
    TRACT = "tract"
    SUPERTRACT = "supertract"
    COUNTY = "county"
    CBSA = "cbsa"
    STATE = "state"
    NATIONAL = "national"


class WeightingScheme(str, Enum):
    """Weighting schemes for aggregation."""
    EQUAL = "equal"
    VALUE = "value"
    CASE_SHILLER = "case_shiller"
    BMN = "bmn"
    CUSTOM = "custom"


class ClusteringMethod(str, Enum):
    """Clustering methods for supertract generation."""
    KMEANS = "kmeans"
    BISECTING_KMEANS = "bisecting_kmeans"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    HIERARCHICAL = "hierarchical"


# PySpark Schemas for DataFrames
def get_transaction_schema() -> StructType:
    """Get PySpark schema for transaction data."""
    return StructType([
        StructField("transaction_id", StringType(), False),
        StructField("property_id", StringType(), False),
        StructField("sale_date", DateType(), False),
        StructField("sale_price", DoubleType(), False),
        StructField("transaction_type", StringType(), True),
    ])


def get_property_schema() -> StructType:
    """Get PySpark schema for property data."""
    return StructType([
        StructField("property_id", StringType(), False),
        StructField("property_type", StringType(), True),
        StructField("year_built", IntegerType(), True),
        StructField("square_feet", IntegerType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("tract", StringType(), True),
        StructField("county", StringType(), True),
        StructField("cbsa", StringType(), True),
        StructField("state", StringType(), True),
        StructField("address", StringType(), True),
    ])


def get_repeat_sales_schema() -> StructType:
    """Get PySpark schema for repeat sales pairs."""
    return StructType([
        StructField("pair_id", StringType(), False),
        StructField("property_id", StringType(), False),
        StructField("sale1_transaction_id", StringType(), False),
        StructField("sale1_date", DateType(), False),
        StructField("sale1_price", DoubleType(), False),
        StructField("sale2_transaction_id", StringType(), False),
        StructField("sale2_date", DateType(), False),
        StructField("sale2_price", DoubleType(), False),
        StructField("holding_period_days", IntegerType(), False),
        StructField("log_price_ratio", DoubleType(), False),
        StructField("annualized_return", DoubleType(), False),
        StructField("tract", StringType(), True),
        StructField("validation_flags", ArrayType(StringType()), True),
    ])


# Pydantic models for configuration and results
class RSAIConfig(BaseModel):
    """Configuration for RSAI model."""
    # Price filters
    min_price: float = Field(default=10000, gt=0)
    max_price: float = Field(default=10000000, gt=0)
    
    # Time filters
    max_holding_period_years: int = Field(default=20, ge=1)
    
    # Thresholds
    min_pairs_threshold: int = Field(default=10, ge=1)
    outlier_std_threshold: float = Field(default=3.0, gt=0)
    
    # Time frequency
    frequency: str = Field("monthly", pattern="^(daily|monthly|quarterly)$")
    base_period: Optional[date] = Field(default=None)
    
    # Weighting scheme
    weighting_scheme: WeightingScheme = Field(default=WeightingScheme.EQUAL)
    
    # Geographic levels
    geography_levels: List[GeographyLevel] = Field(
        default_factory=lambda: [GeographyLevel.TRACT, GeographyLevel.COUNTY]
    )
    
    # Clustering configuration
    clustering_method: str = Field(default="kmeans")
    n_clusters: int = Field(default=100, ge=1)
    min_cluster_size: int = Field(default=10, ge=1)
    max_cluster_size: int = Field(default=500, ge=1)
    
    # Spark configuration
    spark_app_name: str = Field(default="RSAI Model")
    spark_master: str = Field(default="local[*]")
    spark_executor_memory: str = Field(default="4g")
    spark_driver_memory: str = Field(default="2g")
    spark_config: Dict[str, str] = Field(default_factory=dict)
    
    # Input/Output configuration
    input_files: Dict[str, str] = Field(default_factory=dict)
    output_dir: str = Field(default="output")
    
    # Optional date range
    start_date: Optional[date] = Field(default=None)
    end_date: Optional[date] = Field(default=None)
    
    model_config = ConfigDict(use_enum_values=True)
    
    @field_validator('max_price')
    def validate_price_range(cls, v, info):
        """Validate that max_price is greater than min_price."""
        if 'min_price' in info.data and v <= info.data['min_price']:
            raise ValueError('max_price must be greater than min_price')
        return v


class TractInfo(BaseModel):
    """Information about a census tract."""
    tract_id: str = Field(..., description="Census tract FIPS code")
    county: str = Field(..., description="County FIPS code")
    num_properties: int = Field(..., ge=0)
    num_transactions: int = Field(..., ge=0)
    median_price: float = Field(..., gt=0)
    centroid_lat: Optional[float] = Field(None, ge=-90, le=90)
    centroid_lon: Optional[float] = Field(None, ge=-180, le=180)
    
    model_config = ConfigDict(use_enum_values=True)


class SupertractDefinition(BaseModel):
    """Definition of a supertract (cluster of census tracts)."""
    supertract_id: str = Field(..., description="Unique supertract identifier")
    name: str = Field(..., description="Supertract name")
    county: str = Field(..., description="County FIPS code")
    tract_ids: List[str] = Field(..., description="List of tract IDs in supertract")
    num_properties: int = Field(..., ge=0)
    num_transactions: int = Field(..., ge=0)
    median_price: float = Field(..., gt=0)
    
    # Bounding box
    min_lat: float = Field(..., ge=-90, le=90)
    max_lat: float = Field(..., ge=-90, le=90)
    min_lon: float = Field(..., ge=-180, le=180)
    max_lon: float = Field(..., ge=-180, le=180)
    
    model_config = ConfigDict(use_enum_values=True)


class RepeatSalePair(BaseModel):
    """Repeat sale pair information."""
    pair_id: str = Field(..., description="Unique pair identifier")
    property_id: str = Field(..., description="Property identifier")
    
    # First sale
    sale1_transaction_id: str
    sale1_price: float = Field(..., gt=0)
    sale1_date: date
    
    # Second sale
    sale2_transaction_id: str
    sale2_price: float = Field(..., gt=0)
    sale2_date: date
    
    # Calculated fields
    price_ratio: float = Field(..., gt=0)
    log_price_ratio: float
    holding_period_days: int = Field(..., gt=0)
    annualized_return: float
    
    # Geographic info
    tract: Optional[str] = None
    county: Optional[str] = None
    
    # Validation
    is_valid: bool = Field(default=True)
    validation_flags: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(use_enum_values=True)
    
    @field_validator('price_ratio')
    def validate_price_ratio(cls, v, info):
        """Validate price ratio calculation."""
        if 'sale1_price' in info.data and 'sale2_price' in info.data:
            expected_ratio = info.data['sale2_price'] / info.data['sale1_price']
            if abs(v - expected_ratio) > 0.001:
                raise ValueError("Price ratio calculation mismatch")
        return v


class IndexValue(BaseModel):
    """Index value at a specific time and geography."""
    geography_level: GeographyLevel = Field(..., description="Geographic level")
    geography_id: str = Field(..., description="Geographic identifier")
    period: date = Field(..., description="Index period")
    index_value: float = Field(..., gt=0, description="Index value (base=100)")
    
    # Statistics
    num_pairs: int = Field(..., ge=0, description="Number of repeat sales pairs")
    num_properties: int = Field(..., ge=0, description="Number of unique properties")
    median_price: Optional[float] = Field(None, gt=0, description="Median transaction price")
    
    # Confidence metrics
    standard_error: Optional[float] = Field(None, ge=0, description="Standard error")
    confidence_lower: Optional[float] = Field(None, gt=0, description="Lower confidence bound")
    confidence_upper: Optional[float] = Field(None, gt=0, description="Upper confidence bound")
    
    model_config = ConfigDict(use_enum_values=True)


class BMNRegressionResult(BaseModel):
    """Results from Bailey-Muth-Nourse (BMN) regression."""
    geography_level: GeographyLevel = Field(..., description="Geographic level")
    geography_id: str = Field(..., description="Geographic identifier")
    
    # Time periods
    start_period: date = Field(..., description="Start period")
    end_period: date = Field(..., description="End period")
    num_periods: int = Field(..., gt=0, description="Number of time periods")
    
    # Regression statistics
    num_observations: int = Field(..., gt=0, description="Number of observations")
    r_squared: float = Field(..., ge=0, le=1, description="R-squared")
    adj_r_squared: float = Field(..., description="Adjusted R-squared")
    
    # Coefficients (time dummies)
    coefficients: Dict[str, float] = Field(..., description="Period coefficients")
    standard_errors: Dict[str, float] = Field(..., description="Standard errors")
    t_statistics: Dict[str, float] = Field(..., description="T-statistics")
    p_values: Dict[str, float] = Field(..., description="P-values")
    
    # Index values
    index_values: List[IndexValue] = Field(..., description="Calculated index values")
    
    model_config = ConfigDict(use_enum_values=True)


class QualityMetrics(BaseModel):
    """Data quality metrics."""
    total_records: int = Field(..., ge=0)
    valid_records: int = Field(..., ge=0)
    invalid_records: int = Field(..., ge=0)
    
    # Field-level issues
    missing_fields: Dict[str, int] = Field(default_factory=dict)
    validation_errors: Dict[str, int] = Field(default_factory=dict)
    
    # Metrics
    completeness_score: float = Field(..., ge=0, le=1)
    validity_score: float = Field(..., ge=0, le=1)
    consistency_score: float = Field(default=1.0, ge=0, le=1)
    overall_score: float = Field(..., ge=0, le=1)
    
    # Issues
    issues: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(use_enum_values=True)