"""
Data models for the RSAI (Repeat Sales Price Index) model using Pydantic.

This module defines the core data structures used throughout the RSAI pipeline,
including property transactions, repeat sales pairs, and index calculations.
"""

from datetime import date, datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
import polars as pl


class PropertyType(str, Enum):
    """Enumeration of property types."""
    SINGLE_FAMILY = "single_family"
    CONDO = "condo"
    TOWNHOUSE = "townhouse"
    MULTI_FAMILY = "multi_family"
    OTHER = "other"


class TransactionType(str, Enum):
    """Enumeration of transaction types."""
    ARMS_LENGTH = "arms_length"
    NON_ARMS_LENGTH = "non_arms_length"
    FORECLOSURE = "foreclosure"
    SHORT_SALE = "short_sale"
    REO = "reo"
    OTHER = "other"


class GeographyLevel(str, Enum):
    """Enumeration of geographic aggregation levels."""
    PROPERTY = "property"
    BLOCK = "block"
    TRACT = "tract"
    SUPERTRACT = "supertract"
    ZIP = "zip"
    COUNTY = "county"
    MSA = "msa"
    STATE = "state"
    NATIONAL = "national"


class WeightingScheme(str, Enum):
    """Enumeration of weighting schemes for index calculation."""
    EQUAL = "equal"
    VALUE = "value"
    CASE_SHILLER = "case_shiller"
    BMN = "bmn"
    CUSTOM = "custom"


class PropertyCharacteristics(BaseModel):
    """Property physical characteristics."""
    living_area: Optional[float] = Field(None, gt=0, description="Living area in square feet")
    lot_size: Optional[float] = Field(None, gt=0, description="Lot size in square feet")
    bedrooms: Optional[int] = Field(None, ge=0, le=20, description="Number of bedrooms")
    bathrooms: Optional[float] = Field(None, ge=0, le=20, description="Number of bathrooms")
    age: Optional[int] = Field(None, ge=0, le=500, description="Property age in years")
    stories: Optional[int] = Field(None, ge=1, le=10, description="Number of stories")
    garage_spaces: Optional[int] = Field(None, ge=0, le=10, description="Number of garage spaces")
    pool: Optional[bool] = Field(None, description="Has swimming pool")
    property_type: PropertyType = Field(PropertyType.SINGLE_FAMILY, description="Property type")
    
    model_config = ConfigDict(use_enum_values=True)


class GeographicLocation(BaseModel):
    """Geographic location information."""
    property_id: str = Field(..., description="Unique property identifier")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude")
    address: Optional[str] = Field(None, description="Street address")
    zip_code: Optional[str] = Field(None, description="ZIP code")
    tract: Optional[str] = Field(None, description="Census tract")
    block: Optional[str] = Field(None, description="Census block")
    county_fips: Optional[str] = Field(None, description="County FIPS code")
    msa_code: Optional[str] = Field(None, description="MSA code")
    state: Optional[str] = Field(None, description="State abbreviation")
    
    @field_validator('zip_code')
    @classmethod
    def validate_zip(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) not in [5, 9, 10]:  # 5 digits, 9 digits, or ZIP+4 with dash
            raise ValueError("ZIP code must be 5 or 9 digits")
        return v


class Transaction(BaseModel):
    """Single property transaction record."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    property_id: str = Field(..., description="Property identifier")
    sale_price: float = Field(..., gt=0, description="Sale price")
    sale_date: date = Field(..., description="Sale date")
    transaction_type: TransactionType = Field(TransactionType.ARMS_LENGTH, description="Transaction type")
    
    # Optional fields
    recording_date: Optional[date] = Field(None, description="Recording date")
    buyer_name: Optional[str] = Field(None, description="Buyer name")
    seller_name: Optional[str] = Field(None, description="Seller name")
    mortgage_amount: Optional[float] = Field(None, ge=0, description="Mortgage amount")
    
    model_config = ConfigDict(use_enum_values=True)
    
    @model_validator(mode='after')
    def validate_dates(self) -> 'Transaction':
        """Ensure recording date is after sale date if provided."""
        if self.recording_date and self.recording_date < self.sale_date:
            raise ValueError("Recording date cannot be before sale date")
        return self


class RepeatSalePair(BaseModel):
    """Repeat sale pair for index calculation."""
    pair_id: str = Field(..., description="Unique pair identifier")
    property_id: str = Field(..., description="Property identifier")
    
    # First sale
    sale1_transaction_id: str = Field(..., description="First transaction ID")
    sale1_price: float = Field(..., gt=0, description="First sale price")
    sale1_date: date = Field(..., description="First sale date")
    
    # Second sale
    sale2_transaction_id: str = Field(..., description="Second transaction ID")
    sale2_price: float = Field(..., gt=0, description="Second sale price")
    sale2_date: date = Field(..., description="Second sale date")
    
    # Calculated fields
    price_ratio: float = Field(..., gt=0, description="Price ratio (sale2/sale1)")
    log_price_ratio: float = Field(..., description="Log price ratio")
    holding_period_days: int = Field(..., gt=0, description="Days between sales")
    annualized_return: float = Field(..., description="Annualized return")
    
    # Geographic info
    geography: Optional[GeographicLocation] = Field(None, description="Location information")
    
    # Quality flags
    is_valid: bool = Field(True, description="Validity flag")
    validation_flags: List[str] = Field(default_factory=list, description="Validation warning flags")
    
    @model_validator(mode='after')
    def validate_pair(self) -> 'RepeatSalePair':
        """Validate repeat sale pair consistency."""
        if self.sale2_date <= self.sale1_date:
            raise ValueError("Second sale must be after first sale")
        
        # Validate calculated fields
        expected_ratio = self.sale2_price / self.sale1_price
        if abs(self.price_ratio - expected_ratio) > 0.001:
            raise ValueError("Price ratio calculation mismatch")
        
        return self


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


class SupertractDefinition(BaseModel):
    """Definition of a supertract (aggregation of census tracts)."""
    supertract_id: str = Field(..., description="Unique supertract identifier")
    name: Optional[str] = Field(None, description="Supertract name")
    county_fips: str = Field(..., description="County FIPS code")
    
    # Component tracts
    tract_ids: List[str] = Field(..., min_length=1, description="List of census tract IDs")
    
    # Statistics
    num_properties: int = Field(..., ge=0, description="Number of properties")
    num_transactions: int = Field(..., ge=0, description="Number of transactions")
    median_price: Optional[float] = Field(None, gt=0, description="Median price")
    
    # Geographic bounds
    min_lat: Optional[float] = Field(None, ge=-90, le=90)
    max_lat: Optional[float] = Field(None, ge=-90, le=90)
    min_lon: Optional[float] = Field(None, ge=-180, le=180)
    max_lon: Optional[float] = Field(None, ge=-180, le=180)
    
    @field_validator('tract_ids')
    @classmethod
    def validate_unique_tracts(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError("Tract IDs must be unique")
        return v


class QualityMetrics(BaseModel):
    """Data quality metrics for validation."""
    total_records: int = Field(..., ge=0, description="Total number of records")
    valid_records: int = Field(..., ge=0, description="Number of valid records")
    invalid_records: int = Field(..., ge=0, description="Number of invalid records")
    
    # Field-level metrics
    missing_counts: Dict[str, int] = Field(default_factory=dict, description="Missing value counts by field")
    invalid_counts: Dict[str, int] = Field(default_factory=dict, description="Invalid value counts by field")
    
    # Data quality scores
    completeness_score: float = Field(..., ge=0, le=1, description="Data completeness score")
    validity_score: float = Field(..., ge=0, le=1, description="Data validity score")
    consistency_score: float = Field(..., ge=0, le=1, description="Data consistency score")
    overall_score: float = Field(..., ge=0, le=1, description="Overall quality score")
    
    # Issues
    issues: List[str] = Field(default_factory=list, description="List of quality issues")
    
    @model_validator(mode='after')
    def validate_counts(self) -> 'QualityMetrics':
        """Validate that counts sum correctly."""
        if self.valid_records + self.invalid_records != self.total_records:
            raise ValueError("Valid and invalid records must sum to total records")
        return self


class RSAIConfig(BaseModel):
    """Configuration for RSAI model execution."""
    # Time parameters
    start_date: date = Field(..., description="Analysis start date")
    end_date: date = Field(..., description="Analysis end date")
    frequency: str = Field("monthly", description="Index frequency")
    
    # Geographic parameters
    geography_levels: List[GeographyLevel] = Field(
        default=[GeographyLevel.TRACT, GeographyLevel.COUNTY, GeographyLevel.MSA],
        description="Geographic levels to calculate"
    )
    
    # Model parameters
    weighting_scheme: WeightingScheme = Field(
        WeightingScheme.BMN,
        description="Weighting scheme"
    )
    min_pairs_threshold: int = Field(10, ge=1, description="Minimum repeat pairs threshold")
    max_holding_period_years: int = Field(20, ge=1, description="Maximum holding period")
    
    # Quality filters
    min_price: float = Field(10000, gt=0, description="Minimum valid price")
    max_price: float = Field(10000000, gt=0, description="Maximum valid price")
    outlier_std_threshold: float = Field(3.0, gt=0, description="Outlier threshold in std deviations")
    
    # Output parameters
    output_format: str = Field("parquet", description="Output format")
    include_diagnostics: bool = Field(True, description="Include diagnostic outputs")
    
    model_config = ConfigDict(use_enum_values=True)