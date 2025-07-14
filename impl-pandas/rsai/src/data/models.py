"""Data models and schema definitions for RSAI"""

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Transaction(BaseModel):
    """Individual property transaction record"""
    property_id: str = Field(..., description="Unique identifier for each housing unit")
    transaction_date: date = Field(..., description="Date of sale")
    transaction_price: float = Field(..., gt=0, description="Sale price in USD")
    census_tract_2010: str = Field(..., description="The 2010 Census Tract ID for the property")
    cbsa_id: str = Field(..., description="The Core-Based Statistical Area ID")
    
    @field_validator('transaction_price')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('Transaction price must be positive')
        return v


class RepeatSalePair(BaseModel):
    """Repeat sale pair with calculated metrics"""
    property_id: str
    first_sale_date: date
    first_sale_price: float
    second_sale_date: date
    second_sale_price: float
    census_tract_2010: str
    cbsa_id: str
    log_price_relative: float = Field(..., description="Log difference of prices")
    annual_growth_rate: float = Field(..., description="Compound annual growth rate")
    years_between_sales: float = Field(..., gt=0, description="Years between transactions")
    
    @field_validator('second_sale_date')
    def validate_sale_order(cls, v, info):
        if 'first_sale_date' in info.data and v <= info.data['first_sale_date']:
            raise ValueError('Second sale must be after first sale')
        return v


class GeographicData(BaseModel):
    """Census tract geographic information"""
    census_tract_2010: str = Field(..., description="Unique Census Tract ID")
    centroid_lat: float = Field(..., ge=-90, le=90, description="Latitude of tract centroid")
    centroid_lon: float = Field(..., ge=-180, le=180, description="Longitude of tract centroid")
    cbsa_id: str = Field(..., description="Core-Based Statistical Area ID")


class WeightingData(BaseModel):
    """Tract-level data for constructing aggregation weights"""
    census_tract_2010: str
    year: int
    total_housing_units: Optional[int] = Field(None, ge=0, description="Count of single-family units")
    total_housing_value: Optional[float] = Field(None, ge=0, description="Aggregate value of housing stock")
    total_upb: Optional[float] = Field(None, ge=0, description="Aggregate unpaid principal balance")
    college_population: Optional[int] = Field(None, ge=0, description="College-educated population (static)")
    non_white_population: Optional[int] = Field(None, ge=0, description="Non-white population (static)")


class SupertractDefinition(BaseModel):
    """Definition of a supertract for a specific year"""
    supertract_id: str
    year: int
    cbsa_id: str
    component_tracts: list[str] = Field(..., min_length=1, description="List of census tracts in this supertract")
    half_pairs_count: int = Field(..., ge=0, description="Number of half-pairs in this supertract")
    
    @field_validator('component_tracts')
    def validate_unique_tracts(cls, v):
        if len(v) != len(set(v)):
            raise ValueError('Component tracts must be unique')
        return v


class IndexValue(BaseModel):
    """House price index value for a specific time period"""
    cbsa_id: str
    year: int
    index_value: float = Field(..., gt=0, description="Index value (base year = 100)")
    appreciation_rate: float = Field(..., description="Annual appreciation rate")
    weighting_scheme: str = Field(..., description="Weighting scheme used")
    observations: int = Field(..., ge=0, description="Number of observations used")