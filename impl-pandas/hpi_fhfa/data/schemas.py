"""Data validation schemas using Pandera for HPI-FHFA implementation."""

import pandas as pd
import pandera.pandas as pa
from typing import Optional
import re

from ..config import constants


# Transaction data schema
transaction_schema = pa.DataFrameSchema({
    "property_id": pa.Column(
        str, 
        nullable=False,
        unique=False,  # Properties can have multiple transactions
        description="Unique identifier for each property"
    ),
    "transaction_date": pa.Column(
        pd.Timestamp,
        nullable=False,
        checks=[
            pa.Check.in_range(
                pd.Timestamp(f"{constants.START_YEAR}-01-01"),
                pd.Timestamp(f"{constants.END_YEAR}-12-31")
            )
        ],
        description="Date of transaction"
    ),
    "transaction_price": pa.Column(
        float,
        nullable=False,
        checks=[
            pa.Check.greater_than(0),
            pa.Check.less_than(1e9)  # Reasonable upper bound
        ],
        description="Sale price of the property"
    ),
    "census_tract": pa.Column(
        str,
        nullable=False,
        checks=[
            pa.Check(lambda x: x.str.match(r'^\d{11}$').all(), 
                    error="Census tract must be 11 digits")
        ],
        description="2010 Census tract identifier"
    ),
    "cbsa_code": pa.Column(
        str,
        nullable=False,
        checks=[
            pa.Check(lambda x: x.str.match(r'^\d{5}$').all(),
                    error="CBSA code must be 5 digits")
        ],
        description="Core-Based Statistical Area code"
    ),
    "distance_to_cbd": pa.Column(
        float,
        nullable=False,
        checks=[
            pa.Check.greater_than_or_equal_to(0),
            pa.Check.less_than(1000)  # Reasonable max distance in miles
        ],
        description="Distance from central business district"
    )
})


# Census tract geographic data schema
census_tract_schema = pa.DataFrameSchema({
    "census_tract": pa.Column(
        str,
        nullable=False,
        unique=True,
        checks=[
            pa.Check(lambda x: x.str.match(r'^\d{11}$').all())
        ],
        description="2010 Census tract identifier"
    ),
    "cbsa_code": pa.Column(
        str,
        nullable=False,
        checks=[
            pa.Check(lambda x: x.str.match(r'^\d{5}$').all())
        ],
        description="Core-Based Statistical Area code"
    ),
    "centroid_lat": pa.Column(
        float,
        nullable=False,
        checks=[
            pa.Check.in_range(-90, 90)
        ],
        description="Latitude of tract centroid"
    ),
    "centroid_lon": pa.Column(
        float,
        nullable=False,
        checks=[
            pa.Check.in_range(-180, 180)
        ],
        description="Longitude of tract centroid"
    ),
    "housing_units": pa.Column(
        int,
        nullable=True,
        required=False,
        checks=[
            pa.Check.greater_than_or_equal_to(0)
        ],
        description="Number of housing units in tract"
    ),
    "aggregate_value": pa.Column(
        float,
        nullable=True,
        required=False,
        checks=[
            pa.Check.greater_than_or_equal_to(0)
        ],
        description="Aggregate housing value in tract"
    ),
    "college_share": pa.Column(
        float,
        nullable=True,
        required=False,
        checks=[
            pa.Check.in_range(0, 1)
        ],
        description="Share of college-educated population"
    ),
    "nonwhite_share": pa.Column(
        float,
        nullable=True,
        required=False,
        checks=[
            pa.Check.in_range(0, 1)
        ],
        description="Share of non-white population"
    )
})


# Repeat sales pair schema
repeat_sales_schema = pa.DataFrameSchema({
    "property_id": pa.Column(
        str,
        nullable=False,
        description="Property identifier"
    ),
    "sale1_date": pa.Column(
        pd.Timestamp,
        nullable=False,
        description="Date of first sale"
    ),
    "sale1_price": pa.Column(
        float,
        nullable=False,
        checks=[pa.Check.greater_than(0)],
        description="Price of first sale"
    ),
    "sale2_date": pa.Column(
        pd.Timestamp,
        nullable=False,
        checks=[
            pa.Check(lambda s: True, element_wise=False)  # Will validate sale2 > sale1 in function
        ],
        description="Date of second sale"
    ),
    "sale2_price": pa.Column(
        float,
        nullable=False,
        checks=[pa.Check.greater_than(0)],
        description="Price of second sale"
    ),
    "census_tract": pa.Column(
        str,
        nullable=False,
        description="Census tract of property"
    ),
    "cbsa_code": pa.Column(
        str,
        nullable=False,
        description="CBSA code"
    ),
    "distance_to_cbd": pa.Column(
        float,
        nullable=False,
        checks=[pa.Check.greater_than_or_equal_to(0)],
        description="Distance to CBD"
    ),
    "price_relative": pa.Column(
        float,
        nullable=False,
        description="Log price difference: log(P2/P1)"
    ),
    "time_diff_years": pa.Column(
        float,
        nullable=False,
        checks=[pa.Check.greater_than(0)],
        description="Time between sales in years"
    ),
    "cagr": pa.Column(
        float,
        nullable=False,
        description="Compound annual growth rate"
    )
})


def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate transaction data against schema.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data
        
    Returns
    -------
    pd.DataFrame
        Validated transaction data
        
    Raises
    ------
    pa.errors.SchemaError
        If validation fails
    """
    # Ensure datetime column
    if not pd.api.types.is_datetime64_any_dtype(df['transaction_date']):
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Validate against schema
    return transaction_schema.validate(df)


def validate_census_tracts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate census tract data against schema.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw census tract data
        
    Returns
    -------
    pd.DataFrame
        Validated census tract data
        
    Raises
    ------
    pa.errors.SchemaError
        If validation fails
    """
    return census_tract_schema.validate(df)


def validate_repeat_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate repeat sales pairs against schema.
    
    Parameters
    ----------
    df : pd.DataFrame
        Repeat sales pair data
        
    Returns
    -------
    pd.DataFrame
        Validated repeat sales data
        
    Raises
    ------
    pa.errors.SchemaError
        If validation fails
    """
    # Additional validation: sale2_date > sale1_date
    if not (df['sale2_date'] > df['sale1_date']).all():
        raise ValueError(
            "All sale2_date values must be greater than sale1_date"
        )
    
    return repeat_sales_schema.validate(df)


# Weight data schemas
weight_data_schema = pa.DataFrameSchema({
    "census_tract": pa.Column(str, nullable=False),
    "year": pa.Column(int, nullable=False),
    "half_pairs": pa.Column(int, nullable=True, coerce=True),
    "housing_value": pa.Column(float, nullable=True, coerce=True),
    "housing_units": pa.Column(int, nullable=True, coerce=True),
    "unpaid_balance": pa.Column(float, nullable=True, coerce=True)
})