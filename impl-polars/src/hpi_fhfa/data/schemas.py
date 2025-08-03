"""Polars schema definitions for HPI-FHFA data structures."""

import polars as pl
from typing import Dict, Any


# Transaction data schema
TRANSACTION_SCHEMA: Dict[str, Any] = {
    "property_id": pl.String,
    "transaction_date": pl.Date,
    "transaction_price": pl.Float64,
    "census_tract": pl.String,  # 2010 Census tract ID
    "cbsa_code": pl.String,
    "distance_to_cbd": pl.Float64,
}

# Repeat sales schema (derived from transactions)
REPEAT_SALES_SCHEMA: Dict[str, Any] = {
    **TRANSACTION_SCHEMA,
    "prev_transaction_date": pl.Date,
    "prev_transaction_price": pl.Float64,
    "log_price_diff": pl.Float64,  # p_itτ = log(price_t) - log(price_τ)
    "time_diff_years": pl.Float64,
    "cagr": pl.Float64,  # Compound annual growth rate
}

# Geographic data schema
GEOGRAPHIC_SCHEMA: Dict[str, Any] = {
    "tract_id": pl.String,
    "cbsa_code": pl.String,
    "centroid_lat": pl.Float64,
    "centroid_lon": pl.Float64,
    "housing_units": pl.Int64,
    "housing_value": pl.Float64,
    "college_share": pl.Float64,
    "nonwhite_share": pl.Float64,
}

# Supertract schema
SUPERTRACT_SCHEMA: Dict[str, Any] = {
    "supertract_id": pl.String,
    "period": pl.Int32,
    "component_tracts": pl.List(pl.String),
    "total_half_pairs": pl.Int32,
}

# Half-pairs schema
HALF_PAIRS_SCHEMA: Dict[str, Any] = {
    "tract_id": pl.String,
    "period": pl.Int32,
    "half_pairs_count": pl.Int32,
}

# Index output schema
INDEX_SCHEMA: Dict[str, Any] = {
    "geography_id": pl.String,  # tract_id or cbsa_code
    "period": pl.Int32,
    "index_value": pl.Float64,
    "appreciation_rate": pl.Float64,
    "n_observations": pl.Int32,
}


def validate_schema(df: pl.DataFrame, schema: Dict[str, Any]) -> None:
    """Validate that a DataFrame matches the expected schema.
    
    Args:
        df: DataFrame to validate
        schema: Expected schema dictionary
        
    Raises:
        ValueError: If schema doesn't match
    """
    # Check all required columns are present
    missing_cols = set(schema.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check data types match (with some flexibility for numeric types)
    for col, expected_type in schema.items():
        actual_type = df[col].dtype
        
        # Allow integer types for float columns (they can be cast)
        if expected_type == pl.Float64 and actual_type in [pl.Int32, pl.Int64]:
            continue
        
        if actual_type != expected_type:
            raise ValueError(
                f"Column '{col}' has type {actual_type}, expected {expected_type}"
            )


def cast_to_schema(df: pl.DataFrame, schema: Dict[str, Any]) -> pl.DataFrame:
    """Cast DataFrame columns to match schema types.
    
    Args:
        df: DataFrame to cast
        schema: Target schema dictionary
        
    Returns:
        DataFrame with columns cast to schema types
    """
    cast_exprs = []
    for col, dtype in schema.items():
        if col in df.columns:
            cast_exprs.append(pl.col(col).cast(dtype))
    
    return df.select(cast_exprs)