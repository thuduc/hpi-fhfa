"""
Shared test fixtures for RSAI model tests.

This module provides common test data and fixtures used across
unit and integration tests.
"""

import pytest
from datetime import date, datetime, timedelta
from pathlib import Path
import polars as pl
import numpy as np
from typing import List, Dict, Any

from rsai.src.data.models import (
    RSAIConfig,
    GeographyLevel,
    WeightingScheme,
    PropertyType,
    TransactionType
)


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return RSAIConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31),
        frequency="monthly",
        geography_levels=[GeographyLevel.TRACT, GeographyLevel.COUNTY],
        weighting_scheme=WeightingScheme.BMN,
        min_pairs_threshold=5,
        max_holding_period_years=10,
        min_price=50000,
        max_price=2000000,
        outlier_std_threshold=3.0,
        output_format="parquet",
        include_diagnostics=True
    )


@pytest.fixture
def sample_transactions_df():
    """Create sample transaction data."""
    np.random.seed(42)
    n_transactions = 1000
    n_properties = 200
    
    # Generate dates
    start_date = date(2020, 1, 1)
    end_date = date(2023, 12, 31)
    dates = []
    for i in range(n_transactions):
        days_offset = np.random.randint(0, (end_date - start_date).days)
        dates.append(start_date + timedelta(days=days_offset))
    
    # Generate prices with some growth
    base_prices = np.random.lognormal(12, 0.5, n_transactions)
    time_factors = [(d - start_date).days / 365.0 for d in dates]
    prices = base_prices * (1 + 0.05 * np.array(time_factors))  # 5% annual growth
    
    data = {
        "transaction_id": [f"T{i:06d}" for i in range(n_transactions)],
        "property_id": [f"P{np.random.randint(1, n_properties+1):06d}" for _ in range(n_transactions)],
        "sale_price": prices,
        "sale_date": dates,
        "transaction_type": np.random.choice(
            [TransactionType.ARMS_LENGTH.value, TransactionType.NON_ARMS_LENGTH.value],
            size=n_transactions,
            p=[0.9, 0.1]
        ),
        "recording_date": [d + timedelta(days=np.random.randint(1, 30)) for d in dates],
        "tract": [f"06037{np.random.randint(1000, 2000):04d}" for _ in range(n_transactions)],
        "county_fips": ["06037"] * n_transactions  # Los Angeles County
    }
    
    return pl.DataFrame(data)


@pytest.fixture
def sample_properties_df():
    """Create sample property data."""
    np.random.seed(42)
    n_properties = 200
    
    data = {
        "property_id": [f"P{i:06d}" for i in range(1, n_properties+1)],
        "latitude": np.random.uniform(33.5, 34.5, n_properties),
        "longitude": np.random.uniform(-118.8, -117.8, n_properties),
        "address": [f"{np.random.randint(100, 9999)} Main St" for _ in range(n_properties)],
        "zip_code": [f"900{np.random.randint(10, 99)}" for _ in range(n_properties)],
        "tract": [f"06037{np.random.randint(1000, 2000):04d}" for _ in range(n_properties)],
        "county_fips": ["06037"] * n_properties,
        "living_area": np.random.uniform(800, 5000, n_properties),
        "bedrooms": np.random.choice([1, 2, 3, 4, 5], n_properties),
        "bathrooms": np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0], n_properties),
        "year_built": np.random.randint(1950, 2020, n_properties),
        "property_type": np.random.choice(
            [PropertyType.SINGLE_FAMILY.value, PropertyType.CONDO.value, PropertyType.TOWNHOUSE.value],
            size=n_properties,
            p=[0.6, 0.3, 0.1]
        )
    }
    
    return pl.DataFrame(data)


@pytest.fixture
def sample_repeat_sales_df(sample_transactions_df):
    """Create sample repeat sales data."""
    # Get properties with multiple transactions
    property_counts = sample_transactions_df.group_by("property_id").agg(
        pl.count().alias("num_sales")
    ).filter(pl.col("num_sales") >= 2)
    
    repeat_properties = property_counts["property_id"].to_list()
    
    # Filter to repeat sale properties
    repeat_trans = sample_transactions_df.filter(
        pl.col("property_id").is_in(repeat_properties)
    ).sort(["property_id", "sale_date"])
    
    # Create pairs
    pairs = []
    for prop_id in repeat_properties:
        prop_trans = repeat_trans.filter(pl.col("property_id") == prop_id)
        trans_list = prop_trans.to_dicts()
        
        for i in range(len(trans_list) - 1):
            sale1 = trans_list[i]
            sale2 = trans_list[i + 1]
            
            holding_days = (sale2["sale_date"] - sale1["sale_date"]).days
            price_ratio = sale2["sale_price"] / sale1["sale_price"]
            
            # Skip if holding period is 0
            if holding_days == 0:
                continue
                
            pairs.append({
                "pair_id": f"{prop_id}_{sale1['transaction_id']}_{sale2['transaction_id']}",
                "property_id": prop_id,
                "sale1_transaction_id": sale1["transaction_id"],
                "sale1_price": sale1["sale_price"],
                "sale1_date": sale1["sale_date"],
                "sale2_transaction_id": sale2["transaction_id"],
                "sale2_price": sale2["sale_price"],
                "sale2_date": sale2["sale_date"],
                "price_ratio": price_ratio,
                "log_price_ratio": np.log(price_ratio),
                "holding_period_days": holding_days,
                "annualized_return": (price_ratio ** (365.0 / max(holding_days, 1))) - 1,
                "tract": sale1["tract"],
                "county_fips": sale1["county_fips"],
                "is_valid": True,
                "validation_flags": []
            })
    
    return pl.DataFrame(pairs)


@pytest.fixture
def sample_tract_stats_df(sample_transactions_df):
    """Create sample tract statistics."""
    return sample_transactions_df.group_by(["tract", "county_fips"]).agg([
        pl.count().alias("num_transactions"),
        pl.col("property_id").n_unique().alias("num_properties"),
        pl.col("sale_price").median().alias("median_price")
    ]).with_columns([
        (pl.col("tract").str.slice(-4).cast(pl.Float64) / 10000 * 2 + 33.5).alias("centroid_lat"),
        (pl.col("tract").str.slice(-4).cast(pl.Float64) / 10000 * 2 - 118.8).alias("centroid_lon")
    ])


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory structure."""
    output_dir = tmp_path / "rsai_output"
    output_dir.mkdir()
    (output_dir / "indices").mkdir()
    (output_dir / "reports").mkdir()
    (output_dir / "plots").mkdir()
    (output_dir / "data").mkdir()
    return output_dir


@pytest.fixture
def sample_index_values():
    """Create sample index values."""
    from rsai.src.data.models import IndexValue
    
    values = []
    base_date = date(2020, 1, 1)
    
    for i in range(48):  # 4 years of monthly data
        period = base_date + timedelta(days=30 * i)
        # Add some growth and volatility
        index = 100 * (1 + 0.05 * i / 12) * (1 + np.random.normal(0, 0.02))
        
        values.append(IndexValue(
            geography_level=GeographyLevel.COUNTY,
            geography_id="06037",
            period=period,
            index_value=index,
            num_pairs=np.random.randint(50, 200),
            num_properties=np.random.randint(40, 150),
            median_price=300000 * (1 + 0.05 * i / 12),
            standard_error=index * 0.01,
            confidence_lower=index * 0.98,
            confidence_upper=index * 1.02
        ))
    
    return values


@pytest.fixture
def sample_bmn_result(sample_index_values):
    """Create sample BMN regression result."""
    from rsai.src.data.models import BMNRegressionResult
    
    coefficients = {}
    standard_errors = {}
    t_statistics = {}
    p_values = {}
    
    for i, iv in enumerate(sample_index_values[1:]):  # Skip base period
        period_str = iv.period.strftime("%Y-%m-%d")
        coef = np.log(iv.index_value / 100)
        se = 0.01
        t_stat = coef / se
        p_val = 2 * (1 - np.abs(t_stat) / 10)  # Simplified p-value
        
        coefficients[period_str] = coef
        standard_errors[period_str] = se
        t_statistics[period_str] = t_stat
        p_values[period_str] = max(0, min(1, p_val))
    
    return BMNRegressionResult(
        geography_level=GeographyLevel.COUNTY,
        geography_id="06037",
        start_period=sample_index_values[0].period,
        end_period=sample_index_values[-1].period,
        num_periods=len(sample_index_values),
        num_observations=1000,
        r_squared=0.85,
        adj_r_squared=0.84,
        coefficients=coefficients,
        standard_errors=standard_errors,
        t_statistics=t_statistics,
        p_values=p_values,
        index_values=sample_index_values
    )


@pytest.fixture
def sample_quality_metrics():
    """Create sample quality metrics."""
    from rsai.src.data.models import QualityMetrics
    
    return QualityMetrics(
        total_records=1000,
        valid_records=950,
        invalid_records=50,
        missing_counts={"sale_price": 10, "sale_date": 5, "property_id": 0},
        invalid_counts={"sale_price": 20, "transaction_type": 15},
        completeness_score=0.95,
        validity_score=0.95,
        consistency_score=0.98,
        overall_score=0.96,
        issues=["Some prices below minimum threshold", "Missing geographic data for 5% of records"]
    )