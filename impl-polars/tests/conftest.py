"""Pytest configuration and shared fixtures."""

import pytest
import polars as pl
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import tempfile
from typing import List

from src.hpi_fhfa.config.settings import HPIConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration."""
    # Create dummy files for config validation
    transaction_path = temp_dir / "transactions.parquet"
    geographic_path = temp_dir / "geographic.parquet"
    
    # Write empty parquet files
    pl.DataFrame().write_parquet(transaction_path)
    pl.DataFrame().write_parquet(geographic_path)
    
    return HPIConfig(
        transaction_data_path=transaction_path,
        geographic_data_path=geographic_path,
        output_path=temp_dir / "output",
        start_year=2015,
        end_year=2020,
        chunk_size=1000,
        n_jobs=2,
        validate_data=True,
        strict_validation=False
    )


@pytest.fixture
def sample_transactions():
    """Generate sample transaction data."""
    np.random.seed(42)
    n_properties = 1000
    n_transactions = 5000
    
    # Generate transactions with some properties having multiple sales
    property_ids = np.random.choice([f"P{i:06d}" for i in range(n_properties)], n_transactions)
    
    # Generate dates between 2010 and 2020
    start_date = date(2010, 1, 1)
    dates = [start_date + timedelta(days=int(d)) for d in np.random.randint(0, 3650, n_transactions)]
    
    # Generate prices with some growth
    base_prices = np.random.lognormal(12, 0.5, n_transactions)  # Mean ~$160k
    
    # Generate other fields
    tract_ids = np.random.choice([f"06037{i:06d}" for i in range(100)], n_transactions)
    cbsa_codes = np.random.choice(["31080", "41860", "33100"], n_transactions)  # LA, SF, Miami
    distances = np.random.uniform(0, 50, n_transactions)
    
    return pl.DataFrame({
        "property_id": property_ids,
        "transaction_date": dates,
        "transaction_price": base_prices,
        "census_tract": tract_ids,
        "cbsa_code": cbsa_codes,
        "distance_to_cbd": distances
    })


@pytest.fixture
def sample_repeat_sales(sample_transactions):
    """Generate sample repeat sales data."""
    # Sort by property_id and date
    df = sample_transactions.sort(["property_id", "transaction_date"])
    
    # Add previous transaction info
    df = df.with_columns([
        pl.col("transaction_date").shift(1).over("property_id").alias("prev_transaction_date"),
        pl.col("transaction_price").shift(1).over("property_id").alias("prev_transaction_price"),
    ])
    
    # Filter to only repeat sales
    df = df.filter(pl.col("prev_transaction_date").is_not_null())
    
    # Add calculated fields
    df = df.with_columns([
        (pl.col("transaction_price").log() - pl.col("prev_transaction_price").log()).alias("log_price_diff"),
        ((pl.col("transaction_date") - pl.col("prev_transaction_date")).dt.total_days() / 365.25).alias("time_diff_years"),
    ])
    
    # Calculate CAGR
    df = df.with_columns([
        (
            ((pl.col("transaction_price") / pl.col("prev_transaction_price"))
             .pow(1.0 / pl.col("time_diff_years")) - 1)
            .abs()
        ).alias("cagr")
    ])
    
    return df


@pytest.fixture
def sample_geographic_data():
    """Generate sample geographic/census tract data."""
    np.random.seed(42)
    n_tracts = 100
    
    tract_ids = [f"06037{i:06d}" for i in range(n_tracts)]
    cbsa_codes = np.random.choice(["31080", "41860", "33100"], n_tracts)
    
    # Generate centroid coordinates (roughly Southern California)
    lats = np.random.uniform(32.5, 34.5, n_tracts)
    lons = np.random.uniform(-118.5, -116.5, n_tracts)
    
    # Generate demographic data
    housing_units = np.random.randint(500, 5000, n_tracts)
    housing_values = housing_units * np.random.uniform(200000, 800000, n_tracts)
    college_shares = np.random.beta(2, 5, n_tracts)  # Skewed towards lower values
    nonwhite_shares = np.random.beta(3, 3, n_tracts)  # More uniform
    
    return pl.DataFrame({
        "tract_id": tract_ids,
        "cbsa_code": cbsa_codes,
        "centroid_lat": lats,
        "centroid_lon": lons,
        "housing_units": housing_units,
        "housing_value": housing_values,
        "college_share": college_shares,
        "nonwhite_share": nonwhite_shares
    })


@pytest.fixture 
def time_periods():
    """Generate list of time periods for testing."""
    return list(range(2015, 2021))  # 2015-2020