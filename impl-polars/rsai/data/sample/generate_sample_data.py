"""
Generate sample data for testing the RSAI model.

This script creates synthetic property transaction data with realistic patterns
for testing the RSAI pipeline.
"""

import polars as pl
import numpy as np
from datetime import date, timedelta
import random
from pathlib import Path


def generate_sample_data(
    num_properties: int = 10000,
    num_transactions: int = 50000,
    start_date: date = date(2015, 1, 1),
    end_date: date = date(2023, 12, 31),
    output_dir: Path = Path(".")
):
    """
    Generate sample property transaction and characteristics data.
    
    Args:
        num_properties: Number of unique properties
        num_transactions: Total number of transactions
        start_date: Start date for transactions
        end_date: End date for transactions
        output_dir: Directory to save files
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate properties
    print(f"Generating {num_properties} properties...")
    properties = generate_properties(num_properties)
    
    # Generate transactions
    print(f"Generating {num_transactions} transactions...")
    transactions = generate_transactions(
        properties, 
        num_transactions, 
        start_date, 
        end_date
    )
    
    # Save files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    properties_path = output_dir / "sample_properties.parquet"
    transactions_path = output_dir / "sample_transactions.parquet"
    
    properties.write_parquet(properties_path)
    transactions.write_parquet(transactions_path)
    
    print(f"Saved properties to {properties_path}")
    print(f"Saved transactions to {transactions_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Properties: {len(properties)}")
    print(f"Transactions: {len(transactions)}")
    print(f"Properties with multiple sales: {transactions['property_id'].n_unique()}")
    
    # Check repeat sales
    repeat_props = transactions.group_by("property_id").agg(
        pl.count().alias("num_sales")
    ).filter(pl.col("num_sales") >= 2)
    print(f"Properties with repeat sales: {len(repeat_props)}")
    
    return properties_path, transactions_path


def generate_properties(num_properties: int) -> pl.DataFrame:
    """Generate property characteristics data."""
    
    # Define geographic areas
    counties = ["06037", "06059", "06073", "06075", "06081"]  # LA, Orange, SD, SF, SM
    county_names = ["Los Angeles", "Orange", "San Diego", "San Francisco", "San Mateo"]
    msa_codes = ["31080", "31080", "41740", "41860", "41860"]
    
    # Generate base data
    property_ids = [f"PROP_{i:06d}" for i in range(num_properties)]
    
    # Assign counties with realistic distribution
    county_weights = [0.4, 0.2, 0.15, 0.15, 0.1]
    property_counties = np.random.choice(counties, size=num_properties, p=county_weights)
    
    # Generate tracts within counties (simplified)
    property_tracts = []
    for county in property_counties:
        tract_base = int(county) * 1000
        tract_num = np.random.randint(1, 200)
        property_tracts.append(f"{tract_base + tract_num:011d}")
    
    # Generate coordinates based on county
    latitudes = []
    longitudes = []
    county_centers = {
        "06037": (34.0522, -118.2437),  # LA
        "06059": (33.7175, -117.8311),  # Orange
        "06073": (32.7157, -117.1611),  # San Diego
        "06075": (37.7749, -122.4194),  # SF
        "06081": (37.5630, -122.3255)   # San Mateo
    }
    
    for county in property_counties:
        lat_center, lon_center = county_centers[county]
        # Add random variation
        lat = lat_center + np.random.normal(0, 0.1)
        lon = lon_center + np.random.normal(0, 0.1)
        latitudes.append(lat)
        longitudes.append(lon)
    
    # Generate property characteristics
    living_areas = np.random.lognormal(7.5, 0.4, num_properties)  # Log-normal around 1800 sqft
    living_areas = np.clip(living_areas, 500, 10000)
    
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=num_properties, p=[0.05, 0.25, 0.4, 0.25, 0.05])
    bathrooms = bedrooms + np.random.choice([-1, 0, 0, 1], size=num_properties, p=[0.1, 0.6, 0.2, 0.1])
    bathrooms = np.clip(bathrooms, 1, 5)
    
    # Year built with more recent bias
    current_year = 2023
    years_built = current_year - np.random.beta(2, 5, num_properties) * 100
    years_built = years_built.astype(int)
    
    # Property types
    property_types = np.random.choice(
        ["single_family", "condo", "townhouse"],
        size=num_properties,
        p=[0.6, 0.3, 0.1]
    )
    
    # Create DataFrame
    properties_df = pl.DataFrame({
        "property_id": property_ids,
        "latitude": latitudes,
        "longitude": longitudes,
        "address": [f"{np.random.randint(1, 9999)} Main St" for _ in range(num_properties)],
        "zip_code": [f"{90000 + np.random.randint(0, 999):05d}" for _ in range(num_properties)],
        "tract": property_tracts,
        "county_fips": property_counties,
        "msa_code": [msa_codes[counties.index(c)] for c in property_counties],
        "state": ["CA"] * num_properties,
        "living_area": living_areas.astype(int),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": years_built,
        "property_type": property_types
    })
    
    return properties_df


def generate_transactions(
    properties_df: pl.DataFrame,
    num_transactions: int,
    start_date: date,
    end_date: date
) -> pl.DataFrame:
    """Generate transaction data with realistic patterns."""
    
    # Calculate date range
    date_range = (end_date - start_date).days
    
    # Select properties for transactions (some properties sell multiple times)
    num_properties = len(properties_df)
    
    # Generate transaction distribution
    # Most properties sell 0-1 times, some sell 2-3 times, few sell more
    transactions_per_property = np.random.exponential(0.5, num_properties)
    transactions_per_property = np.round(transactions_per_property).astype(int)
    transactions_per_property = np.clip(transactions_per_property, 0, 10)
    
    # Adjust to match target number of transactions
    total_trans = transactions_per_property.sum()
    if total_trans > num_transactions:
        # Reduce transactions proportionally
        scale = num_transactions / total_trans
        transactions_per_property = (transactions_per_property * scale).astype(int)
    
    # Generate transactions
    transaction_records = []
    transaction_id = 1
    
    property_list = properties_df.to_dicts()
    
    for prop_idx, num_trans in enumerate(transactions_per_property):
        if num_trans == 0:
            continue
            
        property_info = property_list[prop_idx]
        property_id = property_info["property_id"]
        
        # Generate sale dates for this property
        if num_trans == 1:
            # Single sale - random date
            days_offset = np.random.randint(0, date_range)
            sale_dates = [start_date + timedelta(days=days_offset)]
        else:
            # Multiple sales - ensure minimum holding period
            min_holding_days = 365  # 1 year minimum
            
            # Generate sorted random dates with minimum spacing
            date_offsets = sorted(np.random.randint(0, date_range, num_trans))
            
            # Ensure minimum spacing
            for i in range(1, len(date_offsets)):
                if date_offsets[i] - date_offsets[i-1] < min_holding_days:
                    date_offsets[i] = date_offsets[i-1] + min_holding_days
                    
            sale_dates = [start_date + timedelta(days=int(offset)) for offset in date_offsets
                         if offset < date_range]
        
        # Generate prices for each sale
        # Base price depends on property characteristics
        base_price = (
            100000 +  # Base
            property_info["living_area"] * 200 +  # Price per sqft
            property_info["bedrooms"] * 20000 +
            property_info["bathrooms"] * 15000 +
            (2023 - property_info["year_built"]) * -1000  # Depreciation
        )
        
        # Add location premium
        if property_info["county_fips"] in ["06075", "06081"]:  # SF Bay Area
            base_price *= 2.5
        elif property_info["county_fips"] == "06059":  # Orange County
            base_price *= 1.5
            
        # Add market appreciation over time
        for i, sale_date in enumerate(sale_dates):
            # Annual appreciation rate (varies by period)
            years_from_start = (sale_date - start_date).days / 365.25
            
            # Simulate market cycles
            appreciation_rate = 0.05 + 0.03 * np.sin(years_from_start * 0.5)
            
            # Calculate price with appreciation and random variation
            price_multiplier = (1 + appreciation_rate) ** years_from_start
            sale_price = base_price * price_multiplier * np.random.normal(1, 0.05)
            sale_price = max(10000, int(sale_price))
            
            # Create transaction record
            transaction_records.append({
                "transaction_id": f"TRANS_{transaction_id:08d}",
                "property_id": property_id,
                "sale_price": sale_price,
                "sale_date": sale_date,
                "recording_date": sale_date + timedelta(days=np.random.randint(1, 30)),
                "transaction_type": np.random.choice(
                    ["arms_length", "non_arms_length"],
                    p=[0.95, 0.05]
                ),
                "buyer_name": f"Buyer_{transaction_id}",
                "seller_name": f"Seller_{transaction_id}",
                "mortgage_amount": int(sale_price * np.random.uniform(0.5, 0.9))
                    if np.random.random() > 0.2 else None
            })
            
            transaction_id += 1
    
    # Create DataFrame
    transactions_df = pl.DataFrame(transaction_records)
    
    # Join with property data to add geographic info
    transactions_df = transactions_df.join(
        properties_df.select(["property_id", "tract", "county_fips", "msa_code"]),
        on="property_id"
    )
    
    return transactions_df.sort("sale_date")


def create_sample_config(output_path: Path = Path("sample_config.yaml")):
    """Create a sample configuration file."""
    config_content = """# Sample RSAI Model Configuration

# Model parameters
model:
  name: "RSAI_Sample"
  version: "1.0.0"
  
  # Time parameters
  time:
    frequency: "monthly"
    
  # Geographic levels to process
  geography_levels:
    - "tract"
    - "county"
    - "msa"
    
  # Weighting scheme
  weighting_scheme: "case_shiller"
  
  # Minimum data requirements
  min_pairs_threshold: 10
  max_holding_period_years: 20

# Data parameters
data:
  # Input data format
  input_format: "parquet"
  
  # Column mappings
  columns:
    price: "sale_price"
    date: "sale_date"
    property_id: "property_id"
    
  # Data quality thresholds
  quality:
    min_price: 10000
    max_price: 10000000
    max_missing_ratio: 0.1

# Time range for analysis
start_date: "2018-01-01"
end_date: "2023-12-31"

# Preprocessing parameters
preprocessing:
  # Outlier detection
  outliers:
    method: "iqr"
    iqr_multiplier: 1.5
    zscore_threshold: 3.0

# Output parameters
output:
  # Results format
  format: "parquet"
  include_diagnostics: true
  
# Logging parameters
logging:
  level: "INFO"
"""
    
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    print(f"Created sample config at {output_path}")


if __name__ == "__main__":
    # Create sample data directory
    sample_dir = Path("rsai/data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample data
    print("Generating sample data for RSAI model testing...")
    properties_path, transactions_path = generate_sample_data(
        num_properties=10000,
        num_transactions=50000,
        output_dir=sample_dir
    )
    
    # Create sample config
    create_sample_config(sample_dir / "sample_config.yaml")
    
    print("\nSample data generation complete!")
    print(f"Properties: {properties_path}")
    print(f"Transactions: {transactions_path}")
    print(f"Config: {sample_dir / 'sample_config.yaml'}")
    
    print("\nTo run the RSAI pipeline with sample data:")
    print(f"python -m rsai.src.main {sample_dir}/sample_config.yaml {transactions_path} --properties {properties_path}")