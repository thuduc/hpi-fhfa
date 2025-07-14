"""
Generate test data fixtures for RSAI model tests.

This script creates realistic test data files that can be used
for testing the RSAI pipeline.
"""

import polars as pl
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import argparse

from rsai.src.data.models import PropertyType, TransactionType


def generate_properties(n_properties: int, counties: list, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic property data."""
    np.random.seed(seed)
    
    properties = []
    
    for i in range(n_properties):
        # Assign to counties based on proportion
        county_idx = np.random.choice(len(counties))
        county = counties[county_idx]
        
        # Generate location based on county
        if county == "06037":  # Los Angeles
            lat_base, lon_base = 34.05, -118.25
        elif county == "06059":  # Orange
            lat_base, lon_base = 33.70, -117.80
        elif county == "06073":  # San Diego
            lat_base, lon_base = 32.72, -117.16
        else:
            lat_base, lon_base = 34.0, -118.0
        
        properties.append({
            "property_id": f"P{i+1:06d}",
            "latitude": lat_base + np.random.uniform(-0.5, 0.5),
            "longitude": lon_base + np.random.uniform(-0.5, 0.5),
            "address": f"{np.random.randint(100, 9999)} {np.random.choice(['Main', 'Oak', 'Elm', 'Park'])} St",
            "zip_code": f"{np.random.randint(90000, 93000)}",
            "tract": f"{county}{np.random.randint(100000, 999999):06d}",
            "block": f"{np.random.randint(1000, 9999)}",
            "county_fips": county,
            "msa_code": "31080" if county == "06037" else "31100",  # LA or SD MSA
            "state": "CA",
            "living_area": np.random.lognormal(7.5, 0.3),  # Log-normal around 1800 sqft
            "lot_size": np.random.lognormal(8.5, 0.5),  # Log-normal around 5000 sqft
            "bedrooms": np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.4, 0.3, 0.1]),
            "bathrooms": np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 
                                        p=[0.1, 0.1, 0.3, 0.2, 0.2, 0.05, 0.05]),
            "year_built": np.random.randint(1920, 2020),
            "stories": np.random.choice([1, 2, 3], p=[0.6, 0.35, 0.05]),
            "garage_spaces": np.random.choice([0, 1, 2, 3], p=[0.1, 0.2, 0.5, 0.2]),
            "pool": np.random.choice([True, False], p=[0.15, 0.85]),
            "property_type": np.random.choice(
                [PropertyType.SINGLE_FAMILY.value, PropertyType.CONDO.value, 
                 PropertyType.TOWNHOUSE.value, PropertyType.MULTI_FAMILY.value],
                p=[0.6, 0.25, 0.1, 0.05]
            )
        })
    
    return pl.DataFrame(properties)


def generate_transactions(properties_df: pl.DataFrame, 
                         start_date: date,
                         end_date: date,
                         avg_sales_per_property: float = 2.5,
                         annual_appreciation: float = 0.05,
                         seed: int = 42) -> pl.DataFrame:
    """Generate synthetic transaction data."""
    np.random.seed(seed)
    
    transactions = []
    transaction_id = 1
    
    for prop in properties_df.iter_rows(named=True):
        # Number of sales for this property (Poisson distribution)
        n_sales = max(1, np.random.poisson(avg_sales_per_property - 1) + 1)
        
        # Base price based on property characteristics
        base_price = (
            50000 +  # Base
            prop["living_area"] * 200 +  # Price per sqft
            prop["bedrooms"] * 10000 +
            prop["bathrooms"] * 15000 +
            (20000 if prop["pool"] else 0) +
            prop["garage_spaces"] * 5000 -
            max(0, 2024 - prop["year_built"]) * 500  # Depreciation
        )
        
        # Add location premium
        if prop["county_fips"] == "06037":  # LA premium
            base_price *= 1.2
        elif prop["county_fips"] == "06059":  # Orange County premium
            base_price *= 1.15
        
        # Add randomness
        base_price *= np.random.uniform(0.8, 1.2)
        
        # Generate sale dates
        days_range = (end_date - start_date).days
        sale_dates = sorted([
            start_date + timedelta(days=np.random.randint(0, days_range))
            for _ in range(n_sales)
        ])
        
        # Ensure minimum spacing between sales
        for i in range(1, len(sale_dates)):
            if (sale_dates[i] - sale_dates[i-1]).days < 180:
                sale_dates[i] = sale_dates[i-1] + timedelta(days=np.random.randint(180, 365))
                if sale_dates[i] > end_date:
                    sale_dates = sale_dates[:i]
                    break
        
        # Generate transactions
        for i, sale_date in enumerate(sale_dates):
            # Calculate appreciation
            years_passed = (sale_date - start_date).days / 365.25
            market_factor = (1 + annual_appreciation) ** years_passed
            
            # Add some volatility
            volatility = np.random.normal(1.0, 0.05)
            
            # Final price
            sale_price = base_price * market_factor * volatility
            
            # Transaction type (most are arms-length)
            if np.random.random() < 0.03:  # 3% foreclosures
                trans_type = TransactionType.FORECLOSURE.value
                sale_price *= 0.8  # Discount
            elif np.random.random() < 0.02:  # 2% short sales
                trans_type = TransactionType.SHORT_SALE.value
                sale_price *= 0.9  # Discount
            else:
                trans_type = TransactionType.ARMS_LENGTH.value
            
            transactions.append({
                "transaction_id": f"T{transaction_id:08d}",
                "property_id": prop["property_id"],
                "sale_price": max(10000, int(sale_price)),  # Minimum $10k
                "sale_date": sale_date,
                "recording_date": sale_date + timedelta(days=np.random.randint(1, 30)),
                "transaction_type": trans_type,
                "buyer_name": f"Buyer {np.random.randint(1000, 9999)}",
                "seller_name": f"Seller {np.random.randint(1000, 9999)}",
                "mortgage_amount": int(sale_price * np.random.uniform(0.7, 0.9)) if np.random.random() > 0.2 else None
            })
            
            transaction_id += 1
    
    return pl.DataFrame(transactions)


def main():
    """Generate test data files."""
    parser = argparse.ArgumentParser(description="Generate test data for RSAI model")
    parser.add_argument("--output-dir", type=str, default="tests/fixtures/data",
                       help="Output directory for test data files")
    parser.add_argument("--n-properties", type=int, default=1000,
                       help="Number of properties to generate")
    parser.add_argument("--counties", nargs="+", default=["06037", "06059"],
                       help="County FIPS codes to include")
    parser.add_argument("--start-date", type=str, default="2015-01-01",
                       help="Start date for transactions (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2023-12-31",
                       help="End date for transactions (YYYY-MM-DD)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse dates
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    
    print(f"Generating test data with {args.n_properties} properties...")
    
    # Generate properties
    properties_df = generate_properties(
        args.n_properties,
        args.counties,
        seed=args.seed
    )
    
    # Add geographic columns for join
    properties_df = properties_df.with_columns([
        pl.col("tract").alias("census_tract"),
        pl.col("block").alias("census_block")
    ])
    
    print(f"Generated {len(properties_df)} properties")
    
    # Generate transactions
    transactions_df = generate_transactions(
        properties_df,
        start_date,
        end_date,
        avg_sales_per_property=2.5,
        annual_appreciation=0.05,
        seed=args.seed
    )
    
    # Add geographic info to transactions
    transactions_df = transactions_df.join(
        properties_df.select(["property_id", "tract", "county_fips"]),
        on="property_id",
        how="left"
    )
    
    print(f"Generated {len(transactions_df)} transactions")
    
    # Calculate repeat sales statistics
    repeat_props = (
        transactions_df.group_by("property_id")
        .agg(pl.count().alias("n_sales"))
        .filter(pl.col("n_sales") >= 2)
    )
    print(f"Properties with repeat sales: {len(repeat_props)}")
    
    # Save files
    properties_path = output_dir / "properties.parquet"
    properties_df.write_parquet(properties_path)
    print(f"Saved properties to {properties_path}")
    
    transactions_path = output_dir / "transactions.parquet"
    transactions_df.write_parquet(transactions_path)
    print(f"Saved transactions to {transactions_path}")
    
    # Also save as CSV for inspection
    properties_df.write_csv(output_dir / "properties.csv")
    transactions_df.write_csv(output_dir / "transactions.csv")
    
    # Generate summary statistics
    print("\nSummary Statistics:")
    print(f"- Total properties: {len(properties_df)}")
    print(f"- Total transactions: {len(transactions_df)}")
    print(f"- Date range: {transactions_df['sale_date'].min()} to {transactions_df['sale_date'].max()}")
    print(f"- Price range: ${transactions_df['sale_price'].min():,.0f} to ${transactions_df['sale_price'].max():,.0f}")
    print(f"- Median price: ${transactions_df['sale_price'].median():,.0f}")
    
    # County breakdown
    county_stats = transactions_df.group_by("county_fips").agg([
        pl.count().alias("transactions"),
        pl.col("property_id").n_unique().alias("properties"),
        pl.col("sale_price").median().alias("median_price")
    ])
    print("\nCounty Statistics:")
    print(county_stats)


if __name__ == "__main__":
    main()