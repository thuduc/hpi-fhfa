"""Script to generate sample data for testing the HPI pipeline"""

import argparse
import os
from datetime import date
import numpy as np
from pyspark.sql import SparkSession

from hpi_fhfa.schemas.data_schemas import DataSchemas


def generate_transactions(spark, num_properties=10000, output_path="data/transactions"):
    """Generate sample transaction data"""
    print(f"Generating {num_properties} properties with transactions...")
    
    data = []
    base_year = 2000
    
    for i in range(num_properties):
        property_id = f"P{i:06d}"
        census_tract = f"{11001 + (i % 100):05d}"  # 100 different tracts
        cbsa_code = f"{19100 + (i % 10):05d}"  # 10 different CBSAs
        base_price = 100000 + (i * 1000) % 900000
        distance_cbd = 1.0 + (i % 50)
        
        # First transaction
        year1 = base_year + (i % 10)
        data.append((
            property_id,
            date(year1, (i % 12) + 1, 1),
            float(base_price),
            census_tract,
            cbsa_code,
            float(distance_cbd)
        ))
        
        # Second transaction (3-8 years later)
        year2 = year1 + 3 + (i % 6)
        appreciation = 1.0 + np.random.uniform(0.01, 0.20)
        data.append((
            property_id,
            date(year2, (i % 12) + 1, 1),
            float(base_price * appreciation),
            census_tract,
            cbsa_code,
            float(distance_cbd)
        ))
        
        # Third transaction for 30% of properties
        if i % 3 == 0:
            year3 = year2 + 2 + (i % 4)
            appreciation2 = 1.0 + np.random.uniform(0.01, 0.15)
            data.append((
                property_id,
                date(year3, (i % 12) + 1, 1),
                float(base_price * appreciation * appreciation2),
                census_tract,
                cbsa_code,
                float(distance_cbd)
            ))
    
    df = spark.createDataFrame(data, DataSchemas.TRANSACTION_SCHEMA)
    df.write.mode("overwrite").parquet(output_path)
    print(f"Saved {df.count()} transactions to {output_path}")
    return df.count()


def generate_geographic(spark, num_tracts=100, output_path="data/geographic"):
    """Generate sample geographic data"""
    print(f"Generating geographic data for {num_tracts} census tracts...")
    
    data = []
    
    for i in range(num_tracts):
        census_tract = f"{11001 + i:05d}"
        cbsa_code = f"{19100 + (i % 10):05d}"
        
        # Generate coordinates in a grid pattern
        lat = 38.9 + (i // 10) * 0.01
        lon = -77.0 + (i % 10) * 0.01
        
        # Adjacent tracts
        adjacent = []
        if i > 0:
            adjacent.append(f"{11001 + i - 1:05d}")
        if i < num_tracts - 1:
            adjacent.append(f"{11001 + i + 1:05d}")
        if i >= 10:
            adjacent.append(f"{11001 + i - 10:05d}")
        if i < num_tracts - 10:
            adjacent.append(f"{11001 + i + 10:05d}")
        
        data.append((census_tract, cbsa_code, lat, lon, adjacent))
    
    df = spark.createDataFrame(data, DataSchemas.GEOGRAPHIC_SCHEMA)
    df.write.mode("overwrite").parquet(output_path)
    print(f"Saved geographic data to {output_path}")
    return df.count()


def generate_weights(spark, num_tracts=100, start_year=2000, end_year=2021, 
                    output_path="data/weights"):
    """Generate sample weight data"""
    print(f"Generating weight data for {num_tracts} tracts from {start_year} to {end_year}...")
    
    data = []
    
    for year in range(start_year, end_year + 1):
        for i in range(num_tracts):
            census_tract = f"{11001 + i:05d}"
            cbsa_code = f"{19100 + (i % 10):05d}"
            
            # Generate weight measures with some growth over time
            growth_factor = 1 + (year - start_year) * 0.02
            value_measure = 10000000.0 * (1 + i * 0.1) * growth_factor
            unit_measure = 1000.0 * (1 + i * 0.05)
            upb_measure = 8000000.0 * (1 + i * 0.08) * growth_factor
            
            # Static measures (only for 2010)
            if year == 2010:
                college_share = 0.25 + (i % 20) * 0.015
                nonwhite_share = 0.30 + (i % 20) * 0.02
            else:
                college_share = None
                nonwhite_share = None
            
            data.append((
                census_tract, cbsa_code, year,
                value_measure, unit_measure, upb_measure,
                college_share, nonwhite_share
            ))
    
    df = spark.createDataFrame(data, DataSchemas.WEIGHT_SCHEMA)
    df.write.mode("overwrite").parquet(output_path)
    print(f"Saved weight data to {output_path}")
    return df.count()


def main():
    parser = argparse.ArgumentParser(description="Generate sample data for HPI pipeline")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--num-properties", type=int, default=10000, 
                       help="Number of properties to generate")
    parser.add_argument("--num-tracts", type=int, default=100,
                       help="Number of census tracts")
    parser.add_argument("--start-year", type=int, default=2000,
                       help="Start year for data generation")
    parser.add_argument("--end-year", type=int, default=2021,
                       help="End year for data generation")
    
    args = parser.parse_args()
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("Generate Sample Data") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate data
    tx_count = generate_transactions(
        spark, 
        args.num_properties,
        os.path.join(args.output_dir, "transactions")
    )
    
    geo_count = generate_geographic(
        spark,
        args.num_tracts,
        os.path.join(args.output_dir, "geographic")
    )
    
    weight_count = generate_weights(
        spark,
        args.num_tracts,
        args.start_year,
        args.end_year,
        os.path.join(args.output_dir, "weights")
    )
    
    print("\nData generation complete!")
    print(f"Transactions: {tx_count:,}")
    print(f"Geographic records: {geo_count:,}")
    print(f"Weight records: {weight_count:,}")
    
    spark.stop()


if __name__ == "__main__":
    main()