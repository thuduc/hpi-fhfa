"""Generate sample data for RSAI testing and demonstration"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)


def generate_transaction_data(n_properties=1000, n_cbsas=3, years_range=(2015, 2021)):
    """Generate realistic transaction data with repeat sales"""
    
    # Define CBSAs and their census tracts
    cbsa_tracts = {
        '31080': [f'06037{i:06d}' for i in range(100000, 100020)],  # Los Angeles
        '41860': [f'06075{i:06d}' for i in range(200000, 200015)],  # San Francisco
        '47900': [f'08031{i:06d}' for i in range(300000, 300010)]   # Denver
    }
    
    transactions = []
    property_counter = 0
    
    for cbsa_id, tracts in cbsa_tracts.items():
        n_props_cbsa = n_properties // n_cbsas
        
        for _ in range(n_props_cbsa):
            property_id = f'PROP{property_counter:06d}'
            property_counter += 1
            
            # Assign to random tract
            tract = np.random.choice(tracts)
            
            # Generate 1-4 transactions for this property
            n_transactions = np.random.choice([1, 2, 3, 4], p=[0.3, 0.5, 0.15, 0.05])
            
            # First transaction
            first_year = np.random.randint(years_range[0], years_range[1] - 1)
            first_month = np.random.randint(1, 13)
            first_date = datetime(first_year, first_month, 
                                np.random.randint(1, 28))
            
            # Base price varies by CBSA
            base_price_mult = {'31080': 1.0, '41860': 1.5, '47900': 0.8}[cbsa_id]
            base_price = np.random.uniform(200000, 800000) * base_price_mult
            
            transactions.append({
                'property_id': property_id,
                'transaction_date': first_date,
                'transaction_price': base_price,
                'census_tract_2010': tract,
                'cbsa_id': cbsa_id
            })
            
            # Subsequent transactions
            prev_date = first_date
            prev_price = base_price
            
            for i in range(1, n_transactions):
                # Time between sales: 1-5 years
                years_gap = np.random.uniform(1, 5)
                next_date = prev_date + timedelta(days=int(years_gap * 365))
                
                # Skip if beyond date range
                if next_date.year > years_range[1]:
                    break
                
                # Price appreciation: 2-8% per year with some noise
                annual_appr = np.random.normal(0.05, 0.02)
                # Clip to reasonable range
                annual_appr = np.clip(annual_appr, -0.1, 0.15)
                
                price_mult = (1 + annual_appr) ** years_gap
                next_price = prev_price * price_mult
                
                transactions.append({
                    'property_id': property_id,
                    'transaction_date': next_date,
                    'transaction_price': next_price,
                    'census_tract_2010': tract,
                    'cbsa_id': cbsa_id
                })
                
                prev_date = next_date
                prev_price = next_price
    
    return pd.DataFrame(transactions)


def generate_geographic_data(cbsa_tracts):
    """Generate geographic data for census tracts"""
    
    geographic_data = []
    
    # Define approximate centroids for each CBSA
    cbsa_centers = {
        '31080': (34.0522, -118.2437),  # Los Angeles
        '41860': (37.7749, -122.4194),  # San Francisco
        '47900': (39.7392, -104.9903)   # Denver
    }
    
    for cbsa_id, tracts in cbsa_tracts.items():
        center_lat, center_lon = cbsa_centers[cbsa_id]
        
        for i, tract in enumerate(tracts):
            # Create grid around center
            row = i // 5
            col = i % 5
            
            lat = center_lat + (row - 2) * 0.02  # ~2km spacing
            lon = center_lon + (col - 2) * 0.02
            
            geographic_data.append({
                'census_tract_2010': tract,
                'centroid_lat': lat,
                'centroid_lon': lon,
                'cbsa_id': cbsa_id
            })
    
    return pd.DataFrame(geographic_data)


def generate_weighting_data(cbsa_tracts, years_range=(2015, 2021)):
    """Generate weighting data for census tracts"""
    
    weighting_data = []
    
    # Base values vary by CBSA
    cbsa_multipliers = {
        '31080': {'units': 1.0, 'value': 1.0, 'upb': 1.0},
        '41860': {'units': 0.8, 'value': 1.5, 'upb': 1.3},
        '47900': {'units': 1.2, 'value': 0.8, 'upb': 0.9}
    }
    
    for cbsa_id, tracts in cbsa_tracts.items():
        mult = cbsa_multipliers[cbsa_id]
        
        for i, tract in enumerate(tracts):
            # Generate base values with some variation
            base_units = np.random.randint(800, 1500) * mult['units']
            base_value = base_units * np.random.uniform(400000, 600000) * mult['value']
            base_upb = base_value * np.random.uniform(0.7, 0.85) * mult['upb']
            
            # Demographics (static for 2010)
            college_pop = np.random.randint(1500, 3500)
            non_white_pop = np.random.randint(1000, 4000)
            
            # Add 2010 demographic data
            weighting_data.append({
                'census_tract_2010': tract,
                'year': 2010,
                'total_housing_units': None,
                'total_housing_value': None,
                'total_upb': None,
                'college_population': college_pop,
                'non_white_population': non_white_pop
            })
            
            # Add yearly data
            for year in range(years_range[0], years_range[1] + 1):
                # Apply growth over time
                years_from_base = year - years_range[0]
                growth_factor = 1 + 0.02 * years_from_base  # 2% annual growth
                
                weighting_data.append({
                    'census_tract_2010': tract,
                    'year': year,
                    'total_housing_units': int(base_units * growth_factor),
                    'total_housing_value': base_value * growth_factor * (1 + 0.03 * years_from_base),
                    'total_upb': base_upb * growth_factor,
                    'college_population': None,  # Only in 2010
                    'non_white_population': None  # Only in 2010
                })
    
    return pd.DataFrame(weighting_data)


def save_sample_data(output_dir='rsai/data/sample'):
    """Generate and save all sample data files"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating sample data...")
    
    # Define CBSAs and tracts
    cbsa_tracts = {
        '31080': [f'06037{i:06d}' for i in range(100000, 100020)],
        '41860': [f'06075{i:06d}' for i in range(200000, 200015)],
        '47900': [f'08031{i:06d}' for i in range(300000, 300010)]
    }
    
    # Generate transaction data
    print("  - Generating transaction data...")
    transactions_df = generate_transaction_data(n_properties=2000, n_cbsas=3)
    transactions_file = output_path / 'transactions.csv'
    transactions_df.to_csv(transactions_file, index=False)
    print(f"    Saved {len(transactions_df)} transactions to {transactions_file}")
    
    # Generate geographic data
    print("  - Generating geographic data...")
    geographic_df = generate_geographic_data(cbsa_tracts)
    geographic_file = output_path / 'geographic.csv'
    geographic_df.to_csv(geographic_file, index=False)
    print(f"    Saved {len(geographic_df)} census tracts to {geographic_file}")
    
    # Generate weighting data
    print("  - Generating weighting data...")
    weighting_df = generate_weighting_data(cbsa_tracts)
    weighting_file = output_path / 'weighting.csv'
    weighting_df.to_csv(weighting_file, index=False)
    print(f"    Saved {len(weighting_df)} weighting records to {weighting_file}")
    
    # Print summary statistics
    print("\nSample data summary:")
    print(f"  - CBSAs: {len(cbsa_tracts)}")
    print(f"  - Census tracts: {sum(len(tracts) for tracts in cbsa_tracts.values())}")
    print(f"  - Unique properties: {transactions_df['property_id'].nunique()}")
    print(f"  - Date range: {transactions_df['transaction_date'].min()} to {transactions_df['transaction_date'].max()}")
    
    # Calculate repeat sales
    repeat_props = transactions_df.groupby('property_id').size()
    n_repeat = (repeat_props > 1).sum()
    print(f"  - Properties with repeat sales: {n_repeat} ({n_repeat/len(repeat_props)*100:.1f}%)")
    
    return {
        'transactions': transactions_file,
        'geographic': geographic_file,
        'weighting': weighting_file
    }


if __name__ == "__main__":
    # Generate sample data files
    file_paths = save_sample_data()
    
    print("\nSample data files created:")
    for name, path in file_paths.items():
        print(f"  - {name}: {path}")