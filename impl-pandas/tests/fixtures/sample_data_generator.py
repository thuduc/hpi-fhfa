"""Generate sample data for testing HPI-FHFA implementation."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path


def generate_transaction_data(
    n_properties: int = 1000,
    n_tracts: int = 10,
    n_cbsas: int = 3,
    start_date: str = '2015-01-01',
    end_date: str = '2021-12-31',
    repeat_sale_prob: float = 0.6,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate sample property transaction data.
    
    Parameters
    ----------
    n_properties : int
        Number of unique properties
    n_tracts : int
        Number of census tracts
    n_cbsas : int
        Number of CBSAs
    start_date : str
        Start date for transactions
    end_date : str
        End date for transactions
    repeat_sale_prob : float
        Probability that a property has multiple sales
    seed : int
        Random seed for reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate property IDs
    property_ids = [f'P{str(i).zfill(6)}' for i in range(1, n_properties + 1)]
    
    # Generate census tracts (11 digits)
    base_tract = 12345678900
    census_tracts = [str(base_tract + i) for i in range(n_tracts)]
    
    # Generate CBSA codes (5 digits)
    cbsa_codes = [str(10420 + i * 100) for i in range(n_cbsas)]
    
    # Assign properties to tracts and CBSAs
    property_tract_map = {}
    property_cbsa_map = {}
    property_cbd_dist_map = {}
    
    for prop_id in property_ids:
        tract = random.choice(census_tracts)
        # CBSAs are grouped by tracts
        cbsa_idx = census_tracts.index(tract) // (n_tracts // n_cbsas)
        cbsa = cbsa_codes[min(cbsa_idx, n_cbsas - 1)]
        
        property_tract_map[prop_id] = tract
        property_cbsa_map[prop_id] = cbsa
        # Distance to CBD: random between 0 and 50 miles
        property_cbd_dist_map[prop_id] = round(np.random.uniform(0.5, 50), 1)
    
    # Generate transactions
    transactions = []
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = (end - start).days
    
    for prop_id in property_ids:
        # Determine number of transactions
        if np.random.random() < repeat_sale_prob:
            n_trans = np.random.choice([2, 3, 4], p=[0.7, 0.25, 0.05])
        else:
            n_trans = 1
            
        # Generate transaction dates
        trans_dates = sorted([
            start + timedelta(days=np.random.randint(0, date_range))
            for _ in range(n_trans)
        ])
        
        # Ensure minimum spacing between transactions
        for i in range(1, len(trans_dates)):
            if (trans_dates[i] - trans_dates[i-1]).days < 365:
                trans_dates[i] = trans_dates[i-1] + timedelta(days=365 + np.random.randint(0, 365))
                
        # Generate prices with appreciation
        base_price = np.random.uniform(100000, 800000)
        annual_appreciation = np.random.normal(0.05, 0.02)  # 5% average, 2% std
        
        for i, date in enumerate(trans_dates):
            if i == 0:
                price = base_price
            else:
                years_passed = (date - trans_dates[0]).days / 365.25
                # Add some noise to appreciation
                noise = np.random.normal(1, 0.05)
                price = base_price * ((1 + annual_appreciation) ** years_passed) * noise
                
            transactions.append({
                'property_id': prop_id,
                'transaction_date': date,
                'transaction_price': round(price, 2),
                'census_tract': property_tract_map[prop_id],
                'cbsa_code': property_cbsa_map[prop_id],
                'distance_to_cbd': property_cbd_dist_map[prop_id]
            })
    
    return pd.DataFrame(transactions)


def generate_census_tract_data(
    census_tracts: list,
    cbsa_codes: list,
    n_tracts: int = 10,
    n_cbsas: int = 3,
    seed: int = 42
) -> pd.DataFrame:
    """Generate census tract geographic and demographic data."""
    np.random.seed(seed)
    
    tract_data = []
    
    for i, tract in enumerate(census_tracts):
        # Assign CBSA based on tract index
        cbsa_idx = i // (n_tracts // n_cbsas)
        cbsa = cbsa_codes[min(cbsa_idx, n_cbsas - 1)]
        
        # Generate centroid coordinates (example: around NYC area)
        base_lat = 40.7128
        base_lon = -74.0060
        lat = base_lat + np.random.uniform(-0.5, 0.5)
        lon = base_lon + np.random.uniform(-0.5, 0.5)
        
        # Generate demographic data
        housing_units = np.random.randint(500, 5000)
        avg_value = np.random.uniform(200000, 800000)
        
        tract_data.append({
            'census_tract': tract,
            'cbsa_code': cbsa,
            'centroid_lat': round(lat, 6),
            'centroid_lon': round(lon, 6),
            'housing_units': housing_units,
            'aggregate_value': round(housing_units * avg_value),
            'college_share': round(np.random.uniform(0.1, 0.6), 3),
            'nonwhite_share': round(np.random.uniform(0.1, 0.8), 3)
        })
    
    return pd.DataFrame(tract_data)


def save_sample_data(output_dir: str = 'tests/fixtures/data'):
    """Generate and save sample data files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate transaction data
    print("Generating transaction data...")
    transactions = generate_transaction_data(
        n_properties=5000,
        n_tracts=20,
        n_cbsas=5,
        repeat_sale_prob=0.7
    )
    
    # Save in multiple formats
    transactions.to_parquet(output_path / 'transactions.parquet', index=False)
    transactions.to_csv(output_path / 'transactions.csv', index=False)
    
    print(f"Generated {len(transactions):,} transactions from {transactions['property_id'].nunique():,} properties")
    
    # Generate census tract data
    print("\nGenerating census tract data...")
    unique_tracts = transactions['census_tract'].unique()
    unique_cbsas = transactions['cbsa_code'].unique()
    
    census_data = generate_census_tract_data(
        census_tracts=unique_tracts,
        cbsa_codes=unique_cbsas,
        n_tracts=len(unique_tracts),
        n_cbsas=len(unique_cbsas)
    )
    
    census_data.to_parquet(output_path / 'census_tracts.parquet', index=False)
    census_data.to_csv(output_path / 'census_tracts.csv', index=False)
    
    print(f"Generated data for {len(census_data):,} census tracts")
    
    # Create a small test dataset
    print("\nCreating small test dataset...")
    small_transactions = generate_transaction_data(
        n_properties=100,
        n_tracts=5,
        n_cbsas=2,
        repeat_sale_prob=0.8,
        seed=123
    )
    
    small_transactions.to_parquet(output_path / 'test_transactions_small.parquet', index=False)
    
    # Create extreme cases dataset for filter testing
    print("\nCreating extreme cases dataset...")
    extreme_data = create_extreme_cases_data()
    extreme_data.to_parquet(output_path / 'test_transactions_extreme.parquet', index=False)
    
    print("\nSample data generation complete!")
    print(f"Files saved to: {output_path}")
    
    return transactions, census_data


def create_extreme_cases_data() -> pd.DataFrame:
    """Create dataset with extreme cases for testing filters."""
    data = []
    
    # Normal appreciation (should pass all filters)
    data.extend([
        {
            'property_id': 'P_NORMAL_1',
            'transaction_date': pd.Timestamp('2018-01-01'),
            'transaction_price': 200000,
            'census_tract': '12345678901',
            'cbsa_code': '10420',
            'distance_to_cbd': 5.0
        },
        {
            'property_id': 'P_NORMAL_1',
            'transaction_date': pd.Timestamp('2020-01-01'),
            'transaction_price': 220000,  # 10% over 2 years = 4.9% CAGR
            'census_tract': '12345678901',
            'cbsa_code': '10420',
            'distance_to_cbd': 5.0
        }
    ])
    
    # High CAGR (should be filtered)
    data.extend([
        {
            'property_id': 'P_HIGH_CAGR',
            'transaction_date': pd.Timestamp('2019-01-01'),
            'transaction_price': 100000,
            'census_tract': '12345678901',
            'cbsa_code': '10420',
            'distance_to_cbd': 3.0
        },
        {
            'property_id': 'P_HIGH_CAGR',
            'transaction_date': pd.Timestamp('2020-01-01'),
            'transaction_price': 150000,  # 50% in 1 year
            'census_tract': '12345678901',
            'cbsa_code': '10420',
            'distance_to_cbd': 3.0
        }
    ])
    
    # Same year transaction (should be filtered)
    data.extend([
        {
            'property_id': 'P_SAME_YEAR',
            'transaction_date': pd.Timestamp('2020-03-01'),
            'transaction_price': 300000,
            'census_tract': '12345678902',
            'cbsa_code': '10420',
            'distance_to_cbd': 8.0
        },
        {
            'property_id': 'P_SAME_YEAR',
            'transaction_date': pd.Timestamp('2020-11-01'),
            'transaction_price': 310000,
            'census_tract': '12345678902',
            'cbsa_code': '10420',
            'distance_to_cbd': 8.0
        }
    ])
    
    # Extreme cumulative appreciation (should be filtered)
    data.extend([
        {
            'property_id': 'P_EXTREME_APP',
            'transaction_date': pd.Timestamp('2015-01-01'),
            'transaction_price': 50000,
            'census_tract': '12345678903',
            'cbsa_code': '10520',
            'distance_to_cbd': 15.0
        },
        {
            'property_id': 'P_EXTREME_APP',
            'transaction_date': pd.Timestamp('2020-01-01'),
            'transaction_price': 600000,  # 12x appreciation
            'census_tract': '12345678903',
            'cbsa_code': '10520',
            'distance_to_cbd': 15.0
        }
    ])
    
    # Extreme depreciation (should be filtered)
    data.extend([
        {
            'property_id': 'P_EXTREME_DEP',
            'transaction_date': pd.Timestamp('2018-01-01'),
            'transaction_price': 400000,
            'census_tract': '12345678903',
            'cbsa_code': '10520',
            'distance_to_cbd': 20.0
        },
        {
            'property_id': 'P_EXTREME_DEP',
            'transaction_date': pd.Timestamp('2020-01-01'),
            'transaction_price': 80000,  # 0.2x (80% loss)
            'census_tract': '12345678903',
            'cbsa_code': '10520',
            'distance_to_cbd': 20.0
        }
    ])
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    save_sample_data()