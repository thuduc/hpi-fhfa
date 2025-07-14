"""Unit tests for data ingestion module"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from rsai.src.data.ingestion import DataIngestion, RepeatSalesProcessor


class TestDataIngestion:
    """Test DataIngestion class"""
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data"""
        dates = pd.date_range('2018-01-01', '2021-12-31', freq='D')
        n_transactions = 1000
        
        data = {
            'property_id': [f'PROP{i:04d}' for i in np.random.randint(1, 300, n_transactions)],
            'transaction_date': np.random.choice(dates, n_transactions),
            'transaction_price': np.random.uniform(100000, 1000000, n_transactions),
            'census_tract_2010': [f'0603712345{i%10}' for i in range(n_transactions)],
            'cbsa_id': np.random.choice(['31080', '41860', '47900'], n_transactions)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_geographic_data(self):
        """Create sample geographic data"""
        tracts = [f'0603712345{i}' for i in range(10)]
        
        data = {
            'census_tract_2010': tracts,
            'centroid_lat': np.random.uniform(33.5, 34.5, 10),
            'centroid_lon': np.random.uniform(-119, -117, 10),
            'cbsa_id': ['31080'] * 4 + ['41860'] * 3 + ['47900'] * 3
        }
        
        return pd.DataFrame(data)
    
    def test_load_transaction_data(self, sample_transaction_data):
        """Test loading transaction data"""
        ingestion = DataIngestion()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_transaction_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Load data
            df = ingestion.load_transaction_data(temp_file)
            
            # Check data loaded correctly
            assert len(df) == len(sample_transaction_data)
            assert 'property_id' in df.columns
            assert 'transaction_date' in df.columns
            assert df['transaction_price'].min() > 0
        finally:
            os.unlink(temp_file)
    
    def test_missing_columns_error(self):
        """Test error when required columns are missing"""
        ingestion = DataIngestion()
        
        # Create data missing required column
        bad_data = pd.DataFrame({
            'property_id': ['PROP001'],
            'transaction_date': ['2020-01-01'],
            # Missing transaction_price
            'census_tract_2010': ['06037123456'],
            'cbsa_id': ['31080']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            bad_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                ingestion.load_transaction_data(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_load_geographic_data(self, sample_geographic_data):
        """Test loading geographic data"""
        ingestion = DataIngestion()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_geographic_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            df = ingestion.load_geographic_data(temp_file)
            
            assert len(df) == len(sample_geographic_data)
            assert all(-90 <= lat <= 90 for lat in df['centroid_lat'])
            assert all(-180 <= lon <= 180 for lon in df['centroid_lon'])
        finally:
            os.unlink(temp_file)


class TestRepeatSalesProcessor:
    """Test RepeatSalesProcessor class"""
    
    @pytest.fixture
    def repeat_sales_transactions(self):
        """Create transactions with known repeat sales"""
        data = []
        
        # Property with 3 sales
        for i, (date, price) in enumerate([
            ('2018-01-15', 200000),
            ('2019-06-20', 220000),
            ('2021-03-10', 250000)
        ]):
            data.append({
                'property_id': 'PROP001',
                'transaction_date': pd.to_datetime(date),
                'transaction_price': price,
                'census_tract_2010': '06037123456',
                'cbsa_id': '31080'
            })
        
        # Property with 2 sales
        for date, price in [
            ('2017-05-01', 300000),
            ('2020-08-15', 350000)
        ]:
            data.append({
                'property_id': 'PROP002',
                'transaction_date': pd.to_datetime(date),
                'transaction_price': price,
                'census_tract_2010': '06037123457',
                'cbsa_id': '31080'
            })
        
        # Property with only 1 sale (should not appear in repeat sales)
        data.append({
            'property_id': 'PROP003',
            'transaction_date': pd.to_datetime('2019-01-01'),
            'transaction_price': 400000,
            'census_tract_2010': '06037123458',
            'cbsa_id': '31080'
        })
        
        return pd.DataFrame(data)
    
    def test_identify_repeat_sales(self, repeat_sales_transactions):
        """Test identification of repeat sales pairs"""
        processor = RepeatSalesProcessor()
        
        repeat_sales = processor.identify_repeat_sales(repeat_sales_transactions)
        
        # Should have 3 pairs: 2 from PROP001 and 1 from PROP002
        assert len(repeat_sales) == 3
        
        # Check PROP001 pairs
        prop001_pairs = repeat_sales[repeat_sales['property_id'] == 'PROP001']
        assert len(prop001_pairs) == 2
        
        # Check PROP002 pair
        prop002_pairs = repeat_sales[repeat_sales['property_id'] == 'PROP002']
        assert len(prop002_pairs) == 1
        
        # PROP003 should not appear
        assert 'PROP003' not in repeat_sales['property_id'].values
    
    def test_calculate_price_relatives(self, repeat_sales_transactions):
        """Test calculation of price relatives"""
        processor = RepeatSalesProcessor()
        
        repeat_sales = processor.identify_repeat_sales(repeat_sales_transactions)
        repeat_sales_with_metrics = processor.calculate_price_relatives(repeat_sales)
        
        # Check calculations for first pair
        first_pair = repeat_sales_with_metrics.iloc[0]
        
        # Log price relative
        expected_log = np.log(first_pair['second_sale_price'] / first_pair['first_sale_price'])
        assert abs(first_pair['log_price_relative'] - expected_log) < 0.0001
        
        # Years between sales
        days_between = (first_pair['second_sale_date'] - first_pair['first_sale_date']).days
        expected_years = days_between / 365.25
        assert abs(first_pair['years_between_sales'] - expected_years) < 0.01
        
        # Annual growth rate
        expected_growth = (first_pair['second_sale_price'] / first_pair['first_sale_price']) ** (1 / expected_years) - 1
        assert abs(first_pair['annual_growth_rate'] - expected_growth) < 0.0001
    
    def test_apply_filters(self):
        """Test filtering of repeat sales pairs"""
        processor = RepeatSalesProcessor(
            min_period_months=12,
            max_annual_growth=0.30,
            max_appreciation_factor=10.0,
            min_appreciation_factor=0.25
        )
        
        # Create test data with various scenarios
        test_pairs = pd.DataFrame([
            # Valid pair
            {
                'property_id': 'PROP001',
                'first_sale_date': pd.to_datetime('2018-01-01'),
                'second_sale_date': pd.to_datetime('2020-01-01'),
                'first_sale_price': 200000,
                'second_sale_price': 250000,
                'census_tract_2010': '06037123456',
                'cbsa_id': '31080'
            },
            # Same year (should be filtered)
            {
                'property_id': 'PROP002',
                'first_sale_date': pd.to_datetime('2020-01-01'),
                'second_sale_date': pd.to_datetime('2020-06-01'),
                'first_sale_price': 300000,
                'second_sale_price': 310000,
                'census_tract_2010': '06037123456',
                'cbsa_id': '31080'
            },
            # Excessive growth (should be filtered)
            {
                'property_id': 'PROP003',
                'first_sale_date': pd.to_datetime('2018-01-01'),
                'second_sale_date': pd.to_datetime('2020-01-01'),
                'first_sale_price': 200000,
                'second_sale_price': 500000,  # 58% annual growth
                'census_tract_2010': '06037123456',
                'cbsa_id': '31080'
            },
            # Extreme appreciation (should be filtered)
            {
                'property_id': 'PROP004',
                'first_sale_date': pd.to_datetime('2018-01-01'),
                'second_sale_date': pd.to_datetime('2020-01-01'),
                'first_sale_price': 100000,
                'second_sale_price': 1500000,  # 15x appreciation
                'census_tract_2010': '06037123456',
                'cbsa_id': '31080'
            }
        ])
        
        # Calculate metrics
        test_pairs = processor.calculate_price_relatives(test_pairs)
        
        # Apply filters
        filtered = processor.apply_filters(test_pairs)
        
        # Only the first pair should remain
        assert len(filtered) == 1
        assert filtered.iloc[0]['property_id'] == 'PROP001'
    
    def test_process_repeat_sales_integration(self, repeat_sales_transactions):
        """Test complete repeat sales processing pipeline"""
        processor = RepeatSalesProcessor()
        
        result = processor.process_repeat_sales(repeat_sales_transactions)
        
        # Should have repeat sales identified and filtered
        assert len(result) > 0
        assert 'log_price_relative' in result.columns
        assert 'annual_growth_rate' in result.columns
        assert 'years_between_sales' in result.columns
        
        # All pairs should meet filter criteria
        assert all(result['years_between_sales'] >= 1.0)  # At least 12 months
        assert all(abs(result['annual_growth_rate']) <= 0.30)
        assert all(result['cumulative_appreciation'] <= 10.0)
        assert all(result['cumulative_appreciation'] >= 0.25)