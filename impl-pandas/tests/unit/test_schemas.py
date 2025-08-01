"""Unit tests for data validation schemas."""

import pytest
import pandas as pd
import numpy as np
import pandera as pa
from datetime import datetime, timedelta

from hpi_fhfa.data.schemas import (
    transaction_schema,
    census_tract_schema,
    repeat_sales_schema,
    validate_transactions,
    validate_census_tracts,
    validate_repeat_sales
)


class TestTransactionSchema:
    """Test transaction data validation schema."""
    
    def test_valid_transaction_data(self):
        # Create valid transaction data
        df = pd.DataFrame({
            'property_id': ['P001', 'P002', 'P003'],
            'transaction_date': pd.to_datetime(['2020-01-15', '2020-06-20', '2021-03-10']),
            'transaction_price': [250000.0, 350000.0, 425000.0],
            'census_tract': ['12345678901', '12345678902', '12345678903'],
            'cbsa_code': ['10420', '10420', '10420'],
            'distance_to_cbd': [5.2, 3.8, 7.1]
        })
        
        # Should validate without errors
        validated_df = validate_transactions(df)
        assert len(validated_df) == 3
        
    def test_invalid_property_id(self):
        df = pd.DataFrame({
            'property_id': [None, 'P002', 'P003'],  # None is invalid
            'transaction_date': pd.to_datetime(['2020-01-15', '2020-06-20', '2021-03-10']),
            'transaction_price': [250000.0, 350000.0, 425000.0],
            'census_tract': ['12345678901', '12345678902', '12345678903'],
            'cbsa_code': ['10420', '10420', '10420'],
            'distance_to_cbd': [5.2, 3.8, 7.1]
        })
        
        with pytest.raises(pa.errors.SchemaError):
            validate_transactions(df)
            
    def test_invalid_transaction_date(self):
        # Date outside valid range
        df = pd.DataFrame({
            'property_id': ['P001', 'P002', 'P003'],
            'transaction_date': pd.to_datetime(['1970-01-15', '2020-06-20', '2021-03-10']),
            'transaction_price': [250000.0, 350000.0, 425000.0],
            'census_tract': ['12345678901', '12345678902', '12345678903'],
            'cbsa_code': ['10420', '10420', '10420'],
            'distance_to_cbd': [5.2, 3.8, 7.1]
        })
        
        with pytest.raises(pa.errors.SchemaError):
            validate_transactions(df)
            
    def test_invalid_price(self):
        # Negative price
        df = pd.DataFrame({
            'property_id': ['P001', 'P002', 'P003'],
            'transaction_date': pd.to_datetime(['2020-01-15', '2020-06-20', '2021-03-10']),
            'transaction_price': [-250000.0, 350000.0, 425000.0],
            'census_tract': ['12345678901', '12345678902', '12345678903'],
            'cbsa_code': ['10420', '10420', '10420'],
            'distance_to_cbd': [5.2, 3.8, 7.1]
        })
        
        with pytest.raises(pa.errors.SchemaError):
            validate_transactions(df)
            
        # Price too high
        df['transaction_price'] = [1e10, 350000.0, 425000.0]
        with pytest.raises(pa.errors.SchemaError):
            validate_transactions(df)
            
    def test_invalid_census_tract(self):
        # Wrong length
        df = pd.DataFrame({
            'property_id': ['P001', 'P002', 'P003'],
            'transaction_date': pd.to_datetime(['2020-01-15', '2020-06-20', '2021-03-10']),
            'transaction_price': [250000.0, 350000.0, 425000.0],
            'census_tract': ['1234567890', '12345678902', '12345678903'],  # 10 digits instead of 11
            'cbsa_code': ['10420', '10420', '10420'],
            'distance_to_cbd': [5.2, 3.8, 7.1]
        })
        
        with pytest.raises(pa.errors.SchemaError):
            validate_transactions(df)
            
        # Non-numeric
        df['census_tract'] = ['12345678A01', '12345678902', '12345678903']
        with pytest.raises(pa.errors.SchemaError):
            validate_transactions(df)
            
    def test_invalid_cbsa_code(self):
        # Wrong length
        df = pd.DataFrame({
            'property_id': ['P001', 'P002', 'P003'],
            'transaction_date': pd.to_datetime(['2020-01-15', '2020-06-20', '2021-03-10']),
            'transaction_price': [250000.0, 350000.0, 425000.0],
            'census_tract': ['12345678901', '12345678902', '12345678903'],
            'cbsa_code': ['1042', '10420', '10420'],  # 4 digits instead of 5
            'distance_to_cbd': [5.2, 3.8, 7.1]
        })
        
        with pytest.raises(pa.errors.SchemaError):
            validate_transactions(df)
            
    def test_invalid_distance(self):
        # Negative distance
        df = pd.DataFrame({
            'property_id': ['P001', 'P002', 'P003'],
            'transaction_date': pd.to_datetime(['2020-01-15', '2020-06-20', '2021-03-10']),
            'transaction_price': [250000.0, 350000.0, 425000.0],
            'census_tract': ['12345678901', '12345678902', '12345678903'],
            'cbsa_code': ['10420', '10420', '10420'],
            'distance_to_cbd': [-5.2, 3.8, 7.1]
        })
        
        with pytest.raises(pa.errors.SchemaError):
            validate_transactions(df)
            
    def test_string_dates_converted(self):
        # String dates should be converted to datetime
        df = pd.DataFrame({
            'property_id': ['P001', 'P002', 'P003'],
            'transaction_date': ['2020-01-15', '2020-06-20', '2021-03-10'],  # Strings
            'transaction_price': [250000.0, 350000.0, 425000.0],
            'census_tract': ['12345678901', '12345678902', '12345678903'],
            'cbsa_code': ['10420', '10420', '10420'],
            'distance_to_cbd': [5.2, 3.8, 7.1]
        })
        
        validated_df = validate_transactions(df)
        assert pd.api.types.is_datetime64_any_dtype(validated_df['transaction_date'])


class TestCensusTractSchema:
    """Test census tract data validation schema."""
    
    def test_valid_census_data(self):
        df = pd.DataFrame({
            'census_tract': ['12345678901', '12345678902'],
            'cbsa_code': ['10420', '10420'],
            'centroid_lat': [40.7128, 40.7260],
            'centroid_lon': [-74.0060, -73.9897],
            'housing_units': [1500, 2000],
            'aggregate_value': [450000000.0, 600000000.0],
            'college_share': [0.35, 0.42],
            'nonwhite_share': [0.28, 0.31]
        })
        
        validated_df = validate_census_tracts(df)
        assert len(validated_df) == 2
        
    def test_duplicate_census_tracts(self):
        df = pd.DataFrame({
            'census_tract': ['12345678901', '12345678901'],  # Duplicate
            'cbsa_code': ['10420', '10420'],
            'centroid_lat': [40.7128, 40.7260],
            'centroid_lon': [-74.0060, -73.9897],
            'housing_units': [1500, 2000],
            'aggregate_value': [450000000.0, 600000000.0],
            'college_share': [0.35, 0.42],
            'nonwhite_share': [0.28, 0.31]
        })
        
        with pytest.raises(pa.errors.SchemaError):
            validate_census_tracts(df)
            
    def test_invalid_coordinates(self):
        # Latitude out of range
        df = pd.DataFrame({
            'census_tract': ['12345678901', '12345678902'],
            'cbsa_code': ['10420', '10420'],
            'centroid_lat': [95.0, 40.7260],  # > 90
            'centroid_lon': [-74.0060, -73.9897],
            'housing_units': [1500, 2000],
            'aggregate_value': [450000000.0, 600000000.0],
            'college_share': [0.35, 0.42],
            'nonwhite_share': [0.28, 0.31]
        })
        
        with pytest.raises(pa.errors.SchemaError):
            validate_census_tracts(df)
            
        # Longitude out of range
        df['centroid_lat'] = [40.7128, 40.7260]
        df['centroid_lon'] = [-74.0060, 185.0]  # > 180
        
        with pytest.raises(pa.errors.SchemaError):
            validate_census_tracts(df)
            
    def test_invalid_shares(self):
        # Share > 1
        df = pd.DataFrame({
            'census_tract': ['12345678901', '12345678902'],
            'cbsa_code': ['10420', '10420'],
            'centroid_lat': [40.7128, 40.7260],
            'centroid_lon': [-74.0060, -73.9897],
            'housing_units': [1500, 2000],
            'aggregate_value': [450000000.0, 600000000.0],
            'college_share': [1.5, 0.42],  # > 1
            'nonwhite_share': [0.28, 0.31]
        })
        
        with pytest.raises(pa.errors.SchemaError):
            validate_census_tracts(df)
            
        # Share < 0
        df['college_share'] = [-0.1, 0.42]  # < 0
        
        with pytest.raises(pa.errors.SchemaError):
            validate_census_tracts(df)
            
    def test_optional_fields(self):
        # Only required fields
        df = pd.DataFrame({
            'census_tract': ['12345678901', '12345678902'],
            'cbsa_code': ['10420', '10420'],
            'centroid_lat': [40.7128, 40.7260],
            'centroid_lon': [-74.0060, -73.9897]
        })
        
        validated_df = validate_census_tracts(df)
        assert len(validated_df) == 2


class TestRepeatSalesSchema:
    """Test repeat sales pair validation schema."""
    
    def test_valid_repeat_sales(self):
        df = pd.DataFrame({
            'property_id': ['P001', 'P002'],
            'sale1_date': pd.to_datetime(['2019-01-15', '2018-06-20']),
            'sale1_price': [250000.0, 300000.0],
            'sale2_date': pd.to_datetime(['2021-01-15', '2020-06-20']),
            'sale2_price': [300000.0, 350000.0],
            'census_tract': ['12345678901', '12345678902'],
            'cbsa_code': ['10420', '10420'],
            'distance_to_cbd': [5.2, 3.8],
            'price_relative': [0.1823, 0.1542],
            'time_diff_years': [2.0, 2.0],
            'cagr': [0.095, 0.08]
        })
        
        validated_df = validate_repeat_sales(df)
        assert len(validated_df) == 2
        
    def test_invalid_date_order(self):
        df = pd.DataFrame({
            'property_id': ['P001', 'P002'],
            'sale1_date': pd.to_datetime(['2021-01-15', '2018-06-20']),  # sale1 > sale2
            'sale1_price': [250000.0, 300000.0],
            'sale2_date': pd.to_datetime(['2019-01-15', '2020-06-20']),
            'sale2_price': [300000.0, 350000.0],
            'census_tract': ['12345678901', '12345678902'],
            'cbsa_code': ['10420', '10420'],
            'distance_to_cbd': [5.2, 3.8],
            'price_relative': [0.1823, 0.1542],
            'time_diff_years': [2.0, 2.0],
            'cagr': [0.095, 0.08]
        })
        
        with pytest.raises(ValueError, match="sale2_date values must be greater than sale1_date"):
            validate_repeat_sales(df)
            
    def test_invalid_prices(self):
        # Zero price
        df = pd.DataFrame({
            'property_id': ['P001', 'P002'],
            'sale1_date': pd.to_datetime(['2019-01-15', '2018-06-20']),
            'sale1_price': [0.0, 300000.0],  # Zero price
            'sale2_date': pd.to_datetime(['2021-01-15', '2020-06-20']),
            'sale2_price': [300000.0, 350000.0],
            'census_tract': ['12345678901', '12345678902'],
            'cbsa_code': ['10420', '10420'],
            'distance_to_cbd': [5.2, 3.8],
            'price_relative': [0.1823, 0.1542],
            'time_diff_years': [2.0, 2.0],
            'cagr': [0.095, 0.08]
        })
        
        with pytest.raises(pa.errors.SchemaError):
            validate_repeat_sales(df)
            
    def test_invalid_time_diff(self):
        # Negative time difference
        df = pd.DataFrame({
            'property_id': ['P001', 'P002'],
            'sale1_date': pd.to_datetime(['2019-01-15', '2018-06-20']),
            'sale1_price': [250000.0, 300000.0],
            'sale2_date': pd.to_datetime(['2021-01-15', '2020-06-20']),
            'sale2_price': [300000.0, 350000.0],
            'census_tract': ['12345678901', '12345678902'],
            'cbsa_code': ['10420', '10420'],
            'distance_to_cbd': [5.2, 3.8],
            'price_relative': [0.1823, 0.1542],
            'time_diff_years': [-2.0, 2.0],  # Negative
            'cagr': [0.095, 0.08]
        })
        
        with pytest.raises(pa.errors.SchemaError):
            validate_repeat_sales(df)