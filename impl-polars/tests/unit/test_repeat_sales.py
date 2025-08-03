"""Unit tests for repeat sales identification."""

import pytest
import polars as pl
from datetime import date, timedelta
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.hpi_fhfa.processing.repeat_sales import RepeatSalesIdentifier


class TestRepeatSalesIdentifier:
    """Test repeat sales identification functionality."""
    
    def test_identify_repeat_sales_basic(self, sample_transactions):
        """Test basic repeat sales identification."""
        identifier = RepeatSalesIdentifier()
        repeat_sales = identifier.identify_repeat_sales(sample_transactions)
        
        # Should have fewer rows than original (only repeat sales)
        assert len(repeat_sales) < len(sample_transactions)
        
        # Should have additional columns
        assert "prev_transaction_date" in repeat_sales.columns
        assert "prev_transaction_price" in repeat_sales.columns
        assert "log_price_diff" in repeat_sales.columns
        assert "time_diff_years" in repeat_sales.columns
        assert "cagr" in repeat_sales.columns
        
        # Previous transaction should be before current
        assert (repeat_sales["transaction_date"] > repeat_sales["prev_transaction_date"]).all()
    
    def test_identify_repeat_sales_single_property_chain(self):
        """Test identification with a chain of sales for one property."""
        # Create a property with 4 sequential sales
        df = pl.DataFrame({
            "property_id": ["P001"] * 4,
            "transaction_date": [
                date(2015, 1, 1),
                date(2016, 6, 1),
                date(2018, 3, 1),
                date(2020, 12, 1)
            ],
            "transaction_price": [100000, 110000, 125000, 150000],
            "census_tract": ["06037000100"] * 4,
            "cbsa_code": ["31080"] * 4,
            "distance_to_cbd": [5.0] * 4
        })
        
        identifier = RepeatSalesIdentifier()
        repeat_sales = identifier.identify_repeat_sales(df)
        
        # Should have 3 repeat sales (pairs: 1-2, 2-3, 3-4)
        assert len(repeat_sales) == 3
        
        # Check the pairs are correct
        pairs = repeat_sales.sort("transaction_date").to_dicts()
        assert pairs[0]["transaction_price"] == 110000
        assert pairs[0]["prev_transaction_price"] == 100000
        assert pairs[1]["transaction_price"] == 125000
        assert pairs[1]["prev_transaction_price"] == 110000
        assert pairs[2]["transaction_price"] == 150000
        assert pairs[2]["prev_transaction_price"] == 125000
    
    def test_identify_repeat_sales_min_days_filter(self):
        """Test minimum days between sales filter."""
        # Create sales with varying time gaps
        df = pl.DataFrame({
            "property_id": ["P001", "P001", "P002", "P002"],
            "transaction_date": [
                date(2020, 1, 1),
                date(2020, 6, 1),  # 151 days later
                date(2020, 1, 1),
                date(2021, 1, 15)  # 380 days later
            ],
            "transaction_price": [100000, 105000, 200000, 220000],
            "census_tract": ["06037000100"] * 4,
            "cbsa_code": ["31080"] * 4,
            "distance_to_cbd": [5.0] * 4
        })
        
        identifier = RepeatSalesIdentifier()
        
        # With 365 day minimum
        repeat_sales_365 = identifier.identify_repeat_sales(df, min_days_between_sales=365)
        assert len(repeat_sales_365) == 1  # Only P002
        assert repeat_sales_365["property_id"][0] == "P002"
        
        # With 100 day minimum
        repeat_sales_100 = identifier.identify_repeat_sales(df, min_days_between_sales=100)
        assert len(repeat_sales_100) == 2  # Both properties
    
    def test_derived_fields_calculation(self):
        """Test calculation of derived fields."""
        # Create simple test case with known values
        df = pl.DataFrame({
            "property_id": ["P001", "P001"],
            "transaction_date": [date(2015, 1, 1), date(2017, 1, 1)],
            "transaction_price": [100000, 121000],  # 10% annual growth
            "census_tract": ["06037000100"] * 2,
            "cbsa_code": ["31080"] * 2,
            "distance_to_cbd": [5.0] * 2
        })
        
        identifier = RepeatSalesIdentifier()
        repeat_sales = identifier.identify_repeat_sales(df, min_days_between_sales=0)
        
        assert len(repeat_sales) == 1
        row = repeat_sales[0]
        
        # Check log price difference
        expected_log_diff = np.log(121000) - np.log(100000)
        assert abs(row["log_price_diff"][0] - expected_log_diff) < 0.0001
        
        # Check time difference
        assert abs(row["time_diff_years"][0] - 2.0) < 0.01
        
        # Check CAGR (should be ~10%)
        assert abs(row["cagr"][0] - 0.1) < 0.001
    
    def test_statistics_calculation(self, sample_transactions):
        """Test statistics calculation."""
        identifier = RepeatSalesIdentifier()
        repeat_sales = identifier.identify_repeat_sales(sample_transactions)
        
        stats = identifier.get_statistics()
        
        # Check all expected statistics are present
        assert "n_transactions" in stats
        assert "n_properties" in stats
        assert "n_repeat_sales" in stats
        assert "n_repeat_properties" in stats
        assert "repeat_sales_pct" in stats
        assert "properties_with_repeats_pct" in stats
        assert "avg_time_between_sales_years" in stats
        assert "median_time_between_sales_years" in stats
        assert "avg_price_appreciation" in stats
        assert "median_cagr" in stats
        
        # Sanity checks
        assert stats["n_repeat_sales"] <= stats["n_transactions"]
        assert stats["n_repeat_properties"] <= stats["n_properties"]
        assert 0 <= stats["repeat_sales_pct"] <= 100
        assert 0 <= stats["properties_with_repeats_pct"] <= 100
    
    def test_create_balanced_panel(self):
        """Test balanced panel creation."""
        # Create sparse repeat sales data
        df = pl.DataFrame({
            "property_id": ["P001", "P002", "P003"],
            "transaction_date": [date(2017, 1, 1), date(2019, 1, 1), date(2020, 1, 1)],
            "prev_transaction_date": [date(2015, 1, 1), date(2015, 1, 1), date(2018, 1, 1)],
            "transaction_price": [110000, 130000, 150000],
            "prev_transaction_price": [100000, 100000, 120000],
            "log_price_diff": [0.095, 0.262, 0.223],
            "time_diff_years": [2.0, 4.0, 2.0]
        })
        
        identifier = RepeatSalesIdentifier()
        balanced = identifier.create_balanced_panel(df, 2015, 2020)
        
        # Check structure
        assert "prev_period" in balanced.columns
        assert "sale_period" in balanced.columns
        assert "n_observations" in balanced.columns
        
        # Should have all forward-looking pairs
        expected_pairs = 0
        for prev in range(2015, 2020):
            for curr in range(prev + 1, 2021):
                expected_pairs += 1
        
        assert len(balanced) == expected_pairs
        
        # Check specific counts
        count_2015_2017 = balanced.filter(
            (pl.col("prev_period") == 2015) & (pl.col("sale_period") == 2017)
        )["n_observations"][0]
        assert count_2015_2017 == 1  # P001
        
        count_2016_2017 = balanced.filter(
            (pl.col("prev_period") == 2016) & (pl.col("sale_period") == 2017)
        )["n_observations"][0]
        assert count_2016_2017 == 0  # No such pair in data
    
    def test_empty_input(self):
        """Test handling of empty input."""
        empty_df = pl.DataFrame({
            "property_id": [],
            "transaction_date": [],
            "transaction_price": [],
            "census_tract": [],
            "cbsa_code": [],
            "distance_to_cbd": []
        }).with_columns([
            pl.col("transaction_date").cast(pl.Date),
            pl.col("transaction_price").cast(pl.Float64),
            pl.col("distance_to_cbd").cast(pl.Float64)
        ])
        
        identifier = RepeatSalesIdentifier()
        repeat_sales = identifier.identify_repeat_sales(empty_df)
        
        assert len(repeat_sales) == 0
        assert identifier.get_statistics()["n_transactions"] == 0