"""Unit tests for transaction filters."""

import pytest
import polars as pl
from datetime import date, timedelta
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.hpi_fhfa.data.filters import TransactionFilter
from src.hpi_fhfa.config.constants import (
    MAX_CAGR_THRESHOLD,
    MAX_CUMULATIVE_APPRECIATION,
    MIN_CUMULATIVE_APPRECIATION
)


class TestTransactionFilter:
    """Test transaction filtering functionality."""
    
    def test_filter_same_period(self, sample_repeat_sales):
        """Test filtering of same-year transactions."""
        # Create some same-year transactions
        same_year_df = pl.DataFrame({
            "property_id": ["P001", "P002"],
            "transaction_date": [date(2020, 12, 1), date(2019, 12, 1)],
            "prev_transaction_date": [date(2020, 1, 1), date(2019, 1, 1)],  # Same year
            "transaction_price": [110000.0, 220000.0],
            "prev_transaction_price": [100000.0, 200000.0],
            "log_price_diff": [0.095, 0.095],
            "time_diff_years": [0.9, 0.9],
            "cagr": [0.105, 0.105],
            "census_tract": ["06037000100", "06037000200"],
            "cbsa_code": ["31080", "31080"],
            "distance_to_cbd": [5.0, 10.0]
        })
        
        # Also add some valid cross-year transactions
        valid_df = sample_repeat_sales.head(3).select(same_year_df.columns)
        
        df_combined = pl.concat([same_year_df, valid_df])
        initial_count = len(df_combined)
        
        filter = TransactionFilter()
        result = filter._filter_same_period(df_combined)
        
        # Check that we removed same-year transactions
        # The exact count depends on the random sample data
        assert len(result) < initial_count
        
        # More importantly, verify NO same-year transactions remain
        
        # Verify no same-year transactions remain
        result_with_years = result.with_columns([
            pl.col("transaction_date").dt.year().alias("sale_year"),
            pl.col("prev_transaction_date").dt.year().alias("prev_year")
        ])
        assert (result_with_years["sale_year"] == result_with_years["prev_year"]).sum() == 0
    
    def test_filter_cagr(self, sample_repeat_sales):
        """Test CAGR filtering."""
        # Add transactions with extreme CAGR
        extreme_cagr = sample_repeat_sales.head(3).with_columns([
            # Create 50% annual growth
            (pl.col("prev_transaction_price") * (1.5 ** pl.col("time_diff_years"))).alias("transaction_price")
        ])
        
        # Recalculate CAGR for modified data
        extreme_cagr = extreme_cagr.with_columns([
            (
                ((pl.col("transaction_price") / pl.col("prev_transaction_price"))
                 .pow(1.0 / pl.col("time_diff_years")) - 1)
                .abs()
            ).alias("cagr")
        ])
        
        df_with_extreme = pl.concat([sample_repeat_sales, extreme_cagr])
        initial_count = len(df_with_extreme)
        
        filter = TransactionFilter()
        result = filter._filter_cagr(df_with_extreme)
        
        # Should have removed high CAGR transactions
        assert len(result) < initial_count
        assert result["cagr"].max() <= MAX_CAGR_THRESHOLD
    
    def test_filter_cumulative_appreciation_high(self):
        """Test filtering of extreme appreciation (>10x)."""
        # Create test data with extreme appreciation
        df = pl.DataFrame({
            "property_id": ["P001", "P002", "P003"],
            "transaction_date": [date(2020, 1, 1)] * 3,
            "prev_transaction_date": [date(2010, 1, 1)] * 3,
            "transaction_price": [1_000_000.0, 15_000_000.0, 500_000.0],  # 10x, 15x, 5x
            "prev_transaction_price": [100_000.0, 1_000_000.0, 100_000.0],
            "time_diff_years": [10.0] * 3,
            "log_price_diff": [2.303, 2.708, 1.609],  # log ratios
            "cagr": [0.259, 0.311, 0.175],
            "census_tract": ["06037000100"] * 3,
            "cbsa_code": ["31080"] * 3,
            "distance_to_cbd": [5.0] * 3
        })
        
        filter = TransactionFilter()
        result = filter._filter_cumulative_appreciation(df)
        
        # 10x is the boundary (should be excluded), so should keep P001 (10x) and P003 (5x)
        # Actually, let me check the constant...
        # MAX_CUMULATIVE_APPRECIATION = 10.0, so 10x should be included
        assert len(result) == 2  # P001 (10x) and P003 (5x)
        property_ids = result["property_id"].to_list()
        assert "P002" not in property_ids  # 15x should be excluded
        assert "P001" in property_ids  # 10x should be included
        assert "P003" in property_ids  # 5x should be included
    
    def test_filter_cumulative_appreciation_low(self):
        """Test filtering of extreme depreciation (<0.25x)."""
        # Create test data with extreme depreciation
        df = pl.DataFrame({
            "property_id": ["P001", "P002", "P003"],
            "transaction_date": [date(2020, 1, 1)] * 3,
            "prev_transaction_date": [date(2010, 1, 1)] * 3,
            "transaction_price": [20_000, 50_000, 200_000],  # 0.2x, 0.5x, 2x
            "prev_transaction_price": [100_000, 100_000, 100_000],
            "time_diff_years": [10.0] * 3,
            "census_tract": ["06037000100"] * 3,
            "cbsa_code": ["31080"] * 3,
            "distance_to_cbd": [5.0] * 3
        })
        
        filter = TransactionFilter()
        result = filter._filter_cumulative_appreciation(df)
        
        # Should keep only 0.5x and 2x
        assert len(result) == 2
        assert "P001" not in result["property_id"].to_list()
    
    def test_apply_all_filters(self, sample_repeat_sales):
        """Test applying all filters together."""
        filter = TransactionFilter()
        result = filter.apply_filters(sample_repeat_sales)
        
        # Should have applied all filters
        assert len(result) <= len(sample_repeat_sales)
        assert len(filter.filters_applied) == 3
        
        # Verify all filter names are present
        filter_names = [f["filter"] for f in filter.filters_applied]
        assert "same_period" in filter_names
        assert "cagr" in filter_names
        assert "cumulative_appreciation" in filter_names
    
    def test_filter_summary(self, sample_repeat_sales):
        """Test filter summary generation."""
        filter = TransactionFilter()
        result = filter.apply_filters(sample_repeat_sales)
        
        summary = filter.get_filter_summary()
        
        assert isinstance(summary, pl.DataFrame)
        assert len(summary) == 3  # Three filters
        assert "filter" in summary.columns
        assert "removed" in summary.columns
        assert "pct" in summary.columns
    
    def test_no_data_removed(self):
        """Test when no data is filtered out."""
        # Create data that passes all filters
        df = pl.DataFrame({
            "property_id": ["P001", "P002"],
            "transaction_date": [date(2020, 1, 1), date(2019, 1, 1)],
            "prev_transaction_date": [date(2018, 1, 1), date(2015, 1, 1)],
            "transaction_price": [120_000, 110_000],
            "prev_transaction_price": [100_000, 100_000],
            "time_diff_years": [2.0, 4.0],
            "cagr": [0.095, 0.024],  # ~10% and ~2.4% annual growth
            "census_tract": ["06037000100", "06037000200"],
            "cbsa_code": ["31080", "31080"],
            "distance_to_cbd": [5.0, 10.0],
            "log_price_diff": [0.182, 0.095]
        })
        
        filter = TransactionFilter()
        result = filter.apply_filters(df)
        
        assert len(result) == len(df)
        # All filters should show 0 removed
        for f in filter.filters_applied:
            assert f["removed"] == 0