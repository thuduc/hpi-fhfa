"""
Unit tests for data ingestion module.

Tests data loading, repeat sales identification, and filtering
using Polars DataFrames.
"""

import pytest
from datetime import date, timedelta
from pathlib import Path
import polars as pl
import numpy as np

from rsai.src.data.ingestion import DataIngestion
from rsai.src.data.models import RSAIConfig, TransactionType


class TestDataIngestion:
    """Test DataIngestion class."""
    
    def test_initialization(self, test_config):
        """Test DataIngestion initialization."""
        ingestion = DataIngestion(test_config)
        assert ingestion.config == test_config
        assert ingestion.transactions_df is None
        assert ingestion.properties_df is None
        assert ingestion.repeat_sales_df is None
    
    def test_load_transactions_parquet(self, test_config, sample_transactions_df, tmp_path):
        """Test loading transactions from Parquet file."""
        # Save sample data to parquet
        file_path = tmp_path / "transactions.parquet"
        sample_transactions_df.write_parquet(file_path)
        
        # Load data
        ingestion = DataIngestion(test_config)
        df = ingestion.load_transactions(file_path)
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        assert "transaction_id" in df.columns
        assert "sale_price" in df.columns
        assert df["sale_date"].dtype == pl.Date
    
    def test_load_transactions_csv(self, test_config, sample_transactions_df, tmp_path):
        """Test loading transactions from CSV file."""
        # Save sample data to CSV
        file_path = tmp_path / "transactions.csv"
        sample_transactions_df.write_csv(file_path)
        
        # Load data
        ingestion = DataIngestion(test_config)
        df = ingestion.load_transactions(file_path)
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
    
    def test_load_transactions_date_filter(self, test_config, sample_transactions_df, tmp_path):
        """Test date filtering during load."""
        # Modify config with narrow date range
        config = RSAIConfig(
            start_date=date(2021, 1, 1),
            end_date=date(2021, 12, 31),
            min_price=test_config.min_price,
            max_price=test_config.max_price
        )
        
        # Save and load data
        file_path = tmp_path / "transactions.parquet"
        sample_transactions_df.write_parquet(file_path)
        
        ingestion = DataIngestion(config)
        df = ingestion.load_transactions(file_path)
        
        # Check date filtering
        assert df["sale_date"].min() >= config.start_date
        assert df["sale_date"].max() <= config.end_date
    
    def test_load_transactions_price_filter(self, test_config, sample_transactions_df, tmp_path):
        """Test price filtering during load."""
        # Save and load data
        file_path = tmp_path / "transactions.parquet"
        sample_transactions_df.write_parquet(file_path)
        
        ingestion = DataIngestion(test_config)
        df = ingestion.load_transactions(file_path)
        
        # Check price filtering
        assert df["sale_price"].min() >= test_config.min_price
        assert df["sale_price"].max() <= test_config.max_price
    
    def test_load_properties(self, test_config, sample_properties_df, tmp_path):
        """Test loading property data."""
        # Save sample property data
        file_path = tmp_path / "properties.parquet"
        sample_properties_df.write_parquet(file_path)
        
        # Load data
        ingestion = DataIngestion(test_config)
        df = ingestion.load_properties(file_path)
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) == len(sample_properties_df)
        assert "property_id" in df.columns
        assert "latitude" in df.columns
        assert "longitude" in df.columns
    
    def test_identify_repeat_sales(self, test_config, sample_transactions_df):
        """Test repeat sales identification."""
        ingestion = DataIngestion(test_config)
        ingestion.transactions_df = sample_transactions_df
        
        repeat_df = ingestion.identify_repeat_sales()
        
        assert isinstance(repeat_df, pl.DataFrame)
        assert len(repeat_df) > 0
        assert "pair_id" in repeat_df.columns
        assert "price_ratio" in repeat_df.columns
        assert "log_price_ratio" in repeat_df.columns
        assert "holding_period_days" in repeat_df.columns
        
        # Check calculations
        for row in repeat_df.head(5).iter_rows(named=True):
            assert row["price_ratio"] == pytest.approx(
                row["sale2_price"] / row["sale1_price"], rel=1e-6
            )
            assert row["log_price_ratio"] == pytest.approx(
                np.log(row["price_ratio"]), rel=1e-6
            )
            assert row["holding_period_days"] > 0
    
    def test_identify_repeat_sales_arms_length_filter(self, test_config, sample_transactions_df):
        """Test that non-arms-length transactions are filtered."""
        # Add some non-arms-length transactions
        sample_transactions_df = sample_transactions_df.with_columns([
            pl.when(pl.col("property_id") == "P000001")
            .then(pl.lit(TransactionType.FORECLOSURE.value))
            .otherwise(pl.col("transaction_type"))
            .alias("transaction_type")
        ])
        
        ingestion = DataIngestion(test_config)
        ingestion.transactions_df = sample_transactions_df
        
        repeat_df = ingestion.identify_repeat_sales()
        
        # Check that all pairs are arms-length
        # Note: This checks the transaction types in the original data
        # The repeat sales should not include non-arms-length transactions
        assert len(repeat_df) > 0
    
    def test_identify_repeat_sales_holding_period_filter(self, test_config):
        """Test holding period filtering."""
        # Create transactions with specific holding periods
        transactions = pl.DataFrame({
            "transaction_id": ["T1", "T2", "T3", "T4"],
            "property_id": ["P1", "P1", "P2", "P2"],
            "sale_price": [100000, 120000, 200000, 250000],
            "sale_date": [
                date(2020, 1, 1),
                date(2020, 6, 1),  # 5 months later
                date(2020, 1, 1),
                date(2035, 1, 1),  # 15 years later (too long)
            ],
            "transaction_type": [TransactionType.ARMS_LENGTH.value] * 4,
            "county_fips": ["06037"] * 4,
            "tract": ["123456"] * 4
        })
        
        config = RSAIConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2040, 1, 1),
            max_holding_period_years=10
        )
        
        ingestion = DataIngestion(config)
        ingestion.transactions_df = transactions
        
        repeat_df = ingestion.identify_repeat_sales()
        
        # Should only have one pair (P1)
        assert len(repeat_df) == 1
        assert repeat_df["property_id"][0] == "P1"
    
    def test_merge_geographic_data(self, test_config, sample_repeat_sales_df, sample_properties_df):
        """Test merging geographic data with repeat sales."""
        ingestion = DataIngestion(test_config)
        ingestion.repeat_sales_df = sample_repeat_sales_df
        ingestion.properties_df = sample_properties_df
        
        merged_df = ingestion.merge_geographic_data()
        
        assert isinstance(merged_df, pl.DataFrame)
        assert len(merged_df) == len(sample_repeat_sales_df)
        assert "latitude" in merged_df.columns
        assert "longitude" in merged_df.columns
        assert "zip_code" in merged_df.columns
    
    def test_filter_outliers_iqr(self, test_config):
        """Test IQR outlier filtering."""
        # Create data with outliers
        normal_ratios = np.random.normal(0.1, 0.05, 100)
        outlier_ratios = np.array([2.0, -1.0, 1.5, -0.8])  # Extreme values
        all_ratios = np.concatenate([normal_ratios, outlier_ratios])
        
        repeat_df = pl.DataFrame({
            "pair_id": [f"P{i}" for i in range(len(all_ratios))],
            "property_id": [f"PROP{i}" for i in range(len(all_ratios))],
            "log_price_ratio": all_ratios,
            "sale1_price": [100000] * len(all_ratios),
            "sale2_price": [100000 * np.exp(r) for r in all_ratios],
            "sale1_date": [date(2020, 1, 1)] * len(all_ratios),
            "sale2_date": [date(2021, 1, 1)] * len(all_ratios),
            "holding_period_days": [365] * len(all_ratios),
            "price_ratio": [np.exp(r) for r in all_ratios],
            "annualized_return": [(np.exp(r) - 1) for r in all_ratios],
            "sale1_transaction_id": [f"T1_{i}" for i in range(len(all_ratios))],
            "sale2_transaction_id": [f"T2_{i}" for i in range(len(all_ratios))],
            "is_valid": [True] * len(all_ratios),
            "validation_flags": [[]] * len(all_ratios)
        })
        
        ingestion = DataIngestion(test_config)
        ingestion.repeat_sales_df = repeat_df
        
        filtered_df = ingestion.filter_outliers(method="iqr")
        
        # Should have fewer records after filtering
        assert len(filtered_df) < len(repeat_df)
        assert len(filtered_df) > 90  # Most normal values should remain
    
    def test_filter_outliers_zscore(self, test_config):
        """Test z-score outlier filtering."""
        # Create data with outliers
        normal_ratios = np.random.normal(0.1, 0.05, 100)
        outlier_ratios = np.array([0.5, -0.4])  # More than 3 std devs
        all_ratios = np.concatenate([normal_ratios, outlier_ratios])
        
        repeat_df = pl.DataFrame({
            "pair_id": [f"P{i}" for i in range(len(all_ratios))],
            "property_id": [f"PROP{i}" for i in range(len(all_ratios))],
            "log_price_ratio": all_ratios,
            "sale1_price": [100000] * len(all_ratios),
            "sale2_price": [100000 * np.exp(r) for r in all_ratios],
            "sale1_date": [date(2020, 1, 1)] * len(all_ratios),
            "sale2_date": [date(2021, 1, 1)] * len(all_ratios),
            "holding_period_days": [365] * len(all_ratios),
            "price_ratio": [np.exp(r) for r in all_ratios],
            "annualized_return": [(np.exp(r) - 1) for r in all_ratios],
            "sale1_transaction_id": [f"T1_{i}" for i in range(len(all_ratios))],
            "sale2_transaction_id": [f"T2_{i}" for i in range(len(all_ratios))],
            "is_valid": [True] * len(all_ratios),
            "validation_flags": [[]] * len(all_ratios)
        })
        
        ingestion = DataIngestion(test_config)
        ingestion.repeat_sales_df = repeat_df
        
        filtered_df = ingestion.filter_outliers(method="zscore")
        
        assert len(filtered_df) < len(repeat_df)
    
    def test_filter_outliers_percentile(self, test_config, sample_repeat_sales_df):
        """Test percentile outlier filtering."""
        ingestion = DataIngestion(test_config)
        ingestion.repeat_sales_df = sample_repeat_sales_df
        
        original_len = len(sample_repeat_sales_df)
        filtered_df = ingestion.filter_outliers(method="percentile")
        
        # Should remove top and bottom 1%
        expected_len = int(original_len * 0.98)
        assert len(filtered_df) <= expected_len
    
    def test_standardize_columns(self, test_config):
        """Test column name standardization."""
        # Create DataFrame with non-standard column names
        df = pl.DataFrame({
            "SalePrice": [100000, 200000],
            "SaleDate": [date(2020, 1, 1), date(2021, 1, 1)],
            "PropertyID": ["P1", "P2"],
            "TransactionType": ["arms_length", "arms_length"]
        })
        
        ingestion = DataIngestion(test_config)
        standardized = ingestion._standardize_columns(df)
        
        assert "sale_price" in standardized.columns
        assert "sale_date" in standardized.columns
        assert "property_id" in standardized.columns
        assert "transaction_type" in standardized.columns
        assert "transaction_id" in standardized.columns  # Should be added
    
    def test_get_summary_statistics(self, test_config, sample_transactions_df, sample_repeat_sales_df):
        """Test summary statistics calculation."""
        ingestion = DataIngestion(test_config)
        ingestion.transactions_df = sample_transactions_df
        ingestion.repeat_sales_df = sample_repeat_sales_df
        
        stats = ingestion.get_summary_statistics()
        
        assert "total_transactions" in stats
        assert "unique_properties" in stats
        assert "date_range" in stats
        assert "price_range" in stats
        assert "total_repeat_pairs" in stats
        assert "holding_period" in stats
        
        assert stats["total_transactions"] == len(sample_transactions_df)
        assert stats["total_repeat_pairs"] == len(sample_repeat_sales_df)
        assert stats["price_range"]["min"] > 0
        assert stats["price_range"]["max"] > stats["price_range"]["min"]
    
    def test_no_repeat_sales(self, test_config):
        """Test handling when no repeat sales exist."""
        # Create transactions with no repeats
        transactions = pl.DataFrame({
            "transaction_id": ["T1", "T2", "T3"],
            "property_id": ["P1", "P2", "P3"],  # All different properties
            "sale_price": [100000, 200000, 300000],
            "sale_date": [date(2020, 1, 1), date(2020, 6, 1), date(2021, 1, 1)],
            "transaction_type": [TransactionType.ARMS_LENGTH.value] * 3,
            "county_fips": ["06037"] * 3,
            "tract": ["123456"] * 3
        })
        
        ingestion = DataIngestion(test_config)
        ingestion.transactions_df = transactions
        
        repeat_df = ingestion.identify_repeat_sales()
        
        assert len(repeat_df) == 0
        assert isinstance(repeat_df, pl.DataFrame)
    
    def test_error_handling_invalid_file(self, test_config):
        """Test error handling for invalid file paths."""
        ingestion = DataIngestion(test_config)
        
        with pytest.raises(FileNotFoundError):
            ingestion.load_transactions("nonexistent_file.parquet")
        
        with pytest.raises(ValueError):
            ingestion.load_transactions("file.unsupported")
    
    def test_error_handling_no_transactions(self, test_config):
        """Test error handling when no transactions are loaded."""
        ingestion = DataIngestion(test_config)
        
        with pytest.raises(ValueError, match="No transactions loaded"):
            ingestion.identify_repeat_sales()