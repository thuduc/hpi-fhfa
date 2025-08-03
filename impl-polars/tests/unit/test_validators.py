"""Unit tests for data validators."""

import pytest
import polars as pl
from datetime import date
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.hpi_fhfa.data.validators import DataValidator
from src.hpi_fhfa.utils.exceptions import DataValidationError


class TestDataValidator:
    """Test data validation functionality."""
    
    def test_validate_transactions_valid_data(self, sample_transactions, test_config):
        """Test validation with valid transaction data."""
        validator = DataValidator(test_config)
        result = validator.validate_transactions(sample_transactions)
        
        # Result might have fewer rows if there were duplicates
        assert len(result) <= len(sample_transactions)
        assert result.columns == sample_transactions.columns
        
        # Check that no nulls remain in required columns
        for col in ["property_id", "transaction_date", "transaction_price", "census_tract"]:
            assert result[col].null_count() == 0
    
    def test_validate_transactions_missing_columns(self, test_config):
        """Test validation with missing required columns."""
        # Create DataFrame missing required columns
        invalid_df = pl.DataFrame({
            "property_id": ["P001", "P002"],
            "transaction_price": [100000, 200000]
            # Missing: transaction_date, census_tract, cbsa_code, distance_to_cbd
        })
        
        validator = DataValidator(test_config)
        
        with pytest.raises(DataValidationError, match="Schema validation failed"):
            validator.validate_transactions(invalid_df)
    
    def test_validate_transactions_null_values(self, sample_transactions, test_config):
        """Test handling of null values."""
        # Introduce nulls
        df_with_nulls = sample_transactions.with_columns([
            pl.when(pl.col("property_id") == "P000001")
            .then(None)
            .otherwise(pl.col("property_id"))
            .alias("property_id")
        ])
        
        test_config.strict_validation = False
        validator = DataValidator(test_config)
        
        result = validator.validate_transactions(df_with_nulls)
        # Should remove rows with nulls
        assert len(result) < len(df_with_nulls)
        assert result["property_id"].null_count() == 0
    
    def test_validate_transactions_invalid_prices(self, sample_transactions, test_config):
        """Test handling of invalid prices."""
        # Add invalid prices
        df_invalid = sample_transactions.with_columns([
            pl.when(pl.col("property_id") == "P000001")
            .then(-1000)  # Negative price
            .when(pl.col("property_id") == "P000002")
            .then(150_000_000)  # Too high
            .otherwise(pl.col("transaction_price"))
            .alias("transaction_price")
        ])
        
        test_config.strict_validation = False
        validator = DataValidator(test_config)
        
        result = validator.validate_transactions(df_invalid)
        assert len(result) < len(df_invalid)
        assert result["transaction_price"].min() > 0
        assert result["transaction_price"].max() <= 100_000_000
    
    def test_validate_transactions_invalid_dates(self, sample_transactions, test_config):
        """Test handling of invalid dates."""
        # Add dates outside valid range
        df_invalid = sample_transactions.with_columns([
            pl.when(pl.col("property_id") == "P000001")
            .then(date(1970, 1, 1))  # Too early
            .when(pl.col("property_id") == "P000002")
            .then(date(2025, 1, 1))  # Too late
            .otherwise(pl.col("transaction_date"))
            .alias("transaction_date")
        ])
        
        test_config.strict_validation = False
        validator = DataValidator(test_config)
        
        result = validator.validate_transactions(df_invalid)
        assert len(result) < len(df_invalid)
        assert result["transaction_date"].min() >= date(1975, 1, 1)
        assert result["transaction_date"].max() <= date(2021, 12, 31)
    
    def test_validate_transactions_duplicates(self, test_config):
        """Test handling of duplicate transactions."""
        # Create data with duplicates
        df_with_dups = pl.DataFrame({
            "property_id": ["P001", "P001", "P002"],
            "transaction_date": [date(2020, 1, 1), date(2020, 1, 1), date(2020, 2, 1)],
            "transaction_price": [100000, 110000, 200000],
            "census_tract": ["06037000100", "06037000100", "06037000200"],
            "cbsa_code": ["31080", "31080", "31080"],
            "distance_to_cbd": [5.0, 5.0, 10.0]
        })
        
        test_config.strict_validation = False
        validator = DataValidator(test_config)
        
        result = validator.validate_transactions(df_with_dups)
        # Should keep only first duplicate
        assert len(result) == 2
        assert result.filter(
            (pl.col("property_id") == "P001") & 
            (pl.col("transaction_date") == date(2020, 1, 1))
        )["transaction_price"][0] == 100000
    
    def test_validate_transactions_strict_mode(self, sample_transactions, test_config):
        """Test strict validation mode."""
        # Add invalid data
        df_invalid = sample_transactions.with_columns([
            pl.when(pl.col("property_id") == "P000001")
            .then(None)
            .otherwise(pl.col("property_id"))
            .alias("property_id")
        ])
        
        test_config.strict_validation = True
        validator = DataValidator(test_config)
        
        with pytest.raises(DataValidationError, match="Validation failed"):
            validator.validate_transactions(df_invalid)
    
    def test_validate_geographic_data_valid(self, sample_geographic_data, test_config):
        """Test validation with valid geographic data."""
        validator = DataValidator(test_config)
        result = validator.validate_geographic_data(sample_geographic_data)
        
        assert len(result) == len(sample_geographic_data)
        assert result.columns == sample_geographic_data.columns
    
    def test_validate_geographic_data_invalid_coordinates(self, sample_geographic_data, test_config):
        """Test handling of invalid coordinates."""
        # Add invalid coordinates
        df_invalid = sample_geographic_data.with_columns([
            pl.when(pl.col("tract_id") == "06037000000")
            .then(95.0)  # Invalid latitude
            .otherwise(pl.col("centroid_lat"))
            .alias("centroid_lat")
        ])
        
        validator = DataValidator(test_config)
        
        with pytest.raises(DataValidationError, match="invalid coordinates"):
            validator.validate_geographic_data(df_invalid)
    
    def test_validate_geographic_data_invalid_shares(self, sample_geographic_data, test_config):
        """Test handling of invalid share values."""
        # Add invalid shares
        df_invalid = sample_geographic_data.with_columns([
            pl.when(pl.col("tract_id") == "06037000000")
            .then(1.5)  # > 1.0
            .otherwise(pl.col("college_share"))
            .alias("college_share")
        ])
        
        validator = DataValidator(test_config)
        
        with pytest.raises(DataValidationError, match="invalid share values"):
            validator.validate_geographic_data(df_invalid)
    
    def test_validate_geographic_data_duplicate_tracts(self, sample_geographic_data, test_config):
        """Test handling of duplicate tract IDs."""
        # Add duplicate tract
        df_with_dup = pl.concat([
            sample_geographic_data,
            sample_geographic_data.head(1)
        ])
        
        validator = DataValidator(test_config)
        
        with pytest.raises(DataValidationError, match="duplicate tract IDs"):
            validator.validate_geographic_data(df_with_dup)