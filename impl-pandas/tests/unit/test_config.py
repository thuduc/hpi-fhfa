"""Unit tests for configuration module."""

import pytest
import json
import tempfile
from pathlib import Path

from hpi_fhfa.config import constants, Settings, get_default_settings


class TestConstants:
    """Test configuration constants."""
    
    def test_supertract_constants(self):
        assert constants.MIN_HALF_PAIRS == 40
        
    def test_filter_thresholds(self):
        assert constants.MAX_CAGR == 0.30
        assert constants.MAX_CUMULATIVE_APPRECIATION == 10.0
        assert constants.MIN_CUMULATIVE_APPRECIATION == 0.25
        
    def test_time_constants(self):
        assert constants.BASE_YEAR == 1989
        assert constants.START_YEAR == 1975
        assert constants.END_YEAR == 2021
        assert constants.INDEX_START_YEAR == 1989
        assert constants.INDEX_END_YEAR == 2021
        
    def test_weight_types(self):
        assert len(constants.WEIGHT_TYPES) == 6
        assert "sample" in constants.WEIGHT_TYPES
        assert "value" in constants.WEIGHT_TYPES
        assert "unit" in constants.WEIGHT_TYPES
        assert "upb" in constants.WEIGHT_TYPES
        assert "college" in constants.WEIGHT_TYPES
        assert "nonwhite" in constants.WEIGHT_TYPES
        
    def test_weight_categorization(self):
        assert set(constants.TIME_VARYING_WEIGHTS) == {"sample", "value", "unit", "upb"}
        assert set(constants.STATIC_WEIGHTS) == {"college", "nonwhite"}


class TestSettings:
    """Test Settings configuration class."""
    
    def test_default_settings(self):
        settings = get_default_settings()
        
        assert settings.min_half_pairs == 40
        assert settings.max_cagr == 0.30
        assert settings.chunk_size == 100000
        assert settings.n_jobs == -1
        assert settings.base_year == 1989
        assert settings.start_year == 1989
        assert settings.end_year == 2021
        assert settings.default_weight_type == "sample"
        assert settings.use_sparse_matrices is True
        assert settings.use_numba is True
        assert settings.log_level == "INFO"
        
    def test_custom_settings(self):
        settings = Settings(
            min_half_pairs=50,
            max_cagr=0.25,
            chunk_size=50000,
            n_jobs=4,
            base_year=1990,
            use_sparse_matrices=False
        )
        
        assert settings.min_half_pairs == 50
        assert settings.max_cagr == 0.25
        assert settings.chunk_size == 50000
        assert settings.n_jobs == 4
        assert settings.base_year == 1990
        assert settings.use_sparse_matrices is False
        
    def test_settings_validation_valid(self):
        settings = Settings(
            start_year=1980,
            end_year=2020,
            base_year=2000,
            min_half_pairs=10,
            max_cagr=0.5,
            chunk_size=10000
        )
        
        # Should not raise exception
        settings.validate()
        
    def test_settings_validation_invalid_years(self):
        # Start year before 1975
        settings = Settings(start_year=1970)
        with pytest.raises(ValueError, match="Start year cannot be before 1975"):
            settings.validate()
            
        # End year after 2021
        settings = Settings(end_year=2025)
        with pytest.raises(ValueError, match="End year cannot be after 2021"):
            settings.validate()
            
        # Start year after end year
        settings = Settings(start_year=2020, end_year=2010)
        with pytest.raises(ValueError, match="Start year must be before end year"):
            settings.validate()
            
        # Base year out of range
        settings = Settings(start_year=1990, end_year=2000, base_year=1985)
        with pytest.raises(ValueError, match="Base year must be between start and end years"):
            settings.validate()
            
    def test_settings_validation_invalid_parameters(self):
        # Min half pairs < 1
        settings = Settings(min_half_pairs=0)
        with pytest.raises(ValueError, match="Minimum half-pairs must be at least 1"):
            settings.validate()
            
        # Invalid CAGR
        settings = Settings(max_cagr=0)
        with pytest.raises(ValueError, match="Maximum CAGR must be between 0 and 1"):
            settings.validate()
            
        settings = Settings(max_cagr=1.5)
        with pytest.raises(ValueError, match="Maximum CAGR must be between 0 and 1"):
            settings.validate()
            
        # Chunk size too small
        settings = Settings(chunk_size=500)
        with pytest.raises(ValueError, match="Chunk size must be at least 1000"):
            settings.validate()
            
    def test_settings_from_dict(self):
        config_dict = {
            "min_half_pairs": 30,
            "max_cagr": 0.35,
            "base_year": 1995,
            "transaction_data_path": "/path/to/data.parquet"
        }
        
        settings = Settings.from_dict(config_dict)
        
        assert settings.min_half_pairs == 30
        assert settings.max_cagr == 0.35
        assert settings.base_year == 1995
        assert settings.transaction_data_path == "/path/to/data.parquet"
        
    def test_settings_json_io(self):
        # Create settings
        settings = Settings(
            min_half_pairs=45,
            max_cagr=0.28,
            base_year=1992,
            transaction_data_path="/data/transactions.parquet",
            output_path="/output/results/"
        )
        
        # Save to JSON
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
            
        settings.to_json(temp_path)
        
        # Load from JSON
        loaded_settings = Settings.from_json(temp_path)
        
        # Verify
        assert loaded_settings.min_half_pairs == 45
        assert loaded_settings.max_cagr == 0.28
        assert loaded_settings.base_year == 1992
        assert loaded_settings.transaction_data_path == "/data/transactions.parquet"
        assert loaded_settings.output_path == "/output/results/"
        
        # Cleanup
        Path(temp_path).unlink()
        
    def test_settings_to_json_excludes_none(self):
        settings = Settings(
            min_half_pairs=40,
            transaction_data_path=None,  # This should be excluded
            output_path="/output/"
        )
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
            
        settings.to_json(temp_path)
        
        # Read JSON directly
        with open(temp_path, 'r') as f:
            saved_config = json.load(f)
            
        # Verify None values are excluded
        assert "transaction_data_path" not in saved_config
        assert "output_path" in saved_config
        assert saved_config["output_path"] == "/output/"
        
        # Cleanup
        Path(temp_path).unlink()