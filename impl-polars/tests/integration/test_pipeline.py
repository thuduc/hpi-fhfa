"""Integration tests for the main HPI pipeline."""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
from datetime import date, timedelta
import tempfile
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.hpi_fhfa.processing.pipeline import HPIPipeline, HPIResults
from src.hpi_fhfa.config.settings import HPIConfig


class TestPipeline:
    """Test the complete HPI pipeline integration."""
    
    @pytest.fixture
    def large_synthetic_data(self, temp_dir):
        """Generate larger synthetic dataset for pipeline testing."""
        np.random.seed(123)
        
        # Generate 10,000 transactions across 2,000 properties
        n_transactions = 10000
        n_properties = 2000
        n_tracts = 50
        n_cbsas = 3
        
        # Generate property IDs with some having multiple sales
        property_weights = np.random.exponential(2, n_properties)  # Some properties more likely to sell
        property_weights = property_weights / property_weights.sum()
        property_ids = np.random.choice(
            [f"P{i:06d}" for i in range(n_properties)],
            size=n_transactions,
            p=property_weights
        )
        
        # Generate dates over 5 years (2015-2020)
        start_date = date(2015, 1, 1)
        end_date = date(2020, 12, 31)
        date_range = (end_date - start_date).days
        dates = [start_date + timedelta(days=int(d)) for d in np.random.randint(0, date_range, n_transactions)]
        
        # Generate realistic prices with appreciation trend
        base_price = 250000
        years_from_start = [(d - start_date).days / 365.25 for d in dates]
        annual_appreciation = 0.05  # 5% annual appreciation
        price_trend = [base_price * (1 + annual_appreciation) ** year for year in years_from_start]
        price_noise = np.random.lognormal(0, 0.3, n_transactions)  # 30% noise
        prices = [trend * noise for trend, noise in zip(price_trend, price_noise)]
        
        # Generate geographic data
        tract_ids = [f"06037{i:06d}" for i in range(n_tracts)]
        cbsa_codes = ["31080", "41860", "33100"]  # LA, SF, Miami
        
        # Assign transactions to tracts and CBSAs
        transaction_tracts = np.random.choice(tract_ids, n_transactions)
        transaction_cbsas = np.random.choice(cbsa_codes, n_transactions)
        
        # Create transaction DataFrame
        transactions = pl.DataFrame({
            "property_id": property_ids,
            "transaction_date": dates,
            "transaction_price": prices,
            "census_tract": transaction_tracts,
            "cbsa_code": transaction_cbsas,
            "distance_to_cbd": np.random.uniform(0, 50, n_transactions)
        })
        
        # Create geographic DataFrame
        geographic = pl.DataFrame({
            "tract_id": tract_ids,
            "cbsa_code": np.random.choice(cbsa_codes, n_tracts),
            "centroid_lat": np.random.uniform(32.5, 34.5, n_tracts),
            "centroid_lon": np.random.uniform(-118.5, -116.5, n_tracts),
            "housing_units": np.random.randint(1000, 5000, n_tracts),
            "housing_value": np.random.uniform(500_000_000, 2_000_000_000, n_tracts),
            "college_share": np.random.beta(2, 3, n_tracts),
            "nonwhite_share": np.random.beta(3, 2, n_tracts)
        })
        
        # Save to files
        transaction_path = temp_dir / "large_transactions.parquet"
        geographic_path = temp_dir / "large_geographic.parquet"
        
        transactions.write_parquet(transaction_path)
        geographic.write_parquet(geographic_path)
        
        return transaction_path, geographic_path
    
    @pytest.fixture
    def integration_config(self, large_synthetic_data, temp_dir):
        """Create configuration for integration testing."""
        transaction_path, geographic_path = large_synthetic_data
        
        return HPIConfig(
            transaction_data_path=transaction_path,
            geographic_data_path=geographic_path,
            output_path=temp_dir / "output",
            start_year=2016,
            end_year=2019,
            chunk_size=5000,
            n_jobs=2,
            weight_schemes=["sample", "value", "unit"],
            validate_data=True,
            strict_validation=False,
            checkpoint_frequency=2,
            use_lazy_evaluation=False
        )
    
    def test_full_pipeline_execution(self, integration_config):
        """Test complete pipeline execution."""
        pipeline = HPIPipeline(integration_config)
        
        # Run the pipeline
        results = pipeline.run()
        
        # Verify results structure
        assert isinstance(results, HPIResults)
        assert isinstance(results.tract_indices, pl.DataFrame)
        assert isinstance(results.city_indices, dict)
        assert isinstance(results.metadata, dict)
        
        # Check tract indices
        tract_df = results.tract_indices
        assert len(tract_df) > 0
        assert "tract_id" in tract_df.columns
        assert "year" in tract_df.columns
        assert "index_value" in tract_df.columns
        assert "appreciation_rate" in tract_df.columns
        
        # Should have indices for all years
        years = tract_df["year"].unique().sort().to_list()
        expected_years = list(range(2016, 2020))
        assert years == expected_years
        
        # Check city indices
        assert len(results.city_indices) == 3  # sample, value, unit
        for scheme_name, city_df in results.city_indices.items():
            assert scheme_name in ["sample", "value", "unit"]
            assert len(city_df) > 0
            assert "cbsa_code" in city_df.columns
            assert "year" in city_df.columns
            assert "index_value" in city_df.columns
            assert "weight_scheme" in city_df.columns
        
        # Check metadata
        metadata = results.metadata
        assert metadata["start_year"] == 2016
        assert metadata["end_year"] == 2019
        assert metadata["n_transactions"] > 0
        assert metadata["processing_time"] > 0
    
    def test_pipeline_with_checkpoints(self, integration_config):
        """Test pipeline checkpoint functionality."""
        pipeline = HPIPipeline(integration_config)
        
        # Run pipeline first time
        results1 = pipeline.run()
        
        # Verify checkpoints were created
        checkpoint_dir = integration_config.output_path / "checkpoints"
        assert checkpoint_dir.exists()
        
        checkpoint_files = list(checkpoint_dir.glob("*.pkl"))
        assert len(checkpoint_files) > 0
        
        # Run pipeline again - should use checkpoints
        pipeline2 = HPIPipeline(integration_config)
        results2 = pipeline2.run()
        
        # Results should be identical (or very similar due to checkpointing)
        assert len(results1.tract_indices) == len(results2.tract_indices)
        assert len(results1.city_indices) == len(results2.city_indices)
    
    def test_pipeline_output_files(self, integration_config):
        """Test that pipeline saves output files correctly."""
        pipeline = HPIPipeline(integration_config)
        results = pipeline.run()
        
        output_dir = integration_config.output_path
        
        # Check tract indices file
        tract_file = output_dir / "tract_level_indices.parquet"
        assert tract_file.exists()
        
        # Check city indices files
        for scheme in integration_config.weight_schemes:
            city_file = output_dir / f"city_level_indices_{scheme}.parquet"
            assert city_file.exists()
        
        # Check metadata file
        metadata_file = output_dir / "metadata.json"
        assert metadata_file.exists()
        
        # Verify we can read the files back
        tract_df = pl.read_parquet(tract_file)
        assert len(tract_df) > 0
    
    def test_pipeline_error_handling(self, temp_dir):
        """Test pipeline error handling with invalid data."""
        # Create config with non-existent files should raise ConfigurationError
        with pytest.raises(Exception):  # Should raise ConfigurationError
            HPIConfig(
                transaction_data_path=temp_dir / "nonexistent.parquet",
                geographic_data_path=temp_dir / "nonexistent.parquet",
                output_path=temp_dir / "output",
                start_year=2016,
                end_year=2019,
                use_lazy_evaluation=False
            )
    
    def test_pipeline_with_minimal_data(self, temp_dir):
        """Test pipeline with minimal viable dataset."""
        # Create minimal dataset
        transactions = pl.DataFrame({
            "property_id": ["P001", "P001", "P002", "P002", "P003", "P003"] * 5,
            "transaction_date": [
                date(2016, 1, 1), date(2017, 1, 1),
                date(2016, 6, 1), date(2017, 6, 1),
                date(2016, 12, 1), date(2018, 1, 1)
            ] * 5,
            "transaction_price": [100000, 110000, 200000, 220000, 150000, 165000] * 5,
            "census_tract": ["06037000001", "06037000001", "06037000002", "06037000002", "06037000003", "06037000003"] * 5,
            "cbsa_code": ["31080"] * 30,
            "distance_to_cbd": [5.0] * 30
        })
        
        geographic = pl.DataFrame({
            "tract_id": ["06037000001", "06037000002", "06037000003"],
            "cbsa_code": ["31080", "31080", "31080"],
            "centroid_lat": [34.0, 34.1, 34.2],
            "centroid_lon": [-118.0, -118.1, -118.2],
            "housing_units": [1000, 1500, 1200],
            "housing_value": [1_000_000_000, 1_500_000_000, 1_200_000_000],
            "college_share": [0.5, 0.6, 0.4],
            "nonwhite_share": [0.3, 0.4, 0.5]
        })
        
        # Save files
        transaction_path = temp_dir / "minimal_transactions.parquet"
        geographic_path = temp_dir / "minimal_geographic.parquet"
        
        transactions.write_parquet(transaction_path)
        geographic.write_parquet(geographic_path)
        
        # Create config
        config = HPIConfig(
            transaction_data_path=transaction_path,
            geographic_data_path=geographic_path,
            output_path=temp_dir / "output",
            start_year=2016,
            end_year=2018,
            n_jobs=1,  # Sequential for simpler debugging
            weight_schemes=["sample"],
            validate_data=False,  # Skip validation for minimal data
            use_lazy_evaluation=False
        )
        
        # Run pipeline
        pipeline = HPIPipeline(config)
        results = pipeline.run()
        
        # Should complete successfully
        assert isinstance(results, HPIResults)
        # For minimal data, results may be empty due to insufficient half-pairs
        # but pipeline should still complete without errors
        assert isinstance(results.tract_indices, pl.DataFrame)
        assert isinstance(results.city_indices, dict)
        assert len(results.city_indices) == 1  # Should have 'sample' scheme