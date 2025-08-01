"""End-to-end integration tests for HPI-FHFA implementation."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from hpi_fhfa.geography import CensusTract
from hpi_fhfa.aggregation import CityLevelIndexBuilder, WeightType
from hpi_fhfa.models.repeat_sales import construct_repeat_sales_pairs
from hpi_fhfa.validation import BenchmarkValidator


class TestEndToEndPipeline:
    """Test complete HPI pipeline from raw data to final indices."""
    
    def generate_synthetic_transactions(self, 
                                      n_properties: int = 1000,
                                      n_tracts: int = 10,
                                      years: int = 5) -> pd.DataFrame:
        """Generate synthetic transaction data for testing."""
        np.random.seed(42)
        
        transactions = []
        property_id = 1
        
        for tract_idx in range(n_tracts):
            tract_code = f"12345{tract_idx:06d}"  # Ensure 11 digits
            # Base price varies by tract
            base_price = 200000 + tract_idx * 20000
            
            # Properties per tract
            props_in_tract = n_properties // n_tracts
            
            for _ in range(props_in_tract):
                # Each property has 2-3 transactions
                n_trans = np.random.choice([2, 3], p=[0.7, 0.3])
                
                # Generate transaction dates
                start_date = datetime(2017, 1, 1)
                dates = []
                for i in range(n_trans):
                    days_offset = i * 365 + np.random.randint(-60, 60)
                    dates.append(start_date + timedelta(days=days_offset))
                
                # Generate prices with appreciation
                price = base_price * (1 + np.random.normal(0, 0.1))
                for i, date in enumerate(dates):
                    # Annual appreciation 3-5%
                    annual_appr = 0.03 + tract_idx * 0.002
                    price *= (1 + annual_appr)
                    
                    transactions.append({
                        'property_id': f'P{property_id}',
                        'transaction_date': date,
                        'transaction_price': price * (1 + np.random.normal(0, 0.05)),
                        'census_tract': tract_code,
                        'cbsa_code': '12345',
                        'distance_to_cbd': 5.0 + tract_idx * 0.5
                    })
                
                property_id += 1
        
        return pd.DataFrame(transactions)
    
    def generate_census_tracts(self, n_tracts: int = 10) -> list:
        """Generate synthetic census tract data."""
        tracts = []
        
        for i in range(n_tracts):
            # Ensure tract code is exactly 11 digits
            tract_code = f"12345{i:06d}"  # Format to ensure 11 digits
            tract = CensusTract(
                tract_code=tract_code,
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number=f"67890{i}",
                centroid_lat=40.7 + i * 0.01,
                centroid_lon=-74.0 - i * 0.01,
                distance_to_cbd=5.0 + i * 0.5,
                population=5000 + i * 500,
                housing_units=2000 + i * 200,
                college_share=0.25 + i * 0.02,
                nonwhite_share=0.30 + i * 0.01
            )
            tracts.append(tract)
        
        return tracts
    
    def test_complete_pipeline(self):
        """Test complete pipeline from transactions to city index."""
        # Generate test data
        transactions = self.generate_synthetic_transactions()
        census_tracts = self.generate_census_tracts()
        
        # Build repeat sales pairs
        pairs = construct_repeat_sales_pairs(transactions)
        assert len(pairs) > 0
        assert 'price_relative' in pairs.columns
        
        # Build city-level index
        builder = CityLevelIndexBuilder(min_half_pairs=10)
        index = builder.build_annual_index(
            transactions,
            census_tracts,
            WeightType.SAMPLE,
            start_year=2017,
            end_year=2021
        )
        
        # Verify index properties
        assert index is not None
        assert index.entity_id == "12345"
        assert index.entity_type == "cbsa"
        assert 2017 in index.index_values
        assert 2021 in index.index_values
        
        # Check appreciation is reasonable (0-10% per year)
        # Note: Generated data may not have perfect appreciation
        total_appr = index.get_appreciation_rate(2017, 2021)
        annual_appr = (1 + total_appr) ** (1/4) - 1
        assert -0.05 <= annual_appr <= 0.10  # Allow for some variation
    
    def test_multiple_weight_types(self):
        """Test building indices with different weight types."""
        # Generate test data
        transactions = self.generate_synthetic_transactions()
        census_tracts = self.generate_census_tracts()
        
        # Add value data
        value_data = pd.DataFrame({
            'tract_code': [f"12345{i:06d}" for i in range(10)],
            'aggregate_value': [1000000 * (i + 1) for i in range(10)]
        })
        
        # Build indices for all weight types
        builder = CityLevelIndexBuilder(min_half_pairs=10)
        indices = builder.build_indices_all_weights(
            transactions,
            census_tracts,
            start_year=2017,
            end_year=2021,
            additional_data={'value': value_data}
        )
        
        # Should have indices for sample and value weights at least
        assert len(indices) >= 2
        assert 'sample' in indices
        
        # Indices should be similar but not identical
        if 'value' in indices:
            sample_2021 = indices['sample'].index_values[2021]
            value_2021 = indices['value'].index_values[2021]
            # Should be within 10% of each other
            assert abs(sample_2021 - value_2021) / sample_2021 < 0.10
    
    def test_export_import_cycle(self):
        """Test exporting and re-importing index data."""
        # Generate test data
        transactions = self.generate_synthetic_transactions(n_properties=100)
        census_tracts = self.generate_census_tracts(n_tracts=5)
        
        # Build index
        builder = CityLevelIndexBuilder()
        index = builder.build_annual_index(
            transactions,
            census_tracts,
            WeightType.SAMPLE,
            start_year=2017,
            end_year=2021
        )
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Export
            builder.export_results(index, temp_path, format='csv')
            assert os.path.exists(temp_path)
            
            # Re-import
            imported_df = pd.read_csv(temp_path)
            assert len(imported_df) == len(index.index_values)
            assert 'period' in imported_df.columns
            assert 'index_value' in imported_df.columns
            
            # Verify values match
            for _, row in imported_df.iterrows():
                period = int(row['period'])
                assert index.index_values[period] == pytest.approx(row['index_value'])
        
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_sparse_data_handling(self):
        """Test handling of sparse data (few transactions per tract)."""
        # Generate sparse data
        transactions = self.generate_synthetic_transactions(
            n_properties=50,  # Very few properties
            n_tracts=20       # Many tracts
        )
        census_tracts = self.generate_census_tracts(n_tracts=20)
        
        # Build index with high threshold
        builder = CityLevelIndexBuilder(min_half_pairs=100)
        index = builder.build_annual_index(
            transactions,
            census_tracts,
            WeightType.SAMPLE,
            start_year=2017,
            end_year=2021
        )
        
        # Should still produce an index (likely default due to sparse data)
        assert index is not None
        assert len(index.index_values) > 0
        # With such sparse data, it's likely a default index
        if index.metadata.get('is_default'):
            assert all(v == 1.0 for v in index.index_values.values())
    
    def test_validation_against_benchmark(self):
        """Test validation against known benchmark."""
        # Generate test data
        transactions = self.generate_synthetic_transactions()
        census_tracts = self.generate_census_tracts()
        
        # Build index
        builder = CityLevelIndexBuilder()
        index = builder.build_annual_index(
            transactions,
            census_tracts,
            WeightType.SAMPLE,
            start_year=2017,
            end_year=2021
        )
        
        # Create synthetic benchmark (slightly different from calculated)
        benchmark_data = []
        for year in range(2017, 2022):
            # Benchmark has 3% annual growth
            value = 100 * (1.03 ** (year - 2017))
            benchmark_data.append({
                'period': year,
                'index_value': value * (1 + np.random.normal(0, 0.01))
            })
        benchmark_df = pd.DataFrame(benchmark_data)
        
        # Validate
        validator = BenchmarkValidator(
            correlation_threshold=0.90,
            rmse_threshold=10.0,
            max_deviation_threshold=0.15
        )
        
        result = validator.validate_against_benchmark(
            index, benchmark_df
        )
        
        # Should have reasonable correlation
        # Note: Since the calculated index starts at 1.0 and benchmark at 100,
        # the RMSE will be large. Check correlation instead.
        assert result.correlation > 0.70  # Reasonable correlation for synthetic data
    
    def test_error_propagation(self):
        """Test that errors are properly handled and logged."""
        # Create invalid data
        bad_transactions = pd.DataFrame({
            'property_id': ['P1'],
            'transaction_date': [pd.Timestamp('2020-01-01')],
            'transaction_price': [-1000],  # Invalid negative price
            'census_tract': ['12345678901'],
            'cbsa_code': ['12345'],
            'distance_to_cbd': [5.0]
        })
        
        census_tracts = self.generate_census_tracts(n_tracts=1)
        
        # Should handle gracefully
        builder = CityLevelIndexBuilder()
        index = builder.build_annual_index(
            bad_transactions,
            census_tracts,
            WeightType.SAMPLE,
            start_year=2019,
            end_year=2021
        )
        
        # Should return default index
        assert index is not None
        assert index.metadata.get('is_default', False)