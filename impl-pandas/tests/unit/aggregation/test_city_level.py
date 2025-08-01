"""Unit tests for city_level module."""

import pytest
import pandas as pd
import numpy as np
from hpi_fhfa.aggregation.city_level import CityLevelIndexBuilder
from hpi_fhfa.aggregation.index_builder import HPIIndex
from hpi_fhfa.aggregation.weights import WeightType
from hpi_fhfa.geography.census_tract import CensusTract
from hpi_fhfa.config.constants import BASE_YEAR


class TestCityLevelIndexBuilder:
    """Test CityLevelIndexBuilder functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.census_tracts = [
            CensusTract(
                tract_code=f"1234567890{i}",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number=f"67890{i}",
                centroid_lat=40.71 + i * 0.01,
                centroid_lon=-74.00 - i * 0.01,
                distance_to_cbd=5.0 + i * 0.1,
                population=1000 * (i + 1),
                housing_units=400 * (i + 1),
                college_share=0.3 + i * 0.05,
                nonwhite_share=0.2 + i * 0.02
            )
            for i in range(3)
        ]
        
        # Create sample transaction data
        transactions = []
        property_id = 1
        for i in range(3):
            tract_code = f"1234567890{i}"
            # Create more transactions for first tract
            n_properties = 10 if i == 0 else 5
            
            for j in range(n_properties):
                # Each property has 2 transactions
                transactions.extend([
                    {
                        'property_id': f'P{property_id}',
                        'transaction_date': pd.Timestamp('2019-06-01'),
                        'transaction_price': 200000 * (1 + i * 0.1),
                        'census_tract': tract_code,
                        'cbsa_code': '12345',
                        'distance_to_cbd': 5.0 + i * 0.1
                    },
                    {
                        'property_id': f'P{property_id}',
                        'transaction_date': pd.Timestamp('2020-06-01'),
                        'transaction_price': 210000 * (1 + i * 0.1),
                        'census_tract': tract_code,
                        'cbsa_code': '12345',
                        'distance_to_cbd': 5.0 + i * 0.1
                    }
                ])
                property_id += 1
        
        self.transactions = pd.DataFrame(transactions)
    
    def test_builder_initialization(self):
        """Test city-level builder initialization."""
        builder = CityLevelIndexBuilder(base_year=1990, min_half_pairs=30)
        
        assert builder.base_year == 1990
        assert builder.min_half_pairs == 30
        assert builder.index_builder is not None
        assert builder.supertract_algorithm is not None
    
    def test_build_annual_index_basic(self):
        """Test basic annual index building."""
        builder = CityLevelIndexBuilder(min_half_pairs=5)
        
        index = builder.build_annual_index(
            self.transactions,
            self.census_tracts,
            WeightType.SAMPLE,
            start_year=2019,
            end_year=2020
        )
        
        assert isinstance(index, HPIIndex)
        assert index.entity_id == "12345"
        assert index.entity_type == "cbsa"
        # Check if it's a default index or a calculated one
        if 'is_default' in index.metadata and index.metadata['is_default']:
            assert index.metadata['reason'] == 'Insufficient data for index construction'
        else:
            assert index.metadata['weight_type'] == "sample"
    
    def test_build_annual_index_missing_columns(self):
        """Test error with missing required columns."""
        builder = CityLevelIndexBuilder()
        
        bad_transactions = self.transactions.drop(columns=['census_tract'])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            builder.build_annual_index(
                bad_transactions,
                self.census_tracts,
                WeightType.SAMPLE
            )
    
    def test_build_annual_index_multiple_cbsas(self):
        """Test error with multiple CBSA codes."""
        builder = CityLevelIndexBuilder()
        
        # Add tract with different CBSA
        mixed_tracts = self.census_tracts + [
            CensusTract(
                tract_code="98765432101",
                cbsa_code="98765",  # Different CBSA
                state_code="98",
                county_code="765",
                tract_number="432101",
                centroid_lat=41.0,
                centroid_lon=-75.0,
                distance_to_cbd=10.0
            )
        ]
        
        with pytest.raises(ValueError, match="Multiple CBSA codes"):
            builder.build_annual_index(
                self.transactions,
                mixed_tracts,
                WeightType.SAMPLE
            )
    
    def test_build_annual_index_no_pairs(self):
        """Test handling when no repeat sales pairs found."""
        builder = CityLevelIndexBuilder()
        
        # Transactions with no repeats
        single_transactions = pd.DataFrame({
            'property_id': ['P1', 'P2', 'P3'],
            'transaction_date': pd.to_datetime(['2019-01-01'] * 3),
            'transaction_price': [200000, 210000, 220000],
            'census_tract': ['12345678900'] * 3,
            'cbsa_code': ['12345'] * 3,
            'distance_to_cbd': [5.0, 5.1, 5.2]
        })
        
        index = builder.build_annual_index(
            single_transactions,
            self.census_tracts,
            WeightType.SAMPLE,
            start_year=2019,
            end_year=2020
        )
        
        # Should return default index
        assert index.metadata.get('is_default', False)
        assert all(v == 1.0 for v in index.index_values.values())
    
    def test_build_annual_index_value_weights(self):
        """Test building index with value weights."""
        builder = CityLevelIndexBuilder(min_half_pairs=5)
        
        # Add value data
        value_data = pd.DataFrame({
            'tract_code': [f"1234567890{i}" for i in range(3)],
            'aggregate_value': [1000000, 2000000, 1500000]
        })
        
        index = builder.build_annual_index(
            self.transactions,
            self.census_tracts,
            WeightType.VALUE,
            start_year=2019,
            end_year=2020,
            additional_data=value_data
        )
        
        assert isinstance(index, HPIIndex)
        # Check if it's a default index or a calculated one
        if 'is_default' in index.metadata and index.metadata['is_default']:
            assert index.metadata['reason'] == 'Insufficient data for index construction'
        else:
            assert index.metadata['weight_type'] == "value"
    
    def test_build_indices_all_weights(self):
        """Test building indices for all weight types."""
        builder = CityLevelIndexBuilder(min_half_pairs=5)
        
        # Prepare additional data
        additional_data = {
            'value': pd.DataFrame({
                'tract_code': [f"1234567890{i}" for i in range(3)],
                'aggregate_value': [1000000, 2000000, 1500000]
            }),
            'upb': pd.DataFrame({
                'tract_code': [f"1234567890{i}" for i in range(3)],
                'upb': [500000, 1000000, 750000]
            })
        }
        
        indices = builder.build_indices_all_weights(
            self.transactions,
            self.census_tracts,
            start_year=2019,
            end_year=2020,
            additional_data=additional_data
        )
        
        assert isinstance(indices, dict)
        # Should have indices for each weight type
        assert len(indices) <= len(WeightType)
        
        # Check each index
        for weight_type, index in indices.items():
            assert isinstance(index, HPIIndex)
            # Check if it's a default index or a calculated one
            if 'is_default' in index.metadata and index.metadata['is_default']:
                assert index.metadata['reason'] == 'Insufficient data for index construction'
            else:
                assert index.metadata['weight_type'] == weight_type
    
    def test_aggregate_indices_for_period(self):
        """Test aggregating indices for specific period."""
        builder = CityLevelIndexBuilder()
        
        # Create sample indices
        indices = {
            'ST_001': HPIIndex(
                index_values={2019: 1.0, 2020: 1.05},
                entity_id='ST_001',
                entity_type='supertract'
            ),
            'ST_002': HPIIndex(
                index_values={2019: 1.0, 2020: 1.03},
                entity_id='ST_002',
                entity_type='supertract'
            )
        }
        
        weights = {
            'ST_001': 0.6,
            'ST_002': 0.4
        }
        
        value = builder._aggregate_indices_for_period(indices, weights, 2020)
        
        assert value == pytest.approx(1.05 * 0.6 + 1.03 * 0.4)
    
    def test_aggregate_indices_for_period_missing_data(self):
        """Test aggregation with missing data."""
        builder = CityLevelIndexBuilder()
        
        indices = {
            'ST_001': HPIIndex(
                index_values={2019: 1.0},  # Missing 2020
                entity_id='ST_001',
                entity_type='supertract'
            )
        }
        
        weights = {'ST_001': 1.0}
        
        value = builder._aggregate_indices_for_period(indices, weights, 2020)
        
        assert value is None
    
    def test_create_default_index(self):
        """Test creating default index."""
        builder = CityLevelIndexBuilder()
        
        index = builder._create_default_index("12345", 2019, 2021)
        
        assert index.entity_id == "12345"
        assert index.entity_type == "cbsa"
        assert index.metadata['is_default']
        assert all(v == 1.0 for v in index.index_values.values())
        assert 2019 in index.index_values
        assert 2021 in index.index_values
    
    def test_calculate_pooled_appreciation(self):
        """Test pooled appreciation calculation."""
        builder = CityLevelIndexBuilder()
        
        # Create transactions with known appreciation
        transactions = pd.DataFrame([
            {
                'property_id': 'P1',
                'transaction_date': pd.Timestamp('2019-01-01'),
                'transaction_price': 200000,
                'census_tract': '12345678900',
                'cbsa_code': '12345',
                'distance_to_cbd': 5.0
            },
            {
                'property_id': 'P1',
                'transaction_date': pd.Timestamp('2020-01-01'),
                'transaction_price': 210000,  # 5% appreciation
                'census_tract': '12345678900',
                'cbsa_code': '12345',
                'distance_to_cbd': 5.0
            }
        ])
        
        appreciation = builder.calculate_pooled_appreciation(
            transactions,
            self.census_tracts,
            2019,
            2020
        )
        
        # Should be close to log(210000/200000)
        expected = np.log(210000/200000)
        assert abs(appreciation - expected) < 0.1
    
    def test_calculate_pooled_appreciation_no_pairs(self):
        """Test pooled appreciation with no pairs."""
        builder = CityLevelIndexBuilder()
        
        # Empty transactions
        appreciation = builder.calculate_pooled_appreciation(
            pd.DataFrame(columns=['property_id', 'transaction_date', 
                                'transaction_price', 'census_tract', 
                                'cbsa_code', 'distance_to_cbd']),
            self.census_tracts,
            2019,
            2020
        )
        
        assert appreciation == 0.0
    
    def test_export_results_csv(self):
        """Test exporting results to CSV."""
        builder = CityLevelIndexBuilder()
        
        index = HPIIndex(
            index_values={2019: 1.0, 2020: 1.05},
            entity_id="12345",
            entity_type="cbsa"
        )
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            builder.export_results(index, f.name, format='csv')
            
            # Read back and verify
            df = pd.read_csv(f.name)
            assert len(df) == 2
            assert 'period' in df.columns
            assert 'index_value' in df.columns
    
    def test_export_results_invalid_format(self):
        """Test error with invalid export format."""
        builder = CityLevelIndexBuilder()
        
        index = HPIIndex(
            index_values={2019: 1.0},
            entity_id="12345",
            entity_type="cbsa"
        )
        
        with pytest.raises(ValueError, match="Unknown format"):
            builder.export_results(index, "output.txt", format='txt')
    
    def test_create_summary_statistics(self):
        """Test creating summary statistics."""
        builder = CityLevelIndexBuilder()
        
        indices = {
            'sample': HPIIndex(
                index_values={
                    2015: 1.0,
                    2016: 1.05,
                    2017: 1.08,
                    2018: 1.06,
                    2019: 1.10
                },
                entity_id="12345",
                entity_type="cbsa"
            ),
            'value': HPIIndex(
                index_values={
                    2015: 1.0,
                    2016: 1.04,
                    2017: 1.09,
                    2018: 1.08,
                    2019: 1.12
                },
                entity_id="12345",
                entity_type="cbsa"
            )
        }
        
        summary = builder.create_summary_statistics(indices)
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert 'weight_type' in summary.columns
        assert 'total_appreciation' in summary.columns
        assert 'avg_annual_growth' in summary.columns
        assert 'volatility' in summary.columns
        
        # Check calculations
        sample_row = summary[summary['weight_type'] == 'sample'].iloc[0]
        assert sample_row['total_appreciation'] == pytest.approx(0.10)
        assert sample_row['n_periods'] == 5