"""Unit tests for index_builder module."""

import pytest
import pandas as pd
import numpy as np
from hpi_fhfa.aggregation.index_builder import IndexBuilder, HPIIndex
from hpi_fhfa.geography.census_tract import CensusTract
from hpi_fhfa.geography.supertract import Supertract
from hpi_fhfa.models.bmn_regression import BMNResults
from hpi_fhfa.config.constants import BASE_YEAR


class TestHPIIndex:
    """Test HPIIndex class functionality."""
    
    def test_valid_index_creation(self):
        """Test creating a valid HPI index."""
        index_values = {
            1989: 1.0,
            1990: 1.05,
            1991: 1.08,
            1992: 1.10
        }
        
        index = HPIIndex(
            index_values=index_values,
            entity_id="12345",
            entity_type="tract",
            base_period=1989
        )
        
        assert index.entity_id == "12345"
        assert index.entity_type == "tract"
        assert index.base_period == 1989
        assert index.index_values[1989] == 1.0
    
    def test_empty_index_error(self):
        """Test error with empty index values."""
        with pytest.raises(ValueError, match="at least one value"):
            HPIIndex(
                index_values={},
                entity_id="12345",
                entity_type="tract"
            )
    
    def test_normalize_to_base(self):
        """Test index normalization."""
        index_values = {
            1989: 2.0,  # Not normalized
            1990: 2.1,
            1991: 2.2
        }
        
        index = HPIIndex(
            index_values=index_values,
            entity_id="12345",
            entity_type="tract",
            base_period=1989
        )
        
        # Should be automatically normalized
        assert index.index_values[1989] == pytest.approx(1.0)
        assert index.index_values[1990] == pytest.approx(1.05)
        assert index.index_values[1991] == pytest.approx(1.1)
    
    def test_normalize_to_new_base(self):
        """Test normalizing to different base period."""
        index_values = {
            1989: 1.0,
            1990: 1.05,
            1991: 1.10
        }
        
        index = HPIIndex(
            index_values=index_values,
            entity_id="12345",
            entity_type="tract",
            base_period=1989
        )
        
        index.normalize_to_base(1990)
        
        assert index.base_period == 1990
        assert index.index_values[1990] == pytest.approx(1.0)
        assert index.index_values[1989] == pytest.approx(1/1.05)
        assert index.index_values[1991] == pytest.approx(1.10/1.05)
    
    def test_normalize_missing_base_error(self):
        """Test error when normalizing to missing base."""
        index = HPIIndex(
            index_values={1989: 1.0, 1990: 1.05},
            entity_id="12345",
            entity_type="tract"
        )
        
        with pytest.raises(ValueError, match="not in index"):
            index.normalize_to_base(1991)
    
    def test_get_appreciation_rate(self):
        """Test appreciation rate calculation."""
        index = HPIIndex(
            index_values={
                1989: 1.0,
                1990: 1.05,
                1991: 1.10
            },
            entity_id="12345",
            entity_type="tract"
        )
        
        rate = index.get_appreciation_rate(1989, 1990)
        assert rate == pytest.approx(0.05)
        
        rate = index.get_appreciation_rate(1989, 1991)
        assert rate == pytest.approx(0.10)
    
    def test_get_appreciation_rate_missing_period(self):
        """Test error with missing periods."""
        index = HPIIndex(
            index_values={1989: 1.0, 1990: 1.05},
            entity_id="12345",
            entity_type="tract"
        )
        
        with pytest.raises(ValueError, match="not in index"):
            index.get_appreciation_rate(1989, 1991)
    
    def test_get_cagr(self):
        """Test CAGR calculation."""
        index = HPIIndex(
            index_values={
                1989: 1.0,
                1990: 1.05,
                1991: 1.1025,  # 5% annual growth
                1992: 1.1576
            },
            entity_id="12345",
            entity_type="tract"
        )
        
        cagr = index.get_cagr(1989, 1992)
        assert cagr == pytest.approx(0.05, rel=0.01)
    
    def test_get_cagr_invalid_periods(self):
        """Test CAGR with invalid period order."""
        index = HPIIndex(
            index_values={1989: 1.0, 1990: 1.05},
            entity_id="12345",
            entity_type="tract"
        )
        
        with pytest.raises(ValueError, match="after period1"):
            index.get_cagr(1990, 1989)
    
    def test_to_series(self):
        """Test conversion to pandas Series."""
        index = HPIIndex(
            index_values={1990: 1.05, 1989: 1.0, 1991: 1.10},
            entity_id="12345",
            entity_type="tract"
        )
        
        series = index.to_series()
        
        assert isinstance(series, pd.Series)
        assert series.name == "12345"
        assert len(series) == 3
        # Should be sorted by period
        assert list(series.index) == [1989, 1990, 1991]
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        index = HPIIndex(
            index_values={1989: 1.0, 1990: 1.05},
            entity_id="12345",
            entity_type="tract",
            metadata={'cbsa': '98765', 'n_pairs': 100}
        )
        
        df = index.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'period' in df.columns
        assert 'index_value' in df.columns
        assert 'entity_id' in df.columns
        assert 'entity_type' in df.columns
        assert 'cbsa' in df.columns
        assert 'n_pairs' in df.columns
        assert df['cbsa'].iloc[0] == '98765'


class TestIndexBuilder:
    """Test IndexBuilder functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.tracts = [
            CensusTract(
                tract_code="12345678901",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678901",
                centroid_lat=40.7128,
                centroid_lon=-74.0060,
                distance_to_cbd=5.5
            )
        ]
        
        self.supertract = Supertract(
            supertract_id="ST_001",
            component_tracts=self.tracts,
            period=2020,
            half_pairs_count=50
        )
        
        # Create sample repeat sales data
        self.repeat_sales_pairs = pd.DataFrame({
            'property_id': [f'P{i}' for i in range(20)],
            'census_tract': ['12345678901'] * 20,
            'period_1': [2018, 2019] * 10,
            'period_2': [2019, 2020] * 10,
            'sale1_period': [0, 1] * 10,  # Period indices for BMN regression
            'sale2_period': [1, 2] * 10,  # Period indices for BMN regression
            'price_relative': np.random.normal(0.05, 0.02, 20),
            'days_diff': [365] * 20,
            'period_1_dummies': [[1, 0, 0]] * 10 + [[0, 1, 0]] * 10,
            'period_2_dummies': [[0, 1, 0]] * 10 + [[0, 0, 1]] * 10
        })
    
    def test_builder_initialization(self):
        """Test index builder initialization."""
        builder = IndexBuilder(base_year=1990)
        
        assert builder.base_year == 1990
        assert builder.bmn_regressor is not None
    
    def test_build_index_for_entity_success(self):
        """Test successful index building for entity."""
        builder = IndexBuilder()
        
        index = builder.build_index_for_entity(
            self.supertract,
            self.repeat_sales_pairs,
            start_period=2018,
            end_period=2020
        )
        
        assert index is not None
        assert isinstance(index, HPIIndex)
        assert index.entity_id == "ST_001"
        assert index.entity_type == "supertract"
    
    def test_build_index_for_entity_no_data(self):
        """Test index building with no data."""
        builder = IndexBuilder()
        empty_pairs = pd.DataFrame(columns=self.repeat_sales_pairs.columns)
        
        index = builder.build_index_for_entity(
            self.supertract,
            empty_pairs,
            start_period=2018,
            end_period=2020
        )
        
        assert index is None
    
    def test_build_index_for_entity_insufficient_periods(self):
        """Test index building with insufficient periods."""
        builder = IndexBuilder()
        
        # Only one period
        single_period_pairs = self.repeat_sales_pairs[
            self.repeat_sales_pairs['period_1'] == 2018
        ].copy()
        
        index = builder.build_index_for_entity(
            self.supertract,
            single_period_pairs,
            start_period=2018,
            end_period=2020
        )
        
        assert index is None
    
    def test_build_indices_for_cbsa(self):
        """Test building indices for multiple supertracts."""
        builder = IndexBuilder()
        
        # Create multiple supertracts
        supertracts = [
            Supertract(
                supertract_id=f"ST_00{i}",
                component_tracts=[self.tracts[0]],
                period=2020,
                half_pairs_count=50
            )
            for i in range(3)
        ]
        
        indices = builder.build_indices_for_cbsa(
            supertracts,
            self.repeat_sales_pairs,
            start_period=2018,
            end_period=2020
        )
        
        assert isinstance(indices, dict)
        assert len(indices) <= 3  # May have fewer if some fail
    
    def test_filter_pairs_for_entity(self):
        """Test filtering pairs for specific entity."""
        builder = IndexBuilder()
        
        # Add some pairs from different tracts
        mixed_pairs = pd.concat([
            self.repeat_sales_pairs,
            self.repeat_sales_pairs.copy().assign(census_tract='98765432101')
        ])
        
        filtered = builder._filter_pairs_for_entity(
            self.supertract,
            mixed_pairs
        )
        
        assert len(filtered) == 20  # Only original tract
        assert all(filtered['census_tract'] == '12345678901')
    
    def test_calculate_chained_index(self):
        """Test chained index calculation."""
        builder = IndexBuilder()
        
        # Period-to-period appreciation rates
        period_indices = [
            (1989, 1990, 0.05),
            (1990, 1991, 0.03),
            (1991, 1992, 0.04)
        ]
        
        chained = builder.calculate_chained_index(period_indices, 1989)
        
        assert chained[1989] == 1.0
        assert chained[1990] == pytest.approx(1.05)
        assert chained[1991] == pytest.approx(1.05 * 1.03)
        assert chained[1992] == pytest.approx(1.05 * 1.03 * 1.04)
    
    def test_calculate_chained_index_backward(self):
        """Test backward calculation in chained index."""
        builder = IndexBuilder()
        
        # Start from middle
        period_indices = [
            (1989, 1990, 0.05),
            (1990, 1991, 0.03)
        ]
        
        chained = builder.calculate_chained_index(period_indices, 1990)
        
        assert chained[1990] == 1.0
        assert chained[1989] == pytest.approx(1/1.05)
        assert chained[1991] == pytest.approx(1.03)
    
    def test_merge_indices_single(self):
        """Test merging single index."""
        builder = IndexBuilder()
        
        index = HPIIndex(
            index_values={1989: 1.0, 1990: 1.05},
            entity_id="12345",
            entity_type="tract"
        )
        
        merged = builder.merge_indices([index])
        assert merged == index
    
    def test_merge_indices_multiple(self):
        """Test merging multiple indices."""
        builder = IndexBuilder()
        
        index1 = HPIIndex(
            index_values={1989: 1.0, 1990: 1.05, 1991: 1.08},
            entity_id="12345",
            entity_type="tract"
        )
        
        index2 = HPIIndex(
            index_values={1989: 1.0, 1990: 1.04, 1991: 1.09},
            entity_id="12346",
            entity_type="tract"
        )
        
        merged = builder.merge_indices([index1, index2])
        
        assert merged.entity_type == "merged"
        assert merged.index_values[1990] == pytest.approx((1.05 + 1.04) / 2)
        assert merged.metadata['n_sources'] == 2
    
    def test_merge_indices_with_weights(self):
        """Test merging indices with custom weights."""
        builder = IndexBuilder()
        
        index1 = HPIIndex(
            index_values={1989: 1.0, 1990: 1.10},
            entity_id="12345",
            entity_type="tract"
        )
        
        index2 = HPIIndex(
            index_values={1989: 1.0, 1990: 1.05},
            entity_id="12346",
            entity_type="tract"
        )
        
        # 75% weight on first index
        merged = builder.merge_indices([index1, index2], weights=[0.75, 0.25])
        
        assert merged.index_values[1990] == pytest.approx(1.10 * 0.75 + 1.05 * 0.25)
    
    def test_merge_indices_missing_periods(self):
        """Test merging indices with different periods."""
        builder = IndexBuilder()
        
        index1 = HPIIndex(
            index_values={1989: 1.0, 1990: 1.05},
            entity_id="12345",
            entity_type="tract"
        )
        
        index2 = HPIIndex(
            index_values={1990: 1.0, 1991: 1.04},
            entity_id="12346",
            entity_type="tract",
            base_period=1990
        )
        
        merged = builder.merge_indices([index1, index2])
        
        # Should have all periods
        assert 1989 in merged.index_values
        assert 1990 in merged.index_values
        assert 1991 in merged.index_values
    
    def test_merge_indices_empty_error(self):
        """Test error when merging empty list."""
        builder = IndexBuilder()
        
        with pytest.raises(ValueError, match="No indices to merge"):
            builder.merge_indices([])
    
    def test_merge_indices_invalid_weights(self):
        """Test error with invalid merge weights."""
        builder = IndexBuilder()
        
        indices = [
            HPIIndex(
                index_values={1989: 1.0},
                entity_id="12345",
                entity_type="tract"
            )
        ]
        
        with pytest.raises(ValueError, match="Number of weights"):
            builder.merge_indices(indices, weights=[0.5, 0.5])
        
        with pytest.raises(ValueError, match="sum to 1.0"):
            builder.merge_indices(indices, weights=[0.5])