"""Unit tests for output generation and export"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from rsai.src.output.export import IndexChainer, OutputGenerator, RSAIExporter


class TestIndexChainer:
    """Test IndexChainer class"""
    
    @pytest.fixture
    def appreciation_data(self):
        """Create sample appreciation rate data"""
        data = []
        
        # Create data for 2 CBSAs, 2 weighting schemes, 5 years
        for cbsa in ['31080', '41860']:
            for scheme in ['sample', 'value']:
                for year in range(2016, 2021):
                    # Different appreciation patterns
                    if cbsa == '31080':
                        appr_rate = 0.05 + 0.01 * (year - 2016)  # 5-9%
                    else:
                        appr_rate = 0.03 + 0.005 * (year - 2016)  # 3-5%
                    
                    data.append({
                        'cbsa_id': cbsa,
                        'year': year,
                        'weighting_scheme': scheme,
                        'appreciation_rate': appr_rate,
                        'n_supertracts': 10,
                        'total_observations': 500
                    })
        
        return pd.DataFrame(data)
    
    def test_chain_appreciation_rates(self, appreciation_data):
        """Test chaining of appreciation rates"""
        chainer = IndexChainer(base_value=100.0, base_year=2016)
        
        result = chainer.chain_appreciation_rates(
            appreciation_data, '31080', 'sample'
        )
        
        # Check structure
        assert len(result) == 5  # 2016-2020
        assert 'year' in result.columns
        assert 'index_value' in result.columns
        assert 'appreciation_rate' in result.columns
        
        # Base year should have index value of 100
        base_row = result[result['year'] == 2016]
        assert base_row['index_value'].iloc[0] == 100.0
        
        # Check chaining calculation
        for i in range(1, len(result)):
            prev_value = result.iloc[i-1]['index_value']
            appr_rate = result.iloc[i]['appreciation_rate']
            expected_value = prev_value * np.exp(appr_rate)
            
            assert abs(result.iloc[i]['index_value'] - expected_value) < 0.0001
    
    def test_chain_with_different_base_year(self, appreciation_data):
        """Test chaining with non-first base year"""
        chainer = IndexChainer(base_value=100.0, base_year=2018)
        
        result = chainer.chain_appreciation_rates(
            appreciation_data, '31080', 'sample'
        )
        
        # 2018 should have index value of 100
        base_row = result[result['year'] == 2018]
        assert base_row['index_value'].iloc[0] == 100.0
        
        # Earlier years should have lower values
        earlier = result[result['year'] < 2018]
        assert all(earlier['index_value'] < 100.0)
        
        # Later years should have higher values
        later = result[result['year'] > 2018]
        assert all(later['index_value'] > 100.0)
    
    def test_chain_all_indices(self, appreciation_data):
        """Test chaining all CBSA/scheme combinations"""
        chainer = IndexChainer(base_value=100.0)
        
        all_indices = chainer.chain_all_indices(appreciation_data)
        
        # Should have 4 combinations (2 CBSAs × 2 schemes)
        unique_combos = all_indices.groupby(['cbsa_id', 'weighting_scheme']).size()
        assert len(unique_combos) == 4
        
        # Each combination should have 5 years
        assert all(unique_combos == 5)
        
        # All should have proper structure
        required_cols = ['year', 'index_value', 'appreciation_rate', 
                        'cbsa_id', 'weighting_scheme']
        assert all(col in all_indices.columns for col in required_cols)


class TestOutputGenerator:
    """Test OutputGenerator class"""
    
    @pytest.fixture
    def index_data(self):
        """Create sample index data"""
        data = []
        
        for year in range(2016, 2021):
            index_value = 100 * (1.05 ** (year - 2016))  # 5% annual growth
            
            data.append({
                'cbsa_id': '31080',
                'year': year,
                'weighting_scheme': 'sample',
                'index_value': index_value,
                'appreciation_rate': 0.05
            })
        
        return pd.DataFrame(data)
    
    def test_prepare_standard_output(self, index_data):
        """Test preparation of standard output format"""
        output_gen = OutputGenerator()
        
        result = output_gen.prepare_standard_output(index_data)
        
        # Check new columns
        assert 'yoy_change' in result.columns
        assert 'cumulative_change' in result.columns
        
        # First year should have NaN yoy_change
        assert pd.isna(result.iloc[0]['yoy_change'])
        
        # Check yoy calculation
        for i in range(1, len(result)):
            expected_yoy = (result.iloc[i]['index_value'] / 
                          result.iloc[i-1]['index_value'] - 1) * 100
            assert abs(result.iloc[i]['yoy_change'] - expected_yoy) < 0.0001
        
        # Check cumulative change
        base_value = result.iloc[0]['index_value']
        for i in range(len(result)):
            expected_cum = (result.iloc[i]['index_value'] / base_value - 1) * 100
            assert abs(result.iloc[i]['cumulative_change'] - expected_cum) < 0.0001
    
    def test_export_to_csv(self, index_data):
        """Test CSV export"""
        output_gen = OutputGenerator()
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            # Test long format
            output_gen.export_to_csv(index_data, temp_file, wide_format=False)
            
            # Read back and verify
            df = pd.read_csv(temp_file)
            assert len(df) == len(index_data)
            assert 'yoy_change' in df.columns
            
            os.unlink(temp_file)
            
            # Test wide format with multiple schemes
            multi_scheme_data = pd.concat([
                index_data,
                index_data.assign(weighting_scheme='value')
            ])
            
            output_gen.export_to_csv(multi_scheme_data, temp_file, wide_format=True)
            
            # Read back and verify
            df = pd.read_csv(temp_file)
            assert 'sample' in df.columns
            assert 'value' in df.columns
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_generate_summary_statistics(self, index_data):
        """Test summary statistics generation"""
        output_gen = OutputGenerator()
        
        # Add more data
        multi_data = pd.concat([
            index_data,
            index_data.assign(cbsa_id='41860', 
                            appreciation_rate=[0.03, 0.08, -0.02, 0.05, 0.04])
        ])
        
        summary = output_gen.generate_summary_statistics(multi_data)
        
        # Check structure
        assert len(summary) == 2  # 2 CBSAs
        required_cols = ['cbsa_id', 'weighting_scheme', 'start_year', 'end_year',
                        'mean_appreciation', 'std_appreciation', 'total_appreciation']
        assert all(col in summary.columns for col in required_cols)
        
        # Check calculations for first CBSA
        cbsa_31080 = summary[summary['cbsa_id'] == '31080'].iloc[0]
        assert cbsa_31080['mean_appreciation'] == 0.05
        assert cbsa_31080['std_appreciation'] == 0.0  # All same value
        
        # Check total appreciation
        expected_total = (index_data.iloc[-1]['index_value'] / 
                         index_data.iloc[0]['index_value'] - 1) * 100
        assert abs(cbsa_31080['total_appreciation'] - expected_total) < 0.0001


class TestRSAIExporter:
    """Test RSAIExporter integration"""
    
    @pytest.fixture
    def appreciation_data(self):
        """Create multi-CBSA appreciation data"""
        data = []
        
        for cbsa in ['31080', '41860']:
            for scheme in ['sample', 'value']:
                for year in range(2016, 2021):
                    data.append({
                        'cbsa_id': cbsa,
                        'year': year,
                        'weighting_scheme': scheme,
                        'appreciation_rate': 0.05 if cbsa == '31080' else 0.03,
                        'n_supertracts': 10,
                        'total_observations': 500
                    })
        
        return pd.DataFrame(data)
    
    def test_process_and_export(self, appreciation_data):
        """Test complete processing and export"""
        exporter = RSAIExporter(base_value=100.0, base_year=2016)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            # Process and export
            index_df = exporter.process_and_export(
                appreciation_data,
                temp_file,
                format='csv',
                wide_format=False,
                include_summary=True
            )
            
            # Check returned data
            assert len(index_df) > 0
            assert 'index_value' in index_df.columns
            
            # Check main file exists
            assert os.path.exists(temp_file)
            
            # Check summary file exists
            summary_path = Path(temp_file).parent / "summary_statistics.csv"
            assert os.path.exists(summary_path)
            
            # Read and verify summary
            summary_df = pd.read_csv(summary_path)
            assert len(summary_df) == 4  # 2 CBSAs × 2 schemes
            
            # Cleanup
            os.unlink(summary_path)
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_parquet_export(self, appreciation_data):
        """Test Parquet format export"""
        exporter = RSAIExporter(base_value=100.0)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_file = f.name
        
        try:
            index_df = exporter.process_and_export(
                appreciation_data,
                temp_file,
                format='parquet',
                include_summary=False
            )
            
            # Check file exists
            assert os.path.exists(temp_file)
            
            # Read back and verify
            df = pd.read_parquet(temp_file)
            assert len(df) == len(index_df)
            assert 'yoy_change' in df.columns
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)