"""Integration tests for complete RSAI pipeline"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import os

from rsai.src.main import RSAIPipeline
from rsai.tests.generate_sample_data import save_sample_data


class TestRSAIPipeline:
    """Test complete RSAI pipeline integration"""
    
    @pytest.fixture(scope="class")
    def sample_data_dir(self):
        """Create sample data in temporary directory"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Generate sample data
        file_paths = save_sample_data(output_dir=temp_dir)
        
        yield temp_dir, file_paths
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        pipeline = RSAIPipeline(
            min_half_pairs=40,
            base_index_value=100.0,
            base_year=2016
        )
        
        assert pipeline.min_half_pairs == 40
        assert pipeline.base_index_value == 100.0
        assert pipeline.base_year == 2016
    
    def test_data_loading(self, sample_data_dir):
        """Test data loading functionality"""
        temp_dir, file_paths = sample_data_dir
        
        pipeline = RSAIPipeline()
        
        data = pipeline.load_data(
            transaction_file=str(file_paths['transactions']),
            geographic_file=str(file_paths['geographic']),
            weighting_file=str(file_paths['weighting'])
        )
        
        # Check all data loaded
        assert 'transactions' in data
        assert 'geographic' in data
        assert 'weighting' in data
        
        # Check data integrity
        assert len(data['transactions']) > 0
        assert len(data['geographic']) > 0
        assert len(data['weighting']) > 0
        
        # Check required columns
        assert 'property_id' in data['transactions'].columns
        assert 'centroid_lat' in data['geographic'].columns
        assert 'total_housing_units' in data['weighting'].columns
    
    def test_complete_pipeline_run(self, sample_data_dir):
        """Test running complete pipeline end-to-end"""
        temp_dir, file_paths = sample_data_dir
        
        pipeline = RSAIPipeline(min_half_pairs=20)  # Lower threshold for test data
        
        # Create output file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            output_file = f.name
        
        try:
            # Run pipeline
            index_df = pipeline.run_pipeline(
                transaction_file=str(file_paths['transactions']),
                geographic_file=str(file_paths['geographic']),
                output_file=output_file,
                start_year=2017,
                end_year=2020,
                weighting_file=str(file_paths['weighting']),
                weighting_schemes=['sample', 'value'],
                output_format='csv',
                wide_format=False
            )
            
            # Check output
            assert len(index_df) > 0
            assert 'index_value' in index_df.columns
            assert 'cbsa_id' in index_df.columns
            assert 'weighting_scheme' in index_df.columns
            
            # Check file was created
            assert os.path.exists(output_file)
            
            # Check summary file
            summary_file = Path(output_file).parent / "summary_statistics.csv"
            assert os.path.exists(summary_file)
            
            # Read and verify output
            output_df = pd.read_csv(output_file)
            assert len(output_df) > 0
            
            # Check we have data for requested years
            years = output_df['year'].unique()
            assert all(year in years for year in [2017, 2018, 2019, 2020])
            
            # Check we have both weighting schemes
            schemes = output_df['weighting_scheme'].unique()
            assert 'sample' in schemes
            assert 'value' in schemes
            
            # Cleanup
            os.unlink(summary_file)
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_pipeline_with_single_weighting_scheme(self, sample_data_dir):
        """Test pipeline with single weighting scheme"""
        temp_dir, file_paths = sample_data_dir
        
        pipeline = RSAIPipeline(min_half_pairs=20)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            output_file = f.name
        
        try:
            index_df = pipeline.run_pipeline(
                transaction_file=str(file_paths['transactions']),
                geographic_file=str(file_paths['geographic']),
                output_file=output_file,
                start_year=2018,
                end_year=2020,
                weighting_schemes=['sample'],  # Only sample weighting
                output_format='csv'
            )
            
            # Should only have sample weighting
            assert all(index_df['weighting_scheme'] == 'sample')
            
            # Cleanup summary
            summary_file = Path(output_file).parent / "summary_statistics.csv"
            if os.path.exists(summary_file):
                os.unlink(summary_file)
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_pipeline_wide_format_output(self, sample_data_dir):
        """Test pipeline with wide format output"""
        temp_dir, file_paths = sample_data_dir
        
        pipeline = RSAIPipeline(min_half_pairs=20)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            output_file = f.name
        
        try:
            index_df = pipeline.run_pipeline(
                transaction_file=str(file_paths['transactions']),
                geographic_file=str(file_paths['geographic']),
                output_file=output_file,
                start_year=2019,
                end_year=2020,
                weighting_file=str(file_paths['weighting']),
                weighting_schemes=['sample', 'value', 'unit'],
                output_format='csv',
                wide_format=True
            )
            
            # Read wide format output
            output_df = pd.read_csv(output_file)
            
            # Should have weighting schemes as columns
            assert 'sample' in output_df.columns
            assert 'value' in output_df.columns
            assert 'unit' in output_df.columns
            
            # Should have CBSA and year columns
            assert 'cbsa_id' in output_df.columns
            assert 'year' in output_df.columns
            
            # Cleanup
            summary_file = Path(output_file).parent / "summary_statistics.csv"
            if os.path.exists(summary_file):
                os.unlink(summary_file)
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_pipeline_parquet_output(self, sample_data_dir):
        """Test pipeline with Parquet output format"""
        temp_dir, file_paths = sample_data_dir
        
        pipeline = RSAIPipeline(min_half_pairs=20)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            output_file = f.name
        
        try:
            index_df = pipeline.run_pipeline(
                transaction_file=str(file_paths['transactions']),
                geographic_file=str(file_paths['geographic']),
                output_file=output_file,
                start_year=2019,
                end_year=2020,
                output_format='parquet'
            )
            
            # Check file exists
            assert os.path.exists(output_file)
            
            # Read and verify
            output_df = pd.read_parquet(output_file)
            assert len(output_df) > 0
            assert 'index_value' in output_df.columns
            
            # Cleanup
            summary_file = Path(output_file).parent / "summary_statistics.csv"
            if os.path.exists(summary_file):
                os.unlink(summary_file)
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_index_continuity(self, sample_data_dir):
        """Test that index values are continuous and reasonable"""
        temp_dir, file_paths = sample_data_dir
        
        pipeline = RSAIPipeline(
            min_half_pairs=20,
            base_index_value=100.0,
            base_year=2017
        )
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            output_file = f.name
        
        try:
            index_df = pipeline.run_pipeline(
                transaction_file=str(file_paths['transactions']),
                geographic_file=str(file_paths['geographic']),
                output_file=output_file,
                start_year=2017,
                end_year=2020,
                weighting_schemes=['sample']
            )
            
            # Check each CBSA
            for cbsa_id in index_df['cbsa_id'].unique():
                cbsa_data = index_df[
                    (index_df['cbsa_id'] == cbsa_id) & 
                    (index_df['weighting_scheme'] == 'sample')
                ].sort_values('year')
                
                if len(cbsa_data) > 0:
                    # Base year should have value 100
                    base_year_data = cbsa_data[cbsa_data['year'] == 2017]
                    if not base_year_data.empty:
                        assert abs(base_year_data.iloc[0]['index_value'] - 100.0) < 0.0001
                    
                    # Check reasonable appreciation (not more than 30% per year)
                    for i in range(1, len(cbsa_data)):
                        prev_value = cbsa_data.iloc[i-1]['index_value']
                        curr_value = cbsa_data.iloc[i]['index_value']
                        
                        annual_change = (curr_value / prev_value) - 1
                        assert -0.3 < annual_change < 0.3
            
            # Cleanup
            summary_file = Path(output_file).parent / "summary_statistics.csv"
            if os.path.exists(summary_file):
                os.unlink(summary_file)
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_error_handling_missing_file(self):
        """Test error handling for missing input files"""
        pipeline = RSAIPipeline()
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            output_file = f.name
        
        try:
            # Should raise error for non-existent file
            with pytest.raises(Exception):  # Could be FileNotFoundError or similar
                pipeline.run_pipeline(
                    transaction_file='non_existent_file.csv',
                    geographic_file='also_missing.csv',
                    output_file=output_file,
                    start_year=2019,
                    end_year=2020
                )
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)