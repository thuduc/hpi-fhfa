"""
Unit tests for output export functionality.

Tests exporting RSAI model results in various formats including
index values, reports, and visualizations.
"""

import pytest
from datetime import datetime, date
from pathlib import Path
import json
import polars as pl
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from rsai.src.output.export import OutputExporter
from rsai.src.data.models import (
    IndexValue,
    BMNRegressionResult,
    QualityMetrics,
    GeographyLevel,
    RSAIConfig
)


class TestOutputExporter:
    """Test OutputExporter class."""
    
    def test_initialization(self, temp_output_dir, test_config):
        """Test OutputExporter initialization."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        assert exporter.output_dir == temp_output_dir
        assert exporter.config == test_config
        
        # Check subdirectories were created
        assert exporter.indices_dir.exists()
        assert exporter.reports_dir.exists()
        assert exporter.plots_dir.exists()
        assert exporter.data_dir.exists()
    
    def test_export_index_values_parquet(self, temp_output_dir, test_config, sample_index_values):
        """Test exporting index values to Parquet format."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        output_path = exporter.export_index_values(
            sample_index_values,
            format="parquet",
            filename_prefix="test_index"
        )
        
        assert output_path.exists()
        assert output_path.suffix == ".parquet"
        
        # Read back and verify
        df = pl.read_parquet(output_path)
        assert len(df) == len(sample_index_values)
        assert "geography_level" in df.columns
        assert "index_value" in df.columns
        assert "period" in df.columns
    
    def test_export_index_values_csv(self, temp_output_dir, test_config, sample_index_values):
        """Test exporting index values to CSV format."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        output_path = exporter.export_index_values(
            sample_index_values,
            format="csv",
            filename_prefix="test_index"
        )
        
        assert output_path.exists()
        assert output_path.suffix == ".csv"
        
        # Read back and verify
        df = pl.read_csv(output_path)
        assert len(df) == len(sample_index_values)
    
    def test_export_index_values_json(self, temp_output_dir, test_config, sample_index_values):
        """Test exporting index values to JSON format."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        output_path = exporter.export_index_values(
            sample_index_values,
            format="json",
            filename_prefix="test_index"
        )
        
        assert output_path.exists()
        assert output_path.suffix == ".json"
        
        # Read back and verify
        df = pl.read_json(output_path)
        assert len(df) == len(sample_index_values)
    
    def test_export_index_values_invalid_format(self, temp_output_dir, test_config, sample_index_values):
        """Test error handling for invalid export format."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export_index_values(
                sample_index_values,
                format="invalid_format"
            )
    
    def test_export_regression_results(self, temp_output_dir, test_config, sample_bmn_result):
        """Test exporting regression results."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        results = {"06037": sample_bmn_result}
        
        output_path = exporter.export_regression_results(
            results,
            include_diagnostics=True
        )
        
        assert output_path.exists()
        assert output_path.suffix == ".json"
        
        # Read back and verify
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert "06037" in data
        assert data["06037"]["geography_level"] == "county"
        assert data["06037"]["r_squared"] == sample_bmn_result.r_squared
        assert "coefficients" in data["06037"]
        assert "standard_errors" in data["06037"]
    
    def test_export_regression_results_no_diagnostics(self, temp_output_dir, test_config, sample_bmn_result):
        """Test exporting regression results without diagnostics."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        results = {"06037": sample_bmn_result}
        
        output_path = exporter.export_regression_results(
            results,
            include_diagnostics=False
        )
        
        # Read back and verify
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert "06037" in data
        assert "coefficients" not in data["06037"]
        assert "standard_errors" not in data["06037"]
    
    def test_generate_summary_report_html(self, temp_output_dir, test_config, sample_index_values):
        """Test generating HTML summary report."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        # Create index DataFrame
        index_df = pl.DataFrame([
            {
                "geography_level": iv.geography_level.value if hasattr(iv.geography_level, 'value') else iv.geography_level,
                "geography_id": iv.geography_id,
                "period": iv.period,
                "index_value": iv.index_value,
                "num_pairs": iv.num_pairs,
                "num_properties": iv.num_properties
            }
            for iv in sample_index_values
        ])
        
        report_path = exporter.generate_summary_report(
            index_df,
            format="html"
        )
        
        assert report_path.exists()
        assert report_path.suffix == ".html"
        
        # Check content
        content = report_path.read_text()
        assert "RSAI Model Report" in content
        assert "Summary Statistics" in content
        assert "Total Geographies" in content
    
    def test_generate_summary_report_with_validation(self, temp_output_dir, test_config, 
                                                   sample_index_values, sample_quality_metrics):
        """Test generating report with validation results."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        # Create index DataFrame
        index_df = pl.DataFrame([
            {
                "geography_level": iv.geography_level.value if hasattr(iv.geography_level, 'value') else iv.geography_level,
                "geography_id": iv.geography_id,
                "period": iv.period,
                "index_value": iv.index_value,
                "num_pairs": iv.num_pairs,
                "num_properties": iv.num_properties
            }
            for iv in sample_index_values
        ])
        
        validation_results = {"transactions": sample_quality_metrics}
        
        report_path = exporter.generate_summary_report(
            index_df,
            validation_results=validation_results,
            format="html"
        )
        
        content = report_path.read_text()
        assert "Data Quality" in content
        assert "Transactions Validation" in content
        assert f"{sample_quality_metrics.total_records:,}" in content
    
    def test_generate_summary_report_markdown(self, temp_output_dir, test_config, sample_index_values):
        """Test generating Markdown summary report."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        # Create index DataFrame
        index_df = pl.DataFrame([
            {
                "geography_level": iv.geography_level.value if hasattr(iv.geography_level, 'value') else iv.geography_level,
                "geography_id": iv.geography_id,
                "period": iv.period,
                "index_value": iv.index_value,
                "num_pairs": iv.num_pairs,
                "num_properties": iv.num_properties
            }
            for iv in sample_index_values
        ])
        
        report_path = exporter.generate_summary_report(
            index_df,
            format="markdown"
        )
        
        assert report_path.exists()
        assert report_path.suffix == ".md"
        
        content = report_path.read_text()
        assert "# RSAI Model Report" in content
        assert "## Summary Statistics" in content
    
    def test_create_index_plots_interactive(self, temp_output_dir, test_config, sample_index_values):
        """Test creating interactive plots."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        # Create index DataFrame
        index_df = pl.DataFrame([
            {
                "geography_level": iv.geography_level.value if hasattr(iv.geography_level, 'value') else iv.geography_level,
                "geography_id": iv.geography_id,
                "period": iv.period,
                "index_value": iv.index_value,
                "num_pairs": iv.num_pairs
            }
            for iv in sample_index_values
        ])
        
        # Mock plotly to avoid actual file creation
        with patch('rsai.src.output.export.go.Figure') as mock_figure:
            mock_fig_instance = MagicMock()
            mock_figure.return_value = mock_fig_instance
            
            plot_paths = exporter.create_index_plots(
                index_df,
                interactive=True
            )
            
            # Should create time series plot
            assert "time_series" in plot_paths
            mock_fig_instance.write_html.assert_called()
    
    def test_create_index_plots_static(self, temp_output_dir, test_config, sample_index_values):
        """Test creating static plots."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        # Create index DataFrame with multiple geographies
        index_data = []
        for geo_id in ["06037", "06059"]:
            for iv in sample_index_values:
                index_data.append({
                    "geography_level": iv.geography_level.value if hasattr(iv.geography_level, 'value') else iv.geography_level,
                    "geography_id": geo_id,
                    "period": iv.period,
                    "index_value": iv.index_value,
                    "num_pairs": iv.num_pairs
                })
        
        index_df = pl.DataFrame(index_data)
        
        # Mock matplotlib and seaborn to avoid actual plot creation
        with patch('rsai.src.output.export.plt') as mock_plt, \
             patch('rsai.src.output.export.sns') as mock_sns:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            plot_paths = exporter.create_index_plots(
                index_df,
                interactive=False
            )
            
            # Should create multiple plots
            assert len(plot_paths) > 0
            mock_fig.savefig.assert_called()
    
    def test_export_for_tableau(self, temp_output_dir, test_config, sample_index_values, sample_repeat_sales_df):
        """Test exporting data for Tableau."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        # Create index DataFrame
        index_df = pl.DataFrame([
            {
                "geography_level": iv.geography_level.value if hasattr(iv.geography_level, 'value') else iv.geography_level,
                "geography_id": iv.geography_id,
                "period": iv.period,
                "index_value": iv.index_value,
                "num_pairs": iv.num_pairs,
                "num_properties": iv.num_properties
            }
            for iv in sample_index_values
        ])
        
        # Add period column to repeat sales
        repeat_sales_with_period = sample_repeat_sales_df.with_columns([
            pl.col("sale2_date").dt.truncate("1mo").alias("period"),
            pl.col("county_fips").alias("geography_id")
        ])
        
        output_path = exporter.export_for_tableau(
            index_df,
            repeat_sales_with_period
        )
        
        assert output_path.exists()
        assert output_path.suffix == ".csv"
        
        # Read back and verify
        tableau_df = pl.read_csv(output_path)
        assert "year" in tableau_df.columns
        assert "month" in tableau_df.columns
        assert "quarter" in tableau_df.columns
        assert "yoy_growth" in tableau_df.columns
        assert "rebased_index" in tableau_df.columns
    
    def test_create_methodology_document(self, temp_output_dir, test_config):
        """Test creating methodology documentation."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        doc_path = exporter.create_methodology_document()
        
        assert doc_path.exists()
        assert doc_path.name == "methodology.md"
        
        content = doc_path.read_text()
        assert "RSAI Model Methodology" in content
        assert "Bailey-Muth-Nourse" in content
        assert (test_config.weighting_scheme.value if hasattr(test_config.weighting_scheme, 'value') else test_config.weighting_scheme) in content
        assert f"{test_config.min_price:,.0f}" in content
    
    def test_calculate_summary_statistics(self, temp_output_dir, test_config, sample_index_values):
        """Test calculation of summary statistics."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        # Create index DataFrame
        index_df = pl.DataFrame([
            {
                "geography_id": iv.geography_id,
                "period": iv.period,
                "index_value": iv.index_value,
                "num_pairs": iv.num_pairs
            }
            for iv in sample_index_values
        ])
        
        stats = exporter._calculate_summary_statistics(index_df)
        
        assert "total_geographies" in stats
        assert "start_date" in stats
        assert "end_date" in stats
        assert "total_observations" in stats
        assert "avg_annual_growth" in stats
        assert "volatility" in stats
        assert "max_drawdown" in stats
        
        assert stats["total_geographies"] == index_df["geography_id"].n_unique()
        assert stats["start_date"] == index_df["period"].min()
        assert stats["end_date"] == index_df["period"].max()
    
    def test_calculate_growth_rates(self, temp_output_dir, test_config, sample_index_values):
        """Test growth rate calculation."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        # Create index DataFrame
        index_df = pl.DataFrame([
            {
                "geography_id": iv.geography_id,
                "period": iv.period,
                "index_value": iv.index_value
            }
            for iv in sample_index_values
        ])
        
        growth_df = exporter._calculate_growth_rates(index_df)
        
        assert "period_growth" in growth_df.columns
        assert "annual_growth" in growth_df.columns
        
        # First period should have null growth
        first_row = growth_df.sort("period").head(1)
        assert first_row["period_growth"][0] is None
    
    def test_create_geographic_comparison(self, temp_output_dir, test_config):
        """Test geographic comparison plot creation."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        # Create index data for multiple geographies
        index_data = []
        for geo_id in ["County1", "County2", "County3"]:
            index_data.append({
                "geography_id": geo_id,
                "period": date(2023, 12, 1),
                "index_value": 100 + np.random.uniform(-10, 20)
            })
        
        index_df = pl.DataFrame(index_data)
        
        with patch('rsai.src.output.export.plt') as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            fig = exporter._create_geographic_comparison(index_df)
            
            assert fig == mock_fig
            mock_ax.barh.assert_called()
            mock_ax.axvline.assert_called_with(x=100, color='red', linestyle='--', alpha=0.5)
    
    def test_empty_data_handling(self, temp_output_dir, test_config):
        """Test handling of empty data."""
        exporter = OutputExporter(temp_output_dir, test_config)
        
        # Empty index values
        output_path = exporter.export_index_values([], format="parquet")
        
        df = pl.read_parquet(output_path)
        assert len(df) == 0
        assert "geography_level" in df.columns  # Schema should still be correct