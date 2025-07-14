"""
Output generation and export using Polars.

This module handles exporting RSAI model results in various formats
including index values, reports, and visualizations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px

from rsai.src.data.models import (
    IndexValue,
    BMNRegressionResult,
    QualityMetrics,
    GeographyLevel,
    RSAIConfig
)

logger = logging.getLogger(__name__)


class OutputExporter:
    """Handles exporting RSAI results in various formats."""
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        config: RSAIConfig
    ):
        """
        Initialize output exporter.
        
        Args:
            output_dir: Base directory for outputs
            config: RSAI configuration
        """
        self.output_dir = Path(output_dir)
        self.config = config
        
        # Create output subdirectories
        self.indices_dir = self.output_dir / "indices"
        self.reports_dir = self.output_dir / "reports"
        self.plots_dir = self.output_dir / "plots"
        self.data_dir = self.output_dir / "data"
        
        for dir_path in [self.indices_dir, self.reports_dir, self.plots_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def export_index_values(
        self,
        index_values: List[IndexValue],
        format: str = "parquet",
        filename_prefix: str = "index"
    ) -> Path:
        """
        Export index values to file.
        
        Args:
            index_values: List of IndexValue objects
            format: Output format ('parquet', 'csv', 'json')
            filename_prefix: Prefix for output filename
            
        Returns:
            Path to exported file
        """
        # Convert to DataFrame
        if not index_values:
            # Create empty DataFrame with correct schema
            index_df = pl.DataFrame({
                "geography_level": [],
                "geography_id": [],
                "period": [],
                "index_value": [],
                "num_pairs": [],
                "num_properties": [],
                "median_price": [],
                "standard_error": [],
                "confidence_lower": [],
                "confidence_upper": []
            }, schema={
                "geography_level": pl.Utf8,
                "geography_id": pl.Utf8,
                "period": pl.Date,
                "index_value": pl.Float64,
                "num_pairs": pl.Int32,
                "num_properties": pl.Int32,
                "median_price": pl.Float64,
                "standard_error": pl.Float64,
                "confidence_lower": pl.Float64,
                "confidence_upper": pl.Float64
            })
        else:
            index_df = pl.DataFrame([
                {
                    "geography_level": iv.geography_level.value if hasattr(iv.geography_level, 'value') else iv.geography_level,
                    "geography_id": iv.geography_id,
                    "period": iv.period,
                    "index_value": iv.index_value,
                    "num_pairs": iv.num_pairs,
                    "num_properties": iv.num_properties,
                    "median_price": iv.median_price,
                    "standard_error": iv.standard_error,
                    "confidence_lower": iv.confidence_lower,
                    "confidence_upper": iv.confidence_upper
                }
                for iv in index_values
            ])
            
            # Sort by geography and period
            index_df = index_df.sort(["geography_level", "geography_id", "period"])
        
        # Export based on format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.{format}"
        output_path = self.indices_dir / filename
        
        if format == "parquet":
            index_df.write_parquet(output_path, compression="snappy")
        elif format == "csv":
            index_df.write_csv(output_path)
        elif format == "json":
            index_df.write_json(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Exported {len(index_values)} index values to {output_path}")
        return output_path
        
    def export_regression_results(
        self,
        regression_results: Dict[str, BMNRegressionResult],
        include_diagnostics: bool = True
    ) -> Path:
        """
        Export regression results to structured format.
        
        Args:
            regression_results: Dictionary of regression results
            include_diagnostics: Include diagnostic statistics
            
        Returns:
            Path to exported results
        """
        output_data = {}
        
        for geo_id, result in regression_results.items():
            geo_data = {
                "geography_level": result.geography_level.value if hasattr(result.geography_level, 'value') else result.geography_level,
                "geography_id": result.geography_id,
                "start_period": result.start_period.isoformat(),
                "end_period": result.end_period.isoformat(),
                "num_periods": result.num_periods,
                "num_observations": result.num_observations,
                "r_squared": result.r_squared,
                "adj_r_squared": result.adj_r_squared
            }
            
            if include_diagnostics:
                geo_data["coefficients"] = result.coefficients
                geo_data["standard_errors"] = result.standard_errors
                geo_data["t_statistics"] = result.t_statistics
                geo_data["p_values"] = result.p_values
                
            output_data[geo_id] = geo_data
            
        # Save as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.reports_dir / f"regression_results_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
            
        logger.info(f"Exported regression results to {output_path}")
        return output_path
        
    def generate_summary_report(
        self,
        index_df: pl.DataFrame,
        validation_results: Optional[Dict[str, QualityMetrics]] = None,
        format: str = "html"
    ) -> Path:
        """
        Generate comprehensive summary report.
        
        Args:
            index_df: DataFrame with all index values
            validation_results: Optional validation results
            format: Report format ('html', 'pdf', 'markdown')
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "html":
            report_path = self._generate_html_report(index_df, validation_results, timestamp)
        elif format == "markdown":
            report_path = self._generate_markdown_report(index_df, validation_results, timestamp)
        else:
            raise ValueError(f"Unsupported report format: {format}")
            
        logger.info(f"Generated summary report: {report_path}")
        return report_path
        
    def create_index_plots(
        self,
        index_df: pl.DataFrame,
        geography_ids: Optional[List[str]] = None,
        interactive: bool = True
    ) -> Dict[str, Path]:
        """
        Create visualizations of index values.
        
        Args:
            index_df: DataFrame with index values
            geography_ids: List of geography IDs to plot (None = all)
            interactive: Create interactive plots using Plotly
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        plot_paths = {}
        
        # Filter geographies if specified
        if geography_ids:
            plot_df = index_df.filter(pl.col("geography_id").is_in(geography_ids))
        else:
            plot_df = index_df
            
        # Time series plot
        if interactive:
            fig = self._create_interactive_time_series(plot_df)
            plot_path = self.plots_dir / "index_time_series.html"
            fig.write_html(str(plot_path))
            plot_paths["time_series"] = plot_path
        else:
            fig = self._create_static_time_series(plot_df)
            plot_path = self.plots_dir / "index_time_series.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plot_paths["time_series"] = plot_path
            
        # Growth rate heatmap
        growth_df = self._calculate_growth_rates(plot_df)
        if len(growth_df) > 0:
            fig = self._create_growth_heatmap(growth_df)
            plot_path = self.plots_dir / "growth_heatmap.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plot_paths["growth_heatmap"] = plot_path
            
        # Geographic comparison
        if plot_df["geography_id"].n_unique() > 1:
            fig = self._create_geographic_comparison(plot_df)
            plot_path = self.plots_dir / "geographic_comparison.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plot_paths["geographic_comparison"] = plot_path
            
        return plot_paths
        
    def export_for_tableau(
        self,
        index_df: pl.DataFrame,
        repeat_sales_df: Optional[pl.DataFrame] = None
    ) -> Path:
        """
        Export data formatted for Tableau visualization.
        
        Args:
            index_df: DataFrame with index values
            repeat_sales_df: Optional repeat sales data
            
        Returns:
            Path to Tableau-ready file
        """
        # Prepare index data with additional calculated fields
        tableau_df = index_df.with_columns([
            # Year and month for easier filtering
            pl.col("period").dt.year().alias("year"),
            pl.col("period").dt.month().alias("month"),
            pl.col("period").dt.quarter().alias("quarter"),
            
            # Year-over-year growth
            pl.col("index_value").pct_change(12).over("geography_id").alias("yoy_growth"),
            
            # Indexed to 100 at start
            (pl.col("index_value") / pl.col("index_value").first().over("geography_id") * 100)
            .alias("rebased_index")
        ])
        
        # Add repeat sales summary if available
        if repeat_sales_df is not None:
            sales_summary = repeat_sales_df.group_by(["geography_id", "period"]).agg([
                pl.col("price_ratio").mean().alias("avg_price_ratio"),
                pl.col("holding_period_days").mean().alias("avg_holding_days"),
                pl.col("annualized_return").mean().alias("avg_return")
            ])
            
            tableau_df = tableau_df.join(
                sales_summary,
                on=["geography_id", "period"],
                how="left"
            )
            
        # Export as Hyper file if tableau-hyper-api available, otherwise CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.data_dir / f"tableau_data_{timestamp}.csv"
        tableau_df.write_csv(output_path)
        
        logger.info(f"Exported Tableau data to {output_path}")
        return output_path
        
    def create_methodology_document(self) -> Path:
        """
        Generate methodology documentation.
        
        Returns:
            Path to methodology document
        """
        methodology = f"""
# RSAI Model Methodology

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Configuration

- **Index Type**: Repeat Sales Price Index
- **Regression Method**: Bailey-Muth-Nourse (BMN)
- **Weighting Scheme**: {self.config.weighting_scheme.value if hasattr(self.config.weighting_scheme, 'value') else self.config.weighting_scheme if self.config else 'Not specified'}
- **Time Frequency**: {self.config.frequency if self.config else 'Not specified'}
- **Geographic Levels**: {', '.join([gl.value if hasattr(gl, 'value') else str(gl) for gl in self.config.geography_levels]) if self.config else 'Not specified'}

## Data Filters

- **Price Range**: ${self.config.min_price if self.config else 0:,.0f} - ${self.config.max_price if self.config else 0:,.0f}
- **Maximum Holding Period**: {self.config.max_holding_period_years if self.config else 'Not specified'} years
- **Minimum Pairs per Geography**: {self.config.min_pairs_threshold if self.config else 'Not specified'}
- **Outlier Threshold**: {self.config.outlier_std_threshold if self.config else 'Not specified'} standard deviations

## Methodology Steps

1. **Data Ingestion**
   - Load property transaction data
   - Standardize fields and formats
   - Apply date and price filters

2. **Repeat Sales Identification**
   - Match properties across transactions
   - Calculate holding periods and price ratios
   - Filter by transaction type (arms-length only)

3. **Data Validation**
   - Check for data quality issues
   - Remove outliers using IQR method
   - Validate geographic consistency

4. **Geographic Aggregation**
   - Generate supertracts from census tracts
   - Create geographic hierarchy mappings
   - Calculate distance-based relationships

5. **Index Calculation**
   - Apply BMN regression by geography
   - Calculate time dummy coefficients
   - Convert to index values (base = 100)

6. **Weighting and Aggregation**
   - Apply selected weighting scheme
   - Aggregate to higher geographic levels
   - Calculate confidence intervals

## Output Files

- **Index Values**: Time series of index values by geography
- **Regression Results**: Detailed regression statistics
- **Quality Metrics**: Data validation results
- **Visualizations**: Time series plots and comparisons

## References

1. Bailey, M. J., Muth, R. F., & Nourse, H. O. (1963). A regression method for real estate price index construction. Journal of the American Statistical Association, 58(304), 933-942.

2. Case, K. E., & Shiller, R. J. (1987). Prices of single-family homes since 1970: New indexes for four cities. New England Economic Review, (Sep), 45-56.
"""
        
        output_path = self.reports_dir / "methodology.md"
        with open(output_path, 'w') as f:
            f.write(methodology)
            
        return output_path
        
    def _generate_html_report(
        self,
        index_df: pl.DataFrame,
        validation_results: Optional[Dict[str, QualityMetrics]],
        timestamp: str
    ) -> Path:
        """Generate HTML summary report."""
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(index_df)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RSAI Model Report - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .warning {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>RSAI Model Report</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>Summary Statistics</h2>
    <div class="metric">
        <h3>Coverage</h3>
        <ul>
            <li>Total Geographies: {summary_stats['total_geographies']}</li>
            <li>Time Period: {summary_stats['start_date']} to {summary_stats['end_date']}</li>
            <li>Total Observations: {summary_stats['total_observations']:,}</li>
        </ul>
    </div>
    
    <div class="metric">
        <h3>Index Performance</h3>
        <ul>
            <li>Average Annual Growth: {summary_stats['avg_annual_growth']:.2%}</li>
            <li>Volatility: {summary_stats['volatility']:.2%}</li>
            <li>Max Drawdown: {summary_stats['max_drawdown']:.2%}</li>
        </ul>
    </div>
"""
        
        # Add validation results if available
        if validation_results:
            html_content += """
    <h2>Data Quality</h2>
"""
            for name, metrics in validation_results.items():
                html_content += f"""
    <div class="metric">
        <h3>{name.title()} Validation</h3>
        <ul>
            <li>Total Records: {metrics.total_records:,}</li>
            <li>Valid Records: {metrics.valid_records:,} ({metrics.validity_score:.1%})</li>
            <li>Completeness Score: {metrics.completeness_score:.1%}</li>
            <li>Overall Score: {metrics.overall_score:.1%}</li>
        </ul>
"""
                if metrics.issues:
                    html_content += """
        <div class="warning">
            <strong>Issues:</strong>
            <ul>
"""
                    for issue in metrics.issues[:5]:  # Show first 5 issues
                        html_content += f"                <li>{issue}</li>\n"
                    html_content += """
            </ul>
        </div>
"""
                html_content += "    </div>\n"
                
        # Add geographic breakdown
        geo_summary = index_df.group_by("geography_level").agg([
            pl.count().alias("num_geographies"),
            pl.col("num_pairs").sum().alias("total_pairs"),
            pl.col("index_value").mean().alias("avg_index")
        ])
        
        html_content += """
    <h2>Geographic Summary</h2>
    <table>
        <tr>
            <th>Geography Level</th>
            <th>Count</th>
            <th>Total Pairs</th>
            <th>Average Index</th>
        </tr>
"""
        
        for row in geo_summary.iter_rows(named=True):
            html_content += f"""
        <tr>
            <td>{row['geography_level']}</td>
            <td>{row['num_geographies']}</td>
            <td>{row['total_pairs']:,}</td>
            <td>{row['avg_index']:.1f}</td>
        </tr>
"""
            
        html_content += """
    </table>
</body>
</html>
"""
        
        output_path = self.reports_dir / f"summary_report_{timestamp}.html"
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        return output_path
        
    def _generate_markdown_report(
        self,
        index_df: pl.DataFrame,
        validation_results: Optional[Dict[str, QualityMetrics]],
        timestamp: str
    ) -> Path:
        """Generate Markdown summary report."""
        summary_stats = self._calculate_summary_statistics(index_df)
        
        md_content = f"""# RSAI Model Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Statistics

### Coverage
- Total Geographies: {summary_stats['total_geographies']}
- Time Period: {summary_stats['start_date']} to {summary_stats['end_date']}
- Total Observations: {summary_stats['total_observations']:,}

### Index Performance
- Average Annual Growth: {summary_stats['avg_annual_growth']:.2%}
- Volatility: {summary_stats['volatility']:.2%}
- Max Drawdown: {summary_stats['max_drawdown']:.2%}
"""
        
        if validation_results:
            md_content += "\n## Data Quality\n\n"
            for name, metrics in validation_results.items():
                md_content += f"""### {name.title()} Validation
- Total Records: {metrics.total_records:,}
- Valid Records: {metrics.valid_records:,} ({metrics.validity_score:.1%})
- Completeness Score: {metrics.completeness_score:.1%}
- Overall Score: {metrics.overall_score:.1%}
"""
                if metrics.issues:
                    md_content += "\n**Issues:**\n"
                    for issue in metrics.issues[:5]:
                        md_content += f"- {issue}\n"
                    md_content += "\n"
                    
        output_path = self.reports_dir / f"summary_report_{timestamp}.md"
        with open(output_path, 'w') as f:
            f.write(md_content)
            
        return output_path
        
    def _calculate_summary_statistics(self, index_df: pl.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics from index data."""
        # Basic coverage stats
        stats = {
            "total_geographies": index_df["geography_id"].n_unique(),
            "start_date": index_df["period"].min(),
            "end_date": index_df["period"].max(),
            "total_observations": index_df["num_pairs"].sum()
        }
        
        # Calculate returns for performance metrics
        returns_df = index_df.sort(["geography_id", "period"]).with_columns([
            (pl.col("index_value").pct_change().over("geography_id")).alias("return")
        ])
        
        # Annual growth (geometric mean)
        if self.config.frequency == "monthly":
            periods_per_year = 12
        elif self.config.frequency == "quarterly":
            periods_per_year = 4
        else:
            periods_per_year = 252
            
        avg_return = returns_df["return"].mean()
        stats["avg_annual_growth"] = (1 + avg_return) ** periods_per_year - 1
        
        # Volatility
        stats["volatility"] = returns_df["return"].std() * np.sqrt(periods_per_year)
        
        # Max drawdown
        cumulative_returns = returns_df.group_by("geography_id").agg([
            pl.col("index_value").max().alias("peak"),
            pl.col("index_value").min().alias("trough")
        ])
        
        drawdowns = (cumulative_returns["trough"] / cumulative_returns["peak"] - 1)
        stats["max_drawdown"] = drawdowns.min() if len(drawdowns) > 0 else 0
        
        return stats
        
    def _calculate_growth_rates(self, index_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate growth rates for visualization."""
        return index_df.sort(["geography_id", "period"]).with_columns([
            pl.col("index_value").pct_change().over("geography_id").alias("period_growth"),
            pl.col("index_value").pct_change(12).over("geography_id").alias("annual_growth")
        ])
        
    def _create_interactive_time_series(self, index_df: pl.DataFrame) -> go.Figure:
        """Create interactive time series plot using Plotly."""
        fig = go.Figure()
        
        for geo_id in index_df["geography_id"].unique():
            geo_data = index_df.filter(pl.col("geography_id") == geo_id).sort("period")
            
            fig.add_trace(go.Scatter(
                x=geo_data["period"].to_list(),
                y=geo_data["index_value"].to_list(),
                mode='lines',
                name=str(geo_id),
                hovertemplate='%{x}<br>Index: %{y:.1f}<extra></extra>'
            ))
            
        fig.update_layout(
            title="RSAI Price Index Time Series",
            xaxis_title="Date",
            yaxis_title="Index Value (Base = 100)",
            hovermode='x unified'
        )
        
        return fig
        
    def _create_static_time_series(self, index_df: pl.DataFrame) -> Figure:
        """Create static time series plot using matplotlib."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for geo_id in index_df["geography_id"].unique()[:10]:  # Limit to 10 for clarity
            geo_data = index_df.filter(pl.col("geography_id") == geo_id).sort("period")
            ax.plot(
                geo_data["period"].to_list(),
                geo_data["index_value"].to_list(),
                label=str(geo_id)
            )
            
        ax.set_xlabel("Date")
        ax.set_ylabel("Index Value (Base = 100)")
        ax.set_title("RSAI Price Index Time Series")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        return fig
        
    def _create_growth_heatmap(self, growth_df: pl.DataFrame) -> Figure:
        """Create heatmap of growth rates."""
        # Pivot data for heatmap
        pivot_df = growth_df.pivot(
            values="annual_growth",
            index="geography_id",
            columns="period",
            aggregate_function="mean"
        )
        
        # Convert to numpy array
        data = pivot_df.drop("geography_id").to_numpy()
        
        # Check if we have valid data
        if data.size == 0 or np.isnan(data).all():
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.text(0.5, 0.5, 'Insufficient data for growth heatmap', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Annual Growth Rates by Geography")
            return fig
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(
            data,
            xticklabels=[d if isinstance(d, str) else d.strftime("%Y-%m") for d in pivot_df.columns[1:]],
            yticklabels=pivot_df["geography_id"].to_list(),
            cmap="RdBu_r",
            center=0,
            cbar_kws={'label': 'Annual Growth Rate'},
            ax=ax
        )
        
        ax.set_title("Annual Growth Rates by Geography")
        ax.set_xlabel("Period")
        ax.set_ylabel("Geography")
        
        return fig
        
    def _create_geographic_comparison(self, index_df: pl.DataFrame) -> Figure:
        """Create geographic comparison plot."""
        # Get latest values by geography
        latest_values = index_df.group_by("geography_id").agg([
            pl.col("index_value").last().alias("latest_index"),
            pl.col("period").last().alias("latest_period")
        ]).sort("latest_index", descending=True).head(20)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(
            latest_values["geography_id"].to_list(),
            latest_values["latest_index"].to_list()
        )
        
        ax.set_xlabel("Index Value")
        ax.set_ylabel("Geography")
        ax.set_title(f"Latest Index Values by Geography")
        ax.axvline(x=100, color='red', linestyle='--', alpha=0.5)
        
        return fig