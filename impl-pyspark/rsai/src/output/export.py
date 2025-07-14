"""
Output generation and export using PySpark.

This module handles exporting RSAI model results in various formats
including index values, reports, and visualizations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import pandas as pd

from rsai.src.data.models import (
    IndexValue,
    BMNRegressionResult,
    QualityMetrics,
    GeographyLevel,
    RSAIConfig
)

logger = logging.getLogger(__name__)


class OutputExporter:
    """Handles exporting RSAI results using PySpark."""
    
    def __init__(
        self,
        spark: SparkSession,
        output_dir: Union[str, Path],
        config: RSAIConfig
    ):
        """
        Initialize output exporter.
        
        Args:
            spark: SparkSession instance
            output_dir: Base directory for outputs
            config: RSAI configuration
        """
        self.spark = spark
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
        index_df: DataFrame,
        format: str = "parquet",
        filename_prefix: str = "index"
    ) -> Path:
        """
        Export index values to file.
        
        Args:
            index_df: DataFrame with index values
            format: Output format ('parquet', 'csv', 'json')
            filename_prefix: Prefix for output filename
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.{format}"
        output_path = self.indices_dir / filename
        
        # Order columns consistently
        index_df = index_df.select(
            "geography_level",
            "geography_id",
            "period",
            "index_value",
            "num_pairs",
            "num_properties",
            "median_price"
        ).orderBy("geography_level", "geography_id", "period")
        
        # Export based on format
        if format == "parquet":
            index_df.write.mode("overwrite").parquet(str(output_path))
        elif format == "csv":
            index_df.coalesce(1).write.mode("overwrite").csv(
                str(output_path), header=True
            )
        elif format == "json":
            index_df.coalesce(1).write.mode("overwrite").json(str(output_path))
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Exported {index_df.count()} index values to {output_path}")
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
        index_df: DataFrame,
        validation_results: Optional[Dict[str, QualityMetrics]] = None,
        format: str = "html"
    ) -> Path:
        """
        Generate comprehensive summary report.
        
        Args:
            index_df: DataFrame with all index values
            validation_results: Optional validation results
            format: Report format ('html', 'markdown')
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(index_df)
        
        if format == "html":
            report_path = self._generate_html_report(
                index_df, summary_stats, validation_results, timestamp
            )
        elif format == "markdown":
            report_path = self._generate_markdown_report(
                index_df, summary_stats, validation_results, timestamp
            )
        else:
            raise ValueError(f"Unsupported report format: {format}")
            
        logger.info(f"Generated summary report: {report_path}")
        return report_path
        
    def create_index_plots(
        self,
        index_df: DataFrame,
        geography_ids: Optional[List[str]] = None,
        save_plots: bool = True
    ) -> Dict[str, Path]:
        """
        Create visualizations of index values.
        
        Args:
            index_df: DataFrame with index values
            geography_ids: List of geography IDs to plot
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        plot_paths = {}
        
        # Convert to Pandas for plotting
        if geography_ids:
            plot_data = index_df.filter(
                F.col("geography_id").isin(geography_ids)
            ).toPandas()
        else:
            # Sample if too many geographies
            n_geos = index_df.select("geography_id").distinct().count()
            if n_geos > 10:
                sample_geos = index_df.select("geography_id").distinct().limit(10).toPandas()
                plot_data = index_df.filter(
                    F.col("geography_id").isin(sample_geos["geography_id"].tolist())
                ).toPandas()
            else:
                plot_data = index_df.toPandas()
                
        # Time series plot
        fig = self._create_time_series_plot(plot_data)
        if save_plots:
            plot_path = self.plots_dir / "index_time_series.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plot_paths["time_series"] = plot_path
            
        # Growth rate heatmap
        if len(plot_data["geography_id"].unique()) > 1:
            fig = self._create_growth_heatmap(plot_data)
            if save_plots:
                plot_path = self.plots_dir / "growth_heatmap.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                plot_paths["growth_heatmap"] = plot_path
                
        # Geographic comparison
        fig = self._create_geographic_comparison(plot_data)
        if save_plots:
            plot_path = self.plots_dir / "geographic_comparison.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plot_paths["geographic_comparison"] = plot_path
            
        return plot_paths
        
    def export_for_tableau(
        self,
        index_df: DataFrame,
        repeat_sales_df: Optional[DataFrame] = None
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
        tableau_df = index_df.withColumn(
            "year", F.year("period")
        ).withColumn(
            "month", F.month("period")
        ).withColumn(
            "quarter", F.quarter("period")
        )
        
        # Calculate year-over-year growth
        window = Window.partitionBy("geography_id").orderBy("period")
        tableau_df = tableau_df.withColumn(
            "yoy_growth",
            (F.col("index_value") / F.lag("index_value", 12).over(window) - 1) * 100
        )
        
        # Add repeat sales summary if available
        if repeat_sales_df is not None:
            if self.config.frequency == "monthly":
                period_col = F.date_trunc("month", F.col("sale2_date"))
            else:
                period_col = F.date_trunc("quarter", F.col("sale2_date"))
            
            # Check if repeat_sales_df has geography_id column, if not try to infer it
            if "geography_id" not in repeat_sales_df.columns:
                # Try different geography columns that might exist
                geo_column = None
                for col in ["tract", "county", "cbsa", "state"]:
                    if col in repeat_sales_df.columns:
                        geo_column = col
                        break
                
                if geo_column:
                    # Add period and group by the available geography column
                    sales_summary = repeat_sales_df.withColumn(
                        "period", period_col
                    ).groupBy(geo_column, "period").agg(
                        F.mean("price_ratio").alias("avg_price_ratio"),
                        F.mean("holding_period_days").alias("avg_holding_days"),
                        F.mean("annualized_return").alias("avg_return")
                    ).withColumnRenamed(geo_column, "geography_id")
                    
                    tableau_df = tableau_df.join(
                        sales_summary,
                        on=["geography_id", "period"],
                        how="left"
                    )
                else:
                    # Skip repeat sales summary if no geography column found
                    logger.warning("No geography column found in repeat sales data, skipping summary")
            else:
                # Standard case where geography_id exists
                sales_summary = repeat_sales_df.withColumn(
                    "period", period_col
                ).groupBy("geography_id", "period").agg(
                    F.mean("price_ratio").alias("avg_price_ratio"),
                    F.mean("holding_period_days").alias("avg_holding_days"),
                    F.mean("annualized_return").alias("avg_return")
                )
                
                tableau_df = tableau_df.join(
                    sales_summary,
                    on=["geography_id", "period"],
                    how="left"
                )
            
        # Export as CSV for Tableau
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.data_dir / f"tableau_data_{timestamp}.csv"
        
        tableau_df.coalesce(1).write.mode("overwrite").csv(
            str(output_path), header=True
        )
        
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
- **Regression Method**: Bailey-Muth-Nourse (BMN) with MLlib
- **Computing Framework**: Apache Spark {self.spark.version}
- **Weighting Scheme**: {self.config.weighting_scheme.value if hasattr(self.config.weighting_scheme, 'value') else self.config.weighting_scheme}
- **Time Frequency**: {self.config.frequency}
- **Geographic Levels**: {', '.join([gl.value if hasattr(gl, 'value') else gl for gl in self.config.geography_levels])}

## Data Filters

- **Price Range**: ${self.config.min_price:,.0f} - ${self.config.max_price:,.0f}
- **Maximum Holding Period**: {self.config.max_holding_period_years} years
- **Minimum Pairs per Geography**: {self.config.min_pairs_threshold}
- **Outlier Threshold**: {self.config.outlier_std_threshold} standard deviations

## Spark Configuration

- **Application Name**: {self.config.spark_app_name}
- **Master**: {self.config.spark_master}
- **Executor Memory**: {self.config.spark_executor_memory}
- **Driver Memory**: {self.config.spark_driver_memory}

## Implementation Details

This implementation uses Apache Spark for distributed computing:

1. **Data Ingestion**: Parallel loading and filtering of transaction data
2. **Repeat Sales Identification**: Window functions for efficient pair matching
3. **Geographic Clustering**: MLlib K-means for supertract generation
4. **BMN Regression**: MLlib LinearRegression with custom feature engineering
5. **Aggregation**: Distributed weighted averaging across geographic levels

## Output Files

- **Index Values**: Time series of index values by geography (Parquet format)
- **Regression Results**: Detailed regression statistics (JSON format)
- **Visualizations**: Time series plots and geographic comparisons
- **Tableau Export**: Pre-formatted data for business intelligence tools

## References

1. Bailey, M. J., Muth, R. F., & Nourse, H. O. (1963). A regression method for real estate price index construction.
2. Apache Spark MLlib Documentation: https://spark.apache.org/docs/latest/ml-guide.html
"""
        
        output_path = self.reports_dir / "methodology.md"
        with open(output_path, 'w') as f:
            f.write(methodology)
            
        return output_path
        
    def _calculate_summary_statistics(self, index_df: DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics from index data."""
        stats = {}
        
        # Basic coverage stats
        coverage = index_df.agg(
            F.countDistinct("geography_id").alias("total_geographies"),
            F.min("period").alias("start_date"),
            F.max("period").alias("end_date"),
            F.sum("num_pairs").alias("total_observations")
        ).collect()[0]
        
        stats.update(coverage.asDict())
        
        # Calculate returns
        window = Window.partitionBy("geography_id").orderBy("period")
        returns_df = index_df.withColumn(
            "return",
            (F.col("index_value") / F.lag("index_value").over(window) - 1)
        )
        
        # Performance metrics
        perf_stats = returns_df.agg(
            F.mean("return").alias("avg_return"),
            F.stddev("return").alias("volatility")
        ).collect()[0]
        
        # Annualize
        if self.config.frequency == "monthly":
            periods_per_year = 12
        elif self.config.frequency == "quarterly":
            periods_per_year = 4
        else:
            periods_per_year = 252
            
        stats["avg_annual_growth"] = (1 + perf_stats["avg_return"]) ** periods_per_year - 1
        stats["volatility"] = perf_stats["volatility"] * (periods_per_year ** 0.5)
        
        return stats
        
    def _generate_html_report(
        self,
        index_df: DataFrame,
        summary_stats: Dict[str, Any],
        validation_results: Optional[Dict[str, QualityMetrics]],
        timestamp: str
    ) -> Path:
        """Generate HTML summary report."""
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
    </style>
</head>
<body>
    <h1>RSAI Model Report - PySpark Implementation</h1>
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
            <li>Average Annual Growth: {summary_stats.get('avg_annual_growth', 0):.2%}</li>
            <li>Volatility: {summary_stats.get('volatility', 0):.2%}</li>
        </ul>
    </div>
"""
        
        # Add validation results if available
        if validation_results:
            html_content += "\n    <h2>Data Quality</h2>\n"
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
    </div>
"""
                
        # Add geographic breakdown
        geo_summary = index_df.groupBy("geography_level").agg(
            F.countDistinct("geography_id").alias("count"),
            F.sum("num_pairs").alias("total_pairs"),
            F.mean("index_value").alias("avg_index")
        ).toPandas()
        
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
        
        for _, row in geo_summary.iterrows():
            html_content += f"""
        <tr>
            <td>{row['geography_level']}</td>
            <td>{row['count']}</td>
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
        index_df: DataFrame,
        summary_stats: Dict[str, Any],
        validation_results: Optional[Dict[str, QualityMetrics]],
        timestamp: str
    ) -> Path:
        """Generate Markdown summary report."""
        md_content = f"""# RSAI Model Report - PySpark Implementation

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Statistics

### Coverage
- Total Geographies: {summary_stats['total_geographies']}
- Time Period: {summary_stats['start_date']} to {summary_stats['end_date']}
- Total Observations: {summary_stats['total_observations']:,}

### Index Performance
- Average Annual Growth: {summary_stats.get('avg_annual_growth', 0):.2%}
- Volatility: {summary_stats.get('volatility', 0):.2%}
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
                
        output_path = self.reports_dir / f"summary_report_{timestamp}.md"
        with open(output_path, 'w') as f:
            f.write(md_content)
            
        return output_path
        
    def _create_time_series_plot(self, data) -> Figure:
        """Create time series plot of index values."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for geo_id in data["geography_id"].unique():
            geo_data = data[data["geography_id"] == geo_id]
            ax.plot(geo_data["period"], geo_data["index_value"], label=str(geo_id))
            
        ax.set_xlabel("Date")
        ax.set_ylabel("Index Value (Base = 100)")
        ax.set_title("RSAI Price Index Time Series")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        return fig
        
    def _create_growth_heatmap(self, data) -> Figure:
        """Create heatmap of growth rates."""
        # Remove duplicates first by taking the first occurrence for each (geography_id, period)
        data_clean = data.drop_duplicates(subset=["geography_id", "period"], keep="first")
        
        # Calculate month-over-month growth
        growth_data = []
        
        for geo_id in data_clean["geography_id"].unique():
            geo_data = data_clean[data_clean["geography_id"] == geo_id].sort_values("period")
            geo_data = geo_data.copy()  # Avoid SettingWithCopyWarning
            geo_data["growth"] = geo_data["index_value"].pct_change() * 100
            growth_data.append(geo_data)
            
        growth_df = pd.concat(growth_data)
        
        # Additional safety check - remove any remaining duplicates before pivot
        growth_df = growth_df.drop_duplicates(subset=["geography_id", "period"], keep="first")
        
        # Pivot for heatmap
        pivot_df = growth_df.pivot(
            index="geography_id",
            columns="period",
            values="growth"
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(
            pivot_df,
            cmap="RdBu_r",
            center=0,
            cbar_kws={'label': 'Growth Rate (%)'},
            ax=ax
        )
        
        ax.set_title("Growth Rates by Geography")
        ax.set_xlabel("Period")
        ax.set_ylabel("Geography")
        
        return fig
        
    def _create_geographic_comparison(self, data) -> Figure:
        """Create geographic comparison plot."""
        # Get latest values by geography
        latest_values = data.groupby("geography_id")["index_value"].last().sort_values(ascending=False)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        latest_values.head(20).plot(kind="barh", ax=ax)
        
        ax.set_xlabel("Index Value")
        ax.set_ylabel("Geography")
        ax.set_title("Latest Index Values by Geography")
        ax.axvline(x=100, color='red', linestyle='--', alpha=0.5)
        
        return fig