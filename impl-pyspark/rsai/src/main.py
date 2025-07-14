"""
Main entry point for RSAI model execution using PySpark.

This module orchestrates the entire repeat sales index calculation pipeline
from data ingestion through index generation and output.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Union, Any
from datetime import datetime
import json

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from rsai.src.data.models import RSAIConfig, GeographyLevel, WeightingScheme
from rsai.src.data.ingestion import DataIngestion
from rsai.src.data.validation import DataValidator
from rsai.src.geography.supertract import SupertractGenerator
from rsai.src.index.bmn_regression import BMNRegression
from rsai.src.index.aggregation import IndexAggregator
from rsai.src.index.weights import WeightCalculator
from rsai.src.output.export import OutputExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RSAIPipeline:
    """Main pipeline for RSAI model execution."""
    
    def __init__(
        self,
        config_path: Union[str, Path],
        spark: Optional[SparkSession] = None
    ):
        """
        Initialize RSAI pipeline.
        
        Args:
            config_path: Path to configuration file
            spark: Optional SparkSession (will create if not provided)
        """
        self.config = self._load_config(config_path)
        self.spark = spark or self._create_spark_session()
        
        # Initialize components
        self.data_ingestion = DataIngestion(self.spark, self.config)
        self.validator = DataValidator(self.spark, self.config)
        self.supertract_gen = SupertractGenerator(self.spark, self.config)
        self.weight_calc = WeightCalculator(self.spark, self.config.weighting_scheme)
        self.bmn_regression = BMNRegression(
            self.spark,
            frequency=self.config.frequency,
            min_pairs_per_period=self.config.min_pairs_threshold
        )
        self.aggregator = IndexAggregator(self.spark)
        self.exporter = OutputExporter(
            self.spark,
            self.config.output_dir,
            self.config
        )
        
    def _load_config(self, config_path: Union[str, Path]) -> RSAIConfig:
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            
        return RSAIConfig(**config_data)
        
    def _create_spark_session(self) -> SparkSession:
        """Create and configure Spark session."""
        builder = SparkSession.builder \
            .appName(self.config.spark_app_name) \
            .master(self.config.spark_master)
            
        # Set memory configurations
        builder = builder \
            .config("spark.executor.memory", self.config.spark_executor_memory) \
            .config("spark.driver.memory", self.config.spark_driver_memory) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            
        # Add any additional Spark configurations
        for key, value in self.config.spark_config.items():
            builder = builder.config(key, value)
            
        spark = builder.getOrCreate()
        
        # Set log level
        spark.sparkContext.setLogLevel("WARN")
        
        return spark
        
    def run(self) -> Dict[str, any]:
        """
        Execute the complete RSAI pipeline.
        
        Returns:
            Dictionary with pipeline results and metrics
        """
        logger.info("Starting RSAI pipeline execution")
        start_time = datetime.now()
        
        try:
            # Step 1: Data Ingestion
            logger.info("Step 1: Loading and preparing data")
            transactions_df = self.data_ingestion.load_transactions(
                self.config.input_files['transactions']
            )
            properties_df = self.data_ingestion.load_properties(
                self.config.input_files['properties']
            )
            
            # Step 2: Identify Repeat Sales
            logger.info("Step 2: Identifying repeat sales")
            repeat_sales_df = self.data_ingestion.identify_repeat_sales(
                transactions_df
            )
            
            # Add geographic data
            repeat_sales_df = self.data_ingestion.merge_geographic_data(
                repeat_sales_df,
                properties_df
            )
            
            # Step 3: Data Validation
            logger.info("Step 3: Validating data quality")
            validation_results = {}
            validation_results['transactions'] = self.validator.validate_transactions(
                transactions_df
            )
            validation_results['repeat_sales'] = self.validator.validate_repeat_sales(
                repeat_sales_df
            )
            
            # Step 4: Filter Outliers
            logger.info("Step 4: Filtering outliers")
            repeat_sales_df = self.data_ingestion.filter_outliers(repeat_sales_df)
            
            # Step 5: Calculate Weights
            logger.info("Step 5: Calculating observation weights")
            repeat_sales_df = self.weight_calc.calculate_weights(
                repeat_sales_df,
                weight_col="weight"
            )
            
            # Step 6: Generate Supertracts (if configured)
            supertract_mapping = None
            if GeographyLevel.SUPERTRACT in self.config.geography_levels:
                logger.info("Step 6: Generating supertracts")
                
                # Prepare tract data
                tract_df = self.supertract_gen.prepare_tract_data(
                    repeat_sales_df,
                    properties_df
                )
                
                # Generate supertracts
                supertract_df, supertract_mapping = self.supertract_gen.generate_supertracts(
                    tract_df,
                    method=self.config.clustering_method,
                    min_pairs=self.config.min_pairs_threshold
                )
                
                # Add supertract IDs to repeat sales
                repeat_sales_df = repeat_sales_df.join(
                    supertract_mapping.select("tract", "supertract"),
                    on="tract",
                    how="left"
                )
            
            # Step 7: Fit BMN Regression Models
            logger.info("Step 7: Fitting BMN regression models")
            regression_results = {}
            
            # Fit models for each geography level
            for geo_level in self.config.geography_levels:
                if geo_level == GeographyLevel.TRACT:
                    geo_col = "tract"
                elif geo_level == GeographyLevel.SUPERTRACT:
                    geo_col = "supertract"
                elif geo_level == GeographyLevel.COUNTY:
                    geo_col = "county"
                elif geo_level == GeographyLevel.CBSA:
                    geo_col = "cbsa"
                elif geo_level == GeographyLevel.STATE:
                    geo_col = "state"
                else:
                    continue
                    
                # Skip if column doesn't exist
                if geo_col not in repeat_sales_df.columns:
                    logger.warning(f"Skipping {geo_level.value}: column {geo_col} not found")
                    continue
                    
                # Fit models (weights are already in repeat_sales_df)
                geo_results = self.bmn_regression.fit_multiple_geographies(
                    repeat_sales_df,
                    geo_col,
                    geo_level,
                    weights_df=None,  # Weights already in repeat_sales_df
                    min_pairs=self.config.min_pairs_threshold
                )
                
                regression_results.update(geo_results)
                
            # Step 8: Create Geography Mappings
            logger.info("Step 8: Creating geography mappings")
            geography_mappings = self._create_geography_mappings(
                properties_df,
                supertract_mapping
            )
            
            # Step 9: Aggregate Indices
            logger.info("Step 9: Aggregating indices to higher geographic levels")
            
            # Convert regression results to DataFrames
            base_indices = {}
            for geo_id, result in regression_results.items():
                # Create DataFrame from index values
                index_data = []
                for iv in result.index_values:
                    index_data.append({
                        "geography_level": iv.geography_level.value if hasattr(iv.geography_level, 'value') else iv.geography_level,
                        "geography_id": iv.geography_id,
                        "period": iv.period,
                        "index_value": iv.index_value,
                        "num_pairs": iv.num_pairs,
                        "num_properties": iv.num_properties,
                        "num_submarkets": 1,  # Base indices represent 1 geography unit
                        "median_price": iv.median_price
                    })
                
                if index_data:
                    df = self.spark.createDataFrame(index_data)
                    base_indices[geo_id] = df
                    
            # Create hierarchical indices
            all_indices = self.aggregator.create_hierarchical_indices(
                base_indices,
                geography_mappings,
                self.config.geography_levels,
                self.config.weighting_scheme
            )
            
            # Step 10: Export Results
            logger.info("Step 10: Exporting results")
            
            # Combine all indices into single DataFrame
            combined_indices = None
            for level, df in all_indices.items():
                if combined_indices is None:
                    combined_indices = df
                else:
                    combined_indices = combined_indices.union(df)
            
            # Check if we have any indices to export
            if combined_indices is None:
                logger.warning("No indices were created - skipping index export")
                index_path = None
            else:
                # Export index values
                index_path = self.exporter.export_index_values(
                    combined_indices,
                    format="parquet"
                )
            
            # Export regression results
            regression_path = self.exporter.export_regression_results(
                regression_results,
                include_diagnostics=True
            )
            
            # Generate summary report
            if combined_indices is not None:
                report_path = self.exporter.generate_summary_report(
                    combined_indices,
                    validation_results,
                    format="html"
                )
            else:
                logger.warning("No indices available - skipping summary report")
                report_path = None
            
            # Create visualizations
            if combined_indices is not None:
                plot_paths = self.exporter.create_index_plots(
                    combined_indices,
                    save_plots=True
                )
                
                # Export for Tableau
                tableau_path = self.exporter.export_for_tableau(
                    combined_indices,
                    repeat_sales_df
                )
            else:
                logger.warning("No indices available - skipping plots and Tableau export")
                plot_paths = {}
                tableau_path = None
            
            # Create methodology document
            methodology_path = self.exporter.create_methodology_document()
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Prepare results
            results = {
                "status": "success",
                "execution_time_seconds": execution_time,
                "total_transactions": transactions_df.count(),
                "total_repeat_sales": repeat_sales_df.count(),
                "regression_models_fitted": len(regression_results),
                "geography_levels_processed": [
                    level.value if hasattr(level, 'value') else level for level in all_indices.keys()
                ],
                "output_files": {
                    "indices": str(index_path) if index_path else None,
                    "regression_results": str(regression_path),
                    "summary_report": str(report_path) if report_path else None,
                    "plots": {k: str(v) for k, v in plot_paths.items()},
                    "tableau_data": str(tableau_path) if tableau_path else None,
                    "methodology": str(methodology_path)
                },
                "validation_results": validation_results
            }
            
            logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
            return results
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Pipeline failed: {str(e)}")
            logger.error(f"Full traceback:\n{error_traceback}")
            return {
                "status": "failed",
                "error": str(e),
                "traceback": error_traceback,
                "execution_time_seconds": (datetime.now() - start_time).total_seconds()
            }
            
    def _create_geography_mappings(
        self,
        properties_df,
        supertract_mapping=None
    ) -> Dict[str, any]:
        """Create mappings between geography levels."""
        mappings = {}
        
        # Tract to County
        if "tract" in properties_df.columns and "county" in properties_df.columns:
            mappings["tract_to_county"] = properties_df.select(
                "tract", "county"
            ).distinct().filter(
                properties_df.tract.isNotNull() &
                properties_df.county.isNotNull()
            )
            
        # Supertract to County
        if supertract_mapping is not None:
            # Join with properties to get county
            mappings["supertract_to_county"] = supertract_mapping.join(
                properties_df.select("tract", "county").distinct(),
                on="tract"
            ).select("supertract", "county").distinct()
            
        # County to CBSA
        if "county" in properties_df.columns and "cbsa" in properties_df.columns:
            mappings["county_to_cbsa"] = properties_df.select(
                "county", "cbsa"
            ).distinct().filter(
                properties_df.county.isNotNull() &
                properties_df.cbsa.isNotNull()
            )
            
        # CBSA to State
        if "cbsa" in properties_df.columns and "state" in properties_df.columns:
            mappings["cbsa_to_state"] = properties_df.select(
                "cbsa", "state"
            ).distinct().filter(
                properties_df.cbsa.isNotNull() &
                properties_df.state.isNotNull()
            )
            
        # State to National
        if "state" in properties_df.columns:
            states = properties_df.select("state").distinct().filter(
                properties_df.state.isNotNull()
            )
            mappings["state_to_national"] = states.withColumn(
                "national", F.lit("US")
            )
            
        return mappings
        
    def stop(self):
        """Stop the Spark session."""
        if self.spark:
            self.spark.stop()


def main():
    """Main entry point for command line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run RSAI model pipeline with PySpark"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "text"],
        help="Format for result output"
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = RSAIPipeline(args.config)
    
    try:
        results = pipeline.run()
        
        # Output results
        if args.output_format == "json":
            print(json.dumps(results, indent=2, default=str))
        else:
            print(f"Pipeline Status: {results['status']}")
            print(f"Execution Time: {results['execution_time_seconds']:.2f} seconds")
            if results['status'] == 'success':
                print(f"Total Transactions: {results['total_transactions']:,}")
                print(f"Total Repeat Sales: {results['total_repeat_sales']:,}")
                print(f"Models Fitted: {results['regression_models_fitted']}")
                print("Output Files:")
                for key, value in results['output_files'].items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"Error: {results.get('error', 'Unknown error')}")
                
    finally:
        pipeline.stop()
        
    return 0 if results.get('status') == 'success' else 1


if __name__ == "__main__":
    sys.exit(main())