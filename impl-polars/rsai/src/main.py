"""
Main pipeline orchestration for RSAI model.

This module coordinates the entire RSAI pipeline from data ingestion
to index calculation and output generation.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
import yaml
import argparse

import polars as pl
from pydantic import BaseModel, ValidationError

from rsai.src.data.models import (
    RSAIConfig,
    GeographyLevel,
    WeightingScheme,
    TransactionType
)
from rsai.src.data.ingestion import DataIngestion
from rsai.src.data.validation import DataValidator
from rsai.src.geography.supertract import SupertractGenerator
from rsai.src.index.bmn_regression import BMNRegression
from rsai.src.index.weights import WeightCalculator
from rsai.src.index.aggregation import IndexAggregator
from rsai.src.output.export import OutputExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RSAIPipeline:
    """Main pipeline for RSAI model execution."""
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results = {}
        
    def _load_config(self) -> RSAIConfig:
        """Load and validate configuration."""
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Convert to RSAIConfig
        return RSAIConfig(
            start_date=date.fromisoformat(config_dict.get('start_date', '2020-01-01')),
            end_date=date.fromisoformat(config_dict.get('end_date', '2023-12-31')),
            frequency=config_dict.get('model', {}).get('time', {}).get('frequency', 'monthly'),
            geography_levels=[
                GeographyLevel(level) for level in 
                config_dict.get('model', {}).get('geography_levels', ['tract', 'county', 'msa'])
            ],
            weighting_scheme=WeightingScheme(
                config_dict.get('model', {}).get('weighting_scheme', 'bmn')
            ),
            min_pairs_threshold=config_dict.get('model', {}).get('min_pairs_threshold', 10),
            max_holding_period_years=config_dict.get('model', {}).get('max_holding_period_years', 20),
            min_price=config_dict.get('data', {}).get('quality', {}).get('min_price', 10000),
            max_price=config_dict.get('data', {}).get('quality', {}).get('max_price', 10000000),
            outlier_std_threshold=config_dict.get('preprocessing', {}).get('outliers', {}).get('zscore_threshold', 3.0),
            output_format=config_dict.get('output', {}).get('format', 'parquet'),
            include_diagnostics=config_dict.get('output', {}).get('include_diagnostics', True)
        )
        
    def run(
        self,
        transactions_path: Union[str, Path],
        properties_path: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "output"
    ) -> Dict[str, Any]:
        """
        Run the complete RSAI pipeline.
        
        Args:
            transactions_path: Path to transactions data
            properties_path: Optional path to properties data
            output_dir: Directory for outputs
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting RSAI pipeline")
        start_time = datetime.now()
        
        try:
            # Step 1: Data Ingestion
            logger.info("Step 1: Data Ingestion")
            transactions_df, properties_df, repeat_sales_df = self._ingest_data(
                transactions_path, properties_path
            )
            
            # Step 2: Data Validation
            logger.info("Step 2: Data Validation")
            validation_results = self._validate_data(
                transactions_df, repeat_sales_df, properties_df
            )
            
            # Step 3: Geographic Processing
            logger.info("Step 3: Geographic Processing")
            geography_mappings = self._process_geography(
                transactions_df, properties_df, repeat_sales_df
            )
            
            # Step 4: Calculate Weights
            logger.info("Step 4: Calculate Weights")
            weighted_df = self._calculate_weights(repeat_sales_df)
            
            # Step 5: Run BMN Regression
            logger.info("Step 5: Run BMN Regression")
            regression_results = self._run_regression(weighted_df)
            
            # Step 6: Aggregate Indices
            logger.info("Step 6: Aggregate Indices")
            aggregated_indices = self._aggregate_indices(
                regression_results, geography_mappings
            )
            
            # Step 7: Generate Outputs
            logger.info("Step 7: Generate Outputs")
            output_paths = self._generate_outputs(
                aggregated_indices,
                regression_results,
                validation_results,
                repeat_sales_df,
                output_dir
            )
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare results summary
            self.results = {
                "status": "success",
                "execution_time_seconds": execution_time,
                "data_summary": {
                    "total_transactions": len(transactions_df),
                    "total_properties": len(properties_df) if properties_df is not None else None,
                    "total_repeat_sales": len(repeat_sales_df),
                    "weighted_pairs": len(weighted_df)
                },
                "validation_summary": {
                    name: {
                        "overall_score": metrics.overall_score,
                        "issues_count": len(metrics.issues)
                    }
                    for name, metrics in validation_results.items()
                },
                "index_summary": {
                    "geographies_processed": len(regression_results),
                    "total_index_values": sum(
                        len(result.index_values) for result in regression_results.values()
                    ),
                    "aggregation_levels": len(aggregated_indices)
                },
                "output_files": output_paths
            }
            
            logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.results = {
                "status": "failed",
                "error": str(e),
                "execution_time_seconds": (datetime.now() - start_time).total_seconds()
            }
            raise
            
        return self.results
        
    def _ingest_data(
        self,
        transactions_path: Union[str, Path],
        properties_path: Optional[Union[str, Path]]
    ) -> tuple[pl.DataFrame, Optional[pl.DataFrame], pl.DataFrame]:
        """Ingest and process data."""
        ingestion = DataIngestion(self.config)
        
        # Load transactions
        transactions_df = ingestion.load_transactions(transactions_path)
        logger.info(f"Loaded {len(transactions_df)} transactions")
        
        # Load properties if provided
        properties_df = None
        if properties_path:
            properties_df = ingestion.load_properties(properties_path)
            logger.info(f"Loaded {len(properties_df)} properties")
            
        # Identify repeat sales
        repeat_sales_df = ingestion.identify_repeat_sales()
        logger.info(f"Identified {len(repeat_sales_df)} repeat sales pairs")
        
        # Merge geographic data if available
        if properties_df is not None:
            repeat_sales_df = ingestion.merge_geographic_data()
            
        # Filter outliers
        repeat_sales_df = ingestion.filter_outliers(method="iqr")
        logger.info(f"Retained {len(repeat_sales_df)} pairs after outlier filtering")
        
        return transactions_df, properties_df, repeat_sales_df
        
    def _validate_data(
        self,
        transactions_df: pl.DataFrame,
        repeat_sales_df: pl.DataFrame,
        properties_df: Optional[pl.DataFrame]
    ) -> Dict[str, Any]:
        """Validate data quality."""
        validator = DataValidator(self.config)
        
        results = {}
        
        # Validate transactions
        results["transactions"] = validator.validate_transactions(transactions_df)
        
        # Validate repeat sales
        results["repeat_sales"] = validator.validate_repeat_sales(repeat_sales_df)
        
        # Validate geographic data if available
        if properties_df is not None:
            results["geographic"] = validator.validate_geographic_data(properties_df)
            
        # Validate time series consistency
        results["time_series"] = validator.validate_time_series_consistency(repeat_sales_df)
        
        # Generate validation report
        validation_report = validator.generate_validation_report()
        logger.info(f"Overall data quality score: {validation_report['summary']['overall_score']:.2f}")
        
        return results
        
    def _process_geography(
        self,
        transactions_df: pl.DataFrame,
        properties_df: Optional[pl.DataFrame],
        repeat_sales_df: pl.DataFrame
    ) -> Dict[str, pl.DataFrame]:
        """Process geographic data and create mappings."""
        mappings = {}
        
        # Create supertract mapping if tract level is included
        if GeographyLevel.TRACT in self.config.geography_levels:
            generator = SupertractGenerator(
                min_transactions=100,
                max_distance_km=10.0,
                method="hierarchical"
            )
            
            # Prepare tract data
            tract_stats = generator.prepare_tract_data(transactions_df, properties_df)
            
            # Generate supertracts
            supertracts = generator.generate_supertracts(tract_stats)
            logger.info(f"Generated {len(supertracts)} supertracts")
            
            # Export mapping
            mappings["tract_to_supertract"] = generator.export_mapping()
            
        # Create other geographic mappings if properties available
        if properties_df is not None:
            aggregator = IndexAggregator()
            
            if all(col in properties_df.columns for col in ["tract", "county_fips"]):
                mappings["tract_to_county"] = aggregator.create_geography_mapping(
                    properties_df, "tract", "county_fips"
                )
                
            if all(col in properties_df.columns for col in ["county_fips", "msa_code"]):
                mappings["county_to_msa"] = aggregator.create_geography_mapping(
                    properties_df, "county_fips", "msa_code"
                )
                
            if all(col in properties_df.columns for col in ["county_fips", "state"]):
                mappings["county_to_state"] = aggregator.create_geography_mapping(
                    properties_df, "county_fips", "state"
                )
                
        return mappings
        
    def _calculate_weights(
        self,
        repeat_sales_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate weights for repeat sales."""
        calculator = WeightCalculator(self.config.weighting_scheme)
        
        # Calculate weights
        weighted_df = calculator.calculate_weights(repeat_sales_df)
        
        # Diagnose weight distribution
        diagnostics = calculator.diagnose_weights(weighted_df)
        logger.info(f"Weight diagnostics - Mean: {diagnostics['mean']:.3f}, CV: {diagnostics['cv']:.3f}")
        
        if diagnostics['warnings']:
            for warning in diagnostics['warnings']:
                logger.warning(f"Weight warning: {warning}")
                
        return weighted_df
        
    def _run_regression(
        self,
        weighted_df: pl.DataFrame
    ) -> Dict[str, Any]:
        """Run BMN regression for each geography."""
        regression = BMNRegression(
            frequency=self.config.frequency,
            min_pairs_per_period=self.config.min_pairs_threshold
        )
        
        results = {}
        
        # Run regression by geography level
        for geo_level in self.config.geography_levels:
            if geo_level == GeographyLevel.PROPERTY:
                continue  # Skip property level
                
            # Determine geography column
            if geo_level == GeographyLevel.TRACT:
                geo_col = "tract"
            elif geo_level == GeographyLevel.COUNTY:
                geo_col = "county_fips"
            elif geo_level == GeographyLevel.MSA:
                geo_col = "msa_code"
            else:
                continue
                
            if geo_col in weighted_df.columns:
                geo_results = regression.fit_multiple_geographies(
                    weighted_df,
                    geo_col,
                    geo_level,
                    weights_df=weighted_df.select(["pair_id", "weight"]),
                    min_pairs=self.config.min_pairs_threshold
                )
                
                results.update(geo_results)
                logger.info(f"Fitted {len(geo_results)} models for {geo_level.value}")
                
        return results
        
    def _aggregate_indices(
        self,
        regression_results: Dict[str, Any],
        geography_mappings: Dict[str, pl.DataFrame]
    ) -> Dict[str, pl.DataFrame]:
        """Aggregate indices to higher geographic levels."""
        aggregator = IndexAggregator(
            aggregation_method="weighted_mean",
            weight_by="transaction_count"
        )
        
        aggregated = {}
        
        # Convert regression results to DataFrame
        all_indices = []
        for result in regression_results.values():
            for idx_val in result.index_values:
                all_indices.append({
                    "geography_level": idx_val.geography_level.value if hasattr(idx_val.geography_level, 'value') else idx_val.geography_level,
                    "geography_id": idx_val.geography_id,
                    "period": idx_val.period,
                    "index_value": idx_val.index_value,
                    "num_pairs": idx_val.num_pairs,
                    "num_properties": idx_val.num_properties,
                    "median_price": idx_val.median_price,
                    "standard_error": idx_val.standard_error
                })
                
        index_df = pl.DataFrame(all_indices)
        aggregated["all_indices"] = index_df
        
        # Aggregate to MSA if county indices available
        county_indices = index_df.filter(pl.col("geography_level") == "county")
        if len(county_indices) > 0 and "county_to_msa" in geography_mappings:
            msa_indices = aggregator.aggregate_to_msa(
                county_indices,
                geography_mappings["county_to_msa"]
            )
            aggregated["msa_indices"] = msa_indices
            logger.info(f"Aggregated to {len(msa_indices)} MSAs")
            
        # Aggregate to state
        if "county_to_state" in geography_mappings:
            state_indices = aggregator.aggregate_to_state(
                county_indices,
                geography_mappings["county_to_state"]
            )
            aggregated["state_indices"] = state_indices
            logger.info(f"Aggregated to {len(state_indices)} states")
            
            # Aggregate to national
            if len(state_indices) > 0:
                national_indices = aggregator.aggregate_to_national(state_indices)
                aggregated["national_indices"] = national_indices
                logger.info("Calculated national index")
                
        return aggregated
        
    def _generate_outputs(
        self,
        aggregated_indices: Dict[str, pl.DataFrame],
        regression_results: Dict[str, Any],
        validation_results: Dict[str, Any],
        repeat_sales_df: pl.DataFrame,
        output_dir: Union[str, Path]
    ) -> Dict[str, Path]:
        """Generate all outputs."""
        exporter = OutputExporter(output_dir, self.config)
        output_paths = {}
        
        # Export index values
        all_indices = []
        for result in regression_results.values():
            all_indices.extend(result.index_values)
            
        if all_indices:
            output_paths["index_values"] = exporter.export_index_values(
                all_indices,
                format=self.config.output_format
            )
            
        # Export regression results
        if self.config.include_diagnostics:
            output_paths["regression_results"] = exporter.export_regression_results(
                regression_results
            )
            
        # Generate summary report
        if "all_indices" in aggregated_indices:
            output_paths["summary_report"] = exporter.generate_summary_report(
                aggregated_indices["all_indices"],
                validation_results
            )
            
        # Create visualizations
        if "all_indices" in aggregated_indices:
            plot_paths = exporter.create_index_plots(
                aggregated_indices["all_indices"],
                interactive=True
            )
            output_paths.update({f"plot_{k}": v for k, v in plot_paths.items()})
            
        # Export for Tableau
        if "all_indices" in aggregated_indices:
            output_paths["tableau_data"] = exporter.export_for_tableau(
                aggregated_indices["all_indices"],
                repeat_sales_df
            )
            
        # Create methodology document
        output_paths["methodology"] = exporter.create_methodology_document()
        
        logger.info(f"Generated {len(output_paths)} output files")
        return output_paths


def main():
    """Command-line interface for RSAI pipeline."""
    parser = argparse.ArgumentParser(description="Run RSAI price index model")
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "transactions",
        type=str,
        help="Path to transactions data"
    )
    parser.add_argument(
        "--properties",
        type=str,
        help="Path to properties data (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(args.log_level)
    
    # Run pipeline
    try:
        pipeline = RSAIPipeline(args.config)
        results = pipeline.run(
            args.transactions,
            args.properties,
            args.output_dir
        )
        
        # Print summary
        print("\n" + "="*50)
        print("RSAI Pipeline Summary")
        print("="*50)
        print(f"Status: {results['status']}")
        print(f"Execution time: {results['execution_time_seconds']:.2f} seconds")
        
        if results['status'] == 'success':
            print(f"\nData Summary:")
            for key, value in results['data_summary'].items():
                print(f"  {key}: {value:,}" if value else f"  {key}: N/A")
                
            print(f"\nIndex Summary:")
            for key, value in results['index_summary'].items():
                print(f"  {key}: {value}")
                
            print(f"\nOutput Files:")
            for key, path in results['output_files'].items():
                print(f"  {key}: {path}")
                
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()