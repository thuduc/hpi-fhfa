"""Main pipeline orchestrator for HPI-FHFA project"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from typing import Dict, List, Optional, Tuple
import logging
import yaml
import argparse
from datetime import datetime

from ..etl.data_processor import DataProcessor
from ..etl.data_validator import DataValidator
from ..core.supertract import SupertractAlgorithm
from ..core.bmn_regression import BMNRegression
from ..core.index_aggregation import IndexAggregator
from ..utils.spark_utils import create_spark_session
from ..utils.logging_config import setup_logging


class HPIPipeline:
    """
    Main pipeline orchestrator for FHFA House Price Index calculation
    """
    
    def __init__(self, config_path: Optional[str] = None, spark_config: Optional[Dict] = None):
        """
        Initialize pipeline with configuration
        
        Args:
            config_path: Path to pipeline configuration YAML
            spark_config: Spark configuration dictionary
        """
        self.logger = setup_logging(self.__class__.__name__)
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        # Initialize Spark
        self.spark = create_spark_session(
            app_name=self.config["pipeline"]["name"],
            mode="cluster" if spark_config else "local",
            additional_config=spark_config
        )
        
        # Initialize components
        self.data_processor = DataProcessor(self.spark)
        self.data_validator = DataValidator(self.spark)
        self.supertract_algo = SupertractAlgorithm(
            self.spark, 
            min_half_pairs=self.config["data"]["min_half_pairs"]
        )
        self.bmn_regression = BMNRegression(self.spark)
        self.aggregator = IndexAggregator(self.spark)
        
        self.logger.info(f"Initialized {self.config['pipeline']['name']} pipeline")
        
    def _get_default_config(self) -> Dict:
        """Get default pipeline configuration"""
        return {
            "pipeline": {
                "name": "HPI-FHFA-Pipeline",
                "version": "0.1.0"
            },
            "data": {
                "min_half_pairs": 40,
                "start_year": 1989,
                "end_year": 2021,
                "base_year": 1989
            },
            "filters": {
                "max_cagr": 0.30,
                "min_appreciation_ratio": 0.25,
                "max_appreciation_ratio": 10.0
            },
            "weights": {
                "types": ["sample", "value", "unit", "upb", "college", "nonwhite"],
                "static_weights_year": 2010
            },
            "output": {
                "format": "parquet",
                "compression": "snappy"
            }
        }
    
    def run_pipeline(
        self,
        transaction_path: str,
        geographic_path: str,
        weight_data_path: str,
        output_path: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Execute full HPI pipeline
        
        Args:
            transaction_path: Path to transaction data
            geographic_path: Path to geographic data
            weight_data_path: Path to weight data
            output_path: Path for output indices
            start_year: Override config start year
            end_year: Override config end year
            
        Returns:
            Dictionary with pipeline execution metrics
        """
        start_time = datetime.now()
        self.logger.info("Starting HPI pipeline execution")
        
        # Use provided years or config defaults
        start_year = start_year or self.config["data"]["start_year"]
        end_year = end_year or self.config["data"]["end_year"]
        
        metrics = {
            "start_time": start_time,
            "start_year": start_year,
            "end_year": end_year
        }
        
        try:
            # Step 1: Load and validate data
            self.logger.info("Step 1: Loading data")
            transactions, geographic_data, weight_data = self._load_data(
                transaction_path, geographic_path, weight_data_path
            )
            
            # Validate input data
            tx_validation = self.data_validator.validate_transactions(transactions)
            if not all(tx_validation.values()):
                self.logger.warning("Transaction data validation warnings found")
            
            # Step 2: Create repeat-sales pairs
            self.logger.info("Step 2: Creating repeat-sales pairs")
            repeat_sales = self.data_processor.create_repeat_sales_pairs(transactions)
            repeat_sales = self.data_processor.apply_filters(repeat_sales)
            metrics["repeat_sales_count"] = repeat_sales.count()
            
            # Step 3: Calculate half-pairs
            self.logger.info("Step 3: Calculating half-pairs")
            half_pairs = self.data_processor.calculate_half_pairs(repeat_sales)
            
            # Step 4: Process by year
            self.logger.info("Step 4: Processing indices by year")
            all_indices = []
            
            for year in range(start_year, end_year + 1):
                self.logger.info(f"Processing year {year}")
                year_indices = self._process_year(
                    year, repeat_sales, half_pairs, 
                    geographic_data, weight_data
                )
                if year_indices:
                    all_indices.append(year_indices)
            
            # Step 5: Combine and create index series
            self.logger.info("Step 5: Creating final index series")
            if all_indices:
                combined_indices = all_indices[0]
                for df in all_indices[1:]:
                    combined_indices = combined_indices.union(df)
                
                # Save combined indices
                self._save_output(combined_indices, output_path)
                metrics["output_rows"] = combined_indices.count()
            else:
                self.logger.error("No indices generated")
                metrics["output_rows"] = 0
            
            # Calculate execution time
            end_time = datetime.now()
            metrics["end_time"] = end_time
            metrics["duration_seconds"] = (end_time - start_time).total_seconds()
            metrics["status"] = "SUCCESS"
            
            self.logger.info(
                f"Pipeline completed successfully in "
                f"{metrics['duration_seconds']:.1f} seconds"
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            metrics["status"] = "FAILED"
            metrics["error"] = str(e)
        
        return metrics
    
    def _load_data(
        self, 
        transaction_path: str,
        geographic_path: str,
        weight_data_path: str
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Load input data with partitioning"""
        
        # Load transactions with partitioning
        transactions = self.spark.read.parquet(transaction_path)
        if transactions.count() > 1000000:  # Large dataset
            transactions = transactions.repartition(200, "cbsa_code", "transaction_date")
        
        # Load geographic data (small, suitable for broadcast)
        geographic_data = self.spark.read.parquet(geographic_path).cache()
        
        # Load weight data
        weight_data = self.spark.read.parquet(weight_data_path)
        
        self.logger.info(
            f"Loaded {transactions.count():,} transactions, "
            f"{geographic_data.count():,} geographic records, "
            f"{weight_data.count():,} weight records"
        )
        
        return transactions, geographic_data, weight_data
    
    def _process_year(
        self,
        year: int,
        repeat_sales: DataFrame,
        half_pairs: DataFrame,
        geographic_data: DataFrame,
        weight_data: DataFrame
    ) -> Optional[DataFrame]:
        """Process indices for a single year"""
        
        # Create supertracts for the year
        supertracts = self.supertract_algo.create_supertracts(
            half_pairs, geographic_data, year
        )
        
        # Add supertract mapping to repeat sales
        tract_mapping = self.supertract_algo.create_tract_to_supertract_mapping(
            supertracts
        )
        
        repeat_sales_mapped = repeat_sales.join(
            tract_mapping,
            on="census_tract",
            how="inner"
        )
        
        # Run BMN regressions for all supertracts
        bmn_results = self.bmn_regression.batch_process_supertracts(
            repeat_sales_mapped,
            supertracts,
            year
        )
        
        if bmn_results and bmn_results.count() > 0:
            # Process all weight types
            all_results = []
            
            # Get unique CBSAs
            cbsa_list = bmn_results.select("cbsa_code").distinct().collect()
            
            for cbsa_row in cbsa_list:
                cbsa = cbsa_row["cbsa_code"]
                
                # Filter BMN results for this CBSA
                cbsa_results = bmn_results.filter(F.col("cbsa_code") == cbsa)
                
                # Process all weight types
                weighted_results = self.aggregator.process_all_weights(
                    cbsa_results,
                    supertracts.filter(F.col("cbsa_code") == cbsa),
                    weight_data,
                    cbsa,
                    year
                )
                
                if weighted_results:
                    all_results.append(weighted_results)
            
            # Combine all results
            if all_results:
                combined = all_results[0]
                for df in all_results[1:]:
                    combined = combined.union(df)
                return combined
            else:
                return None
        else:
            self.logger.warning(f"No BMN results for year {year}")
            return None
    
    def _save_output(self, indices: DataFrame, output_path: str):
        """Save output indices with proper partitioning"""
        
        output_config = self.config["output"]
        
        writer = indices.write.mode("overwrite")
        
        # Apply partitioning if specified
        if "partition_by" in output_config:
            writer = writer.partitionBy(*output_config["partition_by"])
        
        # Set compression
        if "compression" in output_config:
            writer = writer.option("compression", output_config["compression"])
        
        # Save based on format
        if output_config["format"] == "parquet":
            writer.parquet(output_path)
        elif output_config["format"] == "delta":
            writer.format("delta").save(output_path)
        else:
            writer.csv(output_path, header=True)
        
        self.logger.info(f"Saved output to {output_path}")


def main():
    """Command-line interface for pipeline"""
    parser = argparse.ArgumentParser(description="HPI-FHFA Pipeline")
    parser.add_argument("--transaction-path", required=True, help="Path to transaction data")
    parser.add_argument("--geographic-path", required=True, help="Path to geographic data")
    parser.add_argument("--weight-path", required=True, help="Path to weight data")
    parser.add_argument("--output-path", required=True, help="Path for output indices")
    parser.add_argument("--config", help="Path to configuration YAML")
    parser.add_argument("--start-year", type=int, help="Start year")
    parser.add_argument("--end-year", type=int, help="End year")
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = HPIPipeline(config_path=args.config)
    
    metrics = pipeline.run_pipeline(
        args.transaction_path,
        args.geographic_path,
        args.weight_path,
        args.output_path,
        args.start_year,
        args.end_year
    )
    
    print(f"Pipeline completed: {metrics['status']}")
    if metrics['status'] == 'SUCCESS':
        print(f"Duration: {metrics['duration_seconds']:.1f} seconds")
        print(f"Output rows: {metrics['output_rows']:,}")


if __name__ == "__main__":
    main()