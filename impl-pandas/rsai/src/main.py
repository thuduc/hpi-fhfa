"""Main entry point for RSAI model"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, List, Dict
import argparse

from .data.ingestion import DataIngestion, RepeatSalesProcessor
from .data.validation import DataValidator, log_validation_results
from .geography.supertract import SupertractGenerator
from .index.aggregation import CityLevelAggregator
from .output.export import RSAIExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RSAIPipeline:
    """
    Main pipeline for running the complete RSAI model.
    
    This class orchestrates all steps from data loading through
    index generation and export.
    """
    
    def __init__(self, 
                 min_half_pairs: int = 40,
                 base_index_value: float = 100.0,
                 base_year: Optional[int] = None):
        """
        Initialize RSAI pipeline.
        
        Parameters:
        -----------
        min_half_pairs: int
            Minimum half-pairs threshold for supertracts
        base_index_value: float
            Base value for indices
        base_year: int, optional
            Base year for indices
        """
        self.min_half_pairs = min_half_pairs
        self.base_index_value = base_index_value
        self.base_year = base_year
        
        # Initialize components
        self.data_ingestion = DataIngestion()
        self.repeat_sales_processor = RepeatSalesProcessor()
        self.data_validator = DataValidator()
        
    def load_data(self, 
                  transaction_file: str,
                  geographic_file: str,
                  weighting_file: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load all required data files.
        
        Parameters:
        -----------
        transaction_file: str
            Path to transaction data CSV
        geographic_file: str
            Path to geographic data CSV
        weighting_file: str, optional
            Path to weighting data CSV
            
        Returns:
        --------
        dict
            Dictionary with loaded dataframes
        """
        logger.info("Loading data files")
        
        # Load transaction data
        transactions_df = self.data_ingestion.load_transaction_data(transaction_file)
        
        # Validate transaction data
        validation_results = self.data_validator.validate_transactions(transactions_df)
        log_validation_results(validation_results, "Transaction Data")
        
        # Load geographic data
        geographic_df = self.data_ingestion.load_geographic_data(geographic_file)
        
        # Validate geographic data
        validation_results = self.data_validator.validate_geographic_data(geographic_df)
        log_validation_results(validation_results, "Geographic Data")
        
        # Load weighting data if provided
        weighting_df = None
        if weighting_file:
            weighting_df = self.data_ingestion.load_weighting_data(weighting_file)
        
        return {
            'transactions': transactions_df,
            'geographic': geographic_df,
            'weighting': weighting_df
        }
    
    def run_pipeline(self,
                    transaction_file: str,
                    geographic_file: str,
                    output_file: str,
                    start_year: int = 1989,
                    end_year: int = 2021,
                    weighting_file: Optional[str] = None,
                    weighting_schemes: Optional[List[str]] = None,
                    output_format: str = 'csv',
                    wide_format: bool = False) -> pd.DataFrame:
        """
        Run the complete RSAI pipeline.
        
        Parameters:
        -----------
        transaction_file: str
            Path to transaction data
        geographic_file: str
            Path to geographic data
        output_file: str
            Path for output file
        start_year: int
            First year to calculate indices for
        end_year: int
            Last year to calculate indices for
        weighting_file: str, optional
            Path to weighting data
        weighting_schemes: List[str], optional
            Specific weighting schemes to use
        output_format: str
            Output format ('csv' or 'parquet')
        wide_format: bool
            If True, outputs wide format CSV
            
        Returns:
        --------
        pd.DataFrame
            Final index values
        """
        logger.info("Starting RSAI pipeline")
        
        # Step 1: Load data
        data = self.load_data(transaction_file, geographic_file, weighting_file)
        
        # Step 2: Process repeat sales
        logger.info("Processing repeat sales")
        repeat_sales_df = self.repeat_sales_processor.process_repeat_sales(
            data['transactions']
        )
        
        # Validate repeat sales
        validation_results = self.data_validator.validate_repeat_sales(repeat_sales_df)
        log_validation_results(validation_results, "Repeat Sales")
        
        # Step 3: Generate supertracts
        logger.info("Generating supertracts")
        supertract_generator = SupertractGenerator(
            data['geographic'], 
            self.min_half_pairs
        )
        
        supertracts_df = supertract_generator.generate_all_supertracts(
            repeat_sales_df, start_year, end_year
        )
        
        logger.info(f"Generated {len(supertracts_df)} supertract definitions")
        
        # Step 4: Calculate city-level appreciation
        logger.info("Calculating city-level appreciation rates")
        aggregator = CityLevelAggregator()
        
        appreciation_df = aggregator.process_all_years(
            repeat_sales_df,
            supertracts_df,
            start_year,
            end_year,
            data['weighting'],
            weighting_schemes
        )
        
        # Step 5: Chain indices and export
        logger.info("Chaining indices and exporting results")
        exporter = RSAIExporter(self.base_index_value, self.base_year)
        
        index_df = exporter.process_and_export(
            appreciation_df,
            output_file,
            format=output_format,
            wide_format=wide_format,
            include_summary=True
        )
        
        logger.info("RSAI pipeline completed successfully")
        return index_df


def main():
    """Command line interface for RSAI model"""
    parser = argparse.ArgumentParser(
        description="Run the Repeat-Sales Aggregation Index (RSAI) model"
    )
    
    # Required arguments
    parser.add_argument('transaction_file', help='Path to transaction data CSV')
    parser.add_argument('geographic_file', help='Path to geographic data CSV')
    parser.add_argument('output_file', help='Path for output file')
    
    # Optional arguments
    parser.add_argument('--weighting-file', help='Path to weighting data CSV')
    parser.add_argument('--start-year', type=int, default=1989,
                       help='Start year for index calculation (default: 1989)')
    parser.add_argument('--end-year', type=int, default=2021,
                       help='End year for index calculation (default: 2021)')
    parser.add_argument('--min-half-pairs', type=int, default=40,
                       help='Minimum half-pairs threshold (default: 40)')
    parser.add_argument('--base-year', type=int,
                       help='Base year for index (default: first year)')
    parser.add_argument('--base-value', type=float, default=100.0,
                       help='Base index value (default: 100.0)')
    parser.add_argument('--weighting-schemes', nargs='+',
                       choices=['sample', 'value', 'unit', 'upb', 'college', 'non_white'],
                       help='Specific weighting schemes to use (default: all)')
    parser.add_argument('--output-format', choices=['csv', 'parquet'], default='csv',
                       help='Output format (default: csv)')
    parser.add_argument('--wide-format', action='store_true',
                       help='Output CSV in wide format')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = RSAIPipeline(
        min_half_pairs=args.min_half_pairs,
        base_index_value=args.base_value,
        base_year=args.base_year
    )
    
    pipeline.run_pipeline(
        transaction_file=args.transaction_file,
        geographic_file=args.geographic_file,
        output_file=args.output_file,
        start_year=args.start_year,
        end_year=args.end_year,
        weighting_file=args.weighting_file,
        weighting_schemes=args.weighting_schemes,
        output_format=args.output_format,
        wide_format=args.wide_format
    )


if __name__ == "__main__":
    main()