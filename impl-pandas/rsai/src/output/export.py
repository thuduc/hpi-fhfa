"""Output generation and index chaining for RSAI"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class IndexChainer:
    """
    Chains annual appreciation rates into continuous index series.
    
    Implements the formula: P_t^a = P_{t-1}^a × exp(p_t^a)
    where p_t^a is the annual appreciation rate.
    """
    
    def __init__(self, base_value: float = 100.0, base_year: Optional[int] = None):
        """
        Initialize index chainer.
        
        Parameters:
        -----------
        base_value: float
            Initial index value (default 100)
        base_year: int, optional
            Base year for the index (if None, uses first year in data)
        """
        self.base_value = base_value
        self.base_year = base_year
        self.chained_indices = {}
    
    def chain_appreciation_rates(self, appreciation_df: pd.DataFrame,
                               cbsa_id: str,
                               weighting_scheme: str) -> pd.DataFrame:
        """
        Chain annual appreciation rates into index series.
        
        Parameters:
        -----------
        appreciation_df: pd.DataFrame
            DataFrame with columns: cbsa_id, year, weighting_scheme, appreciation_rate
        cbsa_id: str
            CBSA to calculate index for
        weighting_scheme: str
            Weighting scheme to use
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: year, index_value, appreciation_rate
        """
        # Filter for specific CBSA and weighting scheme
        filtered_df = appreciation_df[
            (appreciation_df['cbsa_id'] == cbsa_id) &
            (appreciation_df['weighting_scheme'] == weighting_scheme)
        ].sort_values('year')
        
        if filtered_df.empty:
            logger.warning(f"No data found for CBSA {cbsa_id}, scheme {weighting_scheme}")
            return pd.DataFrame()
        
        # Initialize index series
        years = filtered_df['year'].values
        appreciation_rates = filtered_df['appreciation_rate'].values
        
        # Set base year
        if self.base_year is None:
            self.base_year = years[0]
        
        # Calculate index values
        index_values = np.zeros(len(years))
        
        # Find base year position
        if self.base_year in years:
            base_idx = np.where(years == self.base_year)[0][0]
            index_values[base_idx] = self.base_value
            
            # Forward chaining from base year
            for i in range(base_idx + 1, len(years)):
                index_values[i] = index_values[i-1] * np.exp(appreciation_rates[i])
            
            # Backward chaining from base year
            for i in range(base_idx - 1, -1, -1):
                index_values[i] = index_values[i+1] / np.exp(appreciation_rates[i+1])
        else:
            # Base year not in data, start from first year
            index_values[0] = self.base_value
            for i in range(1, len(years)):
                index_values[i] = index_values[i-1] * np.exp(appreciation_rates[i])
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'year': years,
            'index_value': index_values,
            'appreciation_rate': appreciation_rates,
            'cbsa_id': cbsa_id,
            'weighting_scheme': weighting_scheme
        })
        
        # Store for later use
        key = (cbsa_id, weighting_scheme)
        self.chained_indices[key] = result_df
        
        return result_df
    
    def chain_all_indices(self, appreciation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Chain appreciation rates for all CBSA and weighting scheme combinations.
        
        Parameters:
        -----------
        appreciation_df: pd.DataFrame
            All appreciation rate data
            
        Returns:
        --------
        pd.DataFrame
            All chained index series
        """
        all_indices = []
        
        # Get unique combinations
        combinations = appreciation_df.groupby(['cbsa_id', 'weighting_scheme']).size()
        
        for (cbsa_id, weighting_scheme), _ in combinations.items():
            index_df = self.chain_appreciation_rates(
                appreciation_df, cbsa_id, weighting_scheme
            )
            if not index_df.empty:
                all_indices.append(index_df)
        
        return pd.concat(all_indices, ignore_index=True) if all_indices else pd.DataFrame()


class OutputGenerator:
    """Generates various output formats for RSAI results"""
    
    def __init__(self):
        self.output_data = {}
    
    def prepare_standard_output(self, index_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare standard output format with key columns.
        
        Parameters:
        -----------
        index_df: pd.DataFrame
            Chained index data
            
        Returns:
        --------
        pd.DataFrame
            Standardized output format
        """
        output_df = index_df.copy()
        
        # Calculate year-over-year percentage change
        output_df['yoy_change'] = output_df.groupby(
            ['cbsa_id', 'weighting_scheme']
        )['index_value'].pct_change() * 100
        
        # Calculate cumulative change from base
        output_df['cumulative_change'] = output_df.groupby(
            ['cbsa_id', 'weighting_scheme']
        )['index_value'].transform(lambda x: (x / x.iloc[0] - 1) * 100)
        
        # Reorder columns
        column_order = [
            'cbsa_id', 'year', 'weighting_scheme', 'index_value',
            'appreciation_rate', 'yoy_change', 'cumulative_change'
        ]
        
        return output_df[column_order]
    
    def export_to_csv(self, index_df: pd.DataFrame, 
                     output_path: Union[str, Path],
                     wide_format: bool = False) -> None:
        """
        Export index data to CSV file.
        
        Parameters:
        -----------
        index_df: pd.DataFrame
            Index data to export
        output_path: str or Path
            Output file path
        wide_format: bool
            If True, pivots data to wide format with weighting schemes as columns
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if wide_format:
            # Pivot to wide format
            pivot_df = index_df.pivot_table(
                index=['cbsa_id', 'year'],
                columns='weighting_scheme',
                values='index_value'
            ).reset_index()
            
            pivot_df.to_csv(output_path, index=False)
            logger.info(f"Exported wide format data to {output_path}")
        else:
            # Long format
            standard_df = self.prepare_standard_output(index_df)
            standard_df.to_csv(output_path, index=False)
            logger.info(f"Exported long format data to {output_path}")
    
    def export_to_parquet(self, index_df: pd.DataFrame,
                         output_path: Union[str, Path]) -> None:
        """Export index data to Parquet format for efficient storage"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        standard_df = self.prepare_standard_output(index_df)
        standard_df.to_parquet(output_path, index=False)
        logger.info(f"Exported data to {output_path}")
    
    def generate_summary_statistics(self, index_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for each CBSA and weighting scheme.
        
        Returns:
        --------
        pd.DataFrame
            Summary statistics including mean appreciation, volatility, etc.
        """
        summary_stats = []
        
        grouped = index_df.groupby(['cbsa_id', 'weighting_scheme'])
        
        for (cbsa_id, scheme), group in grouped:
            stats = {
                'cbsa_id': cbsa_id,
                'weighting_scheme': scheme,
                'start_year': group['year'].min(),
                'end_year': group['year'].max(),
                'n_years': len(group),
                'mean_appreciation': group['appreciation_rate'].mean(),
                'std_appreciation': group['appreciation_rate'].std(),
                'total_appreciation': (group['index_value'].iloc[-1] / 
                                     group['index_value'].iloc[0] - 1) * 100,
                'min_annual_appreciation': group['appreciation_rate'].min(),
                'max_annual_appreciation': group['appreciation_rate'].max()
            }
            
            summary_stats.append(stats)
        
        return pd.DataFrame(summary_stats)
    
    def export_by_cbsa(self, index_df: pd.DataFrame,
                      output_dir: Union[str, Path]) -> None:
        """
        Export separate files for each CBSA.
        
        Parameters:
        -----------
        index_df: pd.DataFrame
            All index data
        output_dir: str or Path
            Directory to save CBSA files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for cbsa_id in index_df['cbsa_id'].unique():
            cbsa_data = index_df[index_df['cbsa_id'] == cbsa_id]
            
            # Pivot to wide format for CBSA file
            pivot_df = cbsa_data.pivot_table(
                index='year',
                columns='weighting_scheme',
                values='index_value'
            ).reset_index()
            
            output_path = output_dir / f"hpi_{cbsa_id}.csv"
            pivot_df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(index_df['cbsa_id'].unique())} CBSA files to {output_dir}")


class RSAIExporter:
    """Main exporter class that combines chaining and output generation"""
    
    def __init__(self, base_value: float = 100.0, base_year: Optional[int] = None):
        self.chainer = IndexChainer(base_value, base_year)
        self.output_gen = OutputGenerator()
    
    def process_and_export(self, appreciation_df: pd.DataFrame,
                          output_path: Union[str, Path],
                          format: str = 'csv',
                          wide_format: bool = False,
                          include_summary: bool = True) -> pd.DataFrame:
        """
        Process appreciation rates and export results.
        
        Parameters:
        -----------
        appreciation_df: pd.DataFrame
            City-level appreciation rates
        output_path: str or Path
            Output file path
        format: str
            Output format ('csv' or 'parquet')
        wide_format: bool
            If True and format is CSV, uses wide format
        include_summary: bool
            If True, also generates summary statistics
            
        Returns:
        --------
        pd.DataFrame
            Chained index values
        """
        # Chain all indices
        logger.info("Chaining appreciation rates into index series")
        index_df = self.chainer.chain_all_indices(appreciation_df)
        
        # Export main results
        if format.lower() == 'csv':
            self.output_gen.export_to_csv(index_df, output_path, wide_format)
        elif format.lower() == 'parquet':
            self.output_gen.export_to_parquet(index_df, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Generate summary if requested
        if include_summary:
            summary_df = self.output_gen.generate_summary_statistics(index_df)
            summary_path = Path(output_path).parent / "summary_statistics.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Exported summary statistics to {summary_path}")
        
        return index_df