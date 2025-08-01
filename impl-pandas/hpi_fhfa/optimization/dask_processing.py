"""Dask-based implementations for large-scale HPI processing."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

# Check if dask is available
try:
    import dask.dataframe as dd
    from dask.distributed import Client, as_completed
    from dask import delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dd = None
    Client = None
    as_completed = None
    delayed = None

from ..models.repeat_sales import construct_repeat_sales_pairs
from ..aggregation.city_level import CityLevelIndexBuilder
from ..aggregation.weights import WeightType
from ..geography.census_tract import CensusTract

logger = logging.getLogger(__name__)


class DaskHPIProcessor:
    """Large-scale HPI processing using Dask for distributed computation."""
    
    def __init__(self, client: Optional['Client'] = None):
        """Initialize Dask processor.
        
        Args:
            client: Dask client for distributed processing (creates local if None)
        """
        if not DASK_AVAILABLE:
            raise ImportError(
                "Dask is not installed. Install with: pip install dask[complete]"
            )
        
        self.client = client or Client(n_workers=4, threads_per_worker=2)
        logger.info(f"Initialized Dask processor with client: {self.client}")
    
    def process_large_transactions(self,
                                 transactions_path: str,
                                 census_tracts: List[CensusTract],
                                 cbsa_code: str,
                                 chunk_size: int = 100000) -> dd.DataFrame:
        """Process large transaction files using Dask.
        
        Args:
            transactions_path: Path to large transaction file
            census_tracts: List of census tracts for the CBSA
            cbsa_code: CBSA code to filter for
            chunk_size: Size of chunks for processing
            
        Returns:
            Dask DataFrame with repeat sales pairs
        """
        logger.info(f"Processing large transaction file: {transactions_path}")
        
        # Read transactions as Dask DataFrame
        dtypes = {
            'property_id': str,
            'transaction_price': float,
            'census_tract': str,
            'cbsa_code': str,
            'distance_to_cbd': float
        }
        
        ddf = dd.read_csv(
            transactions_path,
            dtype=dtypes,
            parse_dates=['transaction_date'],
            blocksize=f"{chunk_size}KB"
        )
        
        # Filter for CBSA
        ddf = ddf[ddf['cbsa_code'] == cbsa_code]
        
        # Get valid tract codes
        valid_tracts = {tract.tract_code for tract in census_tracts}
        
        # Filter for valid tracts
        ddf = ddf[ddf['census_tract'].isin(valid_tracts)]
        
        # Sort by property and date (required for repeat sales)
        ddf = ddf.sort_values(['property_id', 'transaction_date'])
        
        # Partition by property_id for efficient grouping
        ddf = ddf.repartition(partition_size="100MB")
        
        return ddf
    
    def construct_repeat_sales_distributed(self,
                                         ddf: dd.DataFrame,
                                         output_path: Optional[str] = None) -> dd.DataFrame:
        """Construct repeat sales pairs in distributed fashion.
        
        Args:
            ddf: Dask DataFrame with transactions
            output_path: Optional path to save results
            
        Returns:
            Dask DataFrame with repeat sales pairs
        """
        logger.info("Constructing repeat sales pairs in distributed mode")
        
        # Define function to process each partition
        def process_partition(df):
            if len(df) == 0:
                return pd.DataFrame()
            
            # Use standard function on partition
            pairs = construct_repeat_sales_pairs(
                df,
                apply_filters=True
            )
            return pairs
        
        # Apply to each partition
        meta = pd.DataFrame({
            'property_id': pd.Series(dtype=str),
            'sale1_date': pd.Series(dtype='datetime64[ns]'),
            'sale1_price': pd.Series(dtype=float),
            'sale2_date': pd.Series(dtype='datetime64[ns]'),
            'sale2_price': pd.Series(dtype=float),
            'census_tract': pd.Series(dtype=str),
            'cbsa_code': pd.Series(dtype=str),
            'distance_to_cbd': pd.Series(dtype=float),
            'price_relative': pd.Series(dtype=float),
            'time_diff_years': pd.Series(dtype=float),
            'days_diff': pd.Series(dtype=int),
            'cagr': pd.Series(dtype=float),
            'sale1_year': pd.Series(dtype=int),
            'sale2_year': pd.Series(dtype=int),
            'sale1_quarter': pd.Series(dtype=int),
            'sale2_quarter': pd.Series(dtype=int),
            'sale1_period': pd.Series(dtype=int),
            'sale2_period': pd.Series(dtype=int),
            'period_1': pd.Series(dtype=int),
            'period_2': pd.Series(dtype=int),
            'time_diff_days': pd.Series(dtype=int)
        })
        
        pairs_ddf = ddf.map_partitions(process_partition, meta=meta)
        
        # Compute and save if path provided
        if output_path:
            logger.info(f"Saving repeat sales pairs to {output_path}")
            pairs_ddf.to_parquet(output_path, engine='pyarrow', compression='snappy')
        
        return pairs_ddf
    
    def build_indices_distributed(self,
                                pairs_ddf: dd.DataFrame,
                                census_tracts: List[CensusTract],
                                weight_type: WeightType,
                                start_year: int,
                                end_year: int,
                                min_half_pairs: int = 50) -> Dict[str, pd.DataFrame]:
        """Build indices for multiple CBSAs in distributed fashion.
        
        Args:
            pairs_ddf: Dask DataFrame with repeat sales pairs
            census_tracts: List of census tracts
            weight_type: Type of weighting to use
            start_year: Start year for index
            end_year: End year for index
            min_half_pairs: Minimum half-pairs for supertract
            
        Returns:
            Dictionary of CBSA codes to index DataFrames
        """
        logger.info("Building indices in distributed mode")
        
        # Group census tracts by CBSA
        cbsa_tracts = {}
        for tract in census_tracts:
            if tract.cbsa_code not in cbsa_tracts:
                cbsa_tracts[tract.cbsa_code] = []
            cbsa_tracts[tract.cbsa_code].append(tract)
        
        # Create delayed tasks for each CBSA
        tasks = []
        for cbsa_code, tracts in cbsa_tracts.items():
            # Filter pairs for this CBSA
            cbsa_pairs = pairs_ddf[pairs_ddf['cbsa_code'] == cbsa_code]
            
            # Create delayed task
            task = delayed(self._build_index_for_cbsa)(
                cbsa_pairs.compute(),  # Convert to pandas
                tracts,
                weight_type,
                start_year,
                end_year,
                min_half_pairs
            )
            tasks.append((cbsa_code, task))
        
        # Execute tasks and collect results
        results = {}
        futures = self.client.compute(dict(tasks))
        
        for cbsa_code, future in as_completed(futures):
            try:
                index_df = future.result()
                results[cbsa_code] = index_df
                logger.info(f"Completed index for CBSA {cbsa_code}")
            except Exception as e:
                logger.error(f"Failed to build index for CBSA {cbsa_code}: {e}")
        
        return results
    
    def _build_index_for_cbsa(self,
                             pairs_df: pd.DataFrame,
                             census_tracts: List[CensusTract],
                             weight_type: WeightType,
                             start_year: int,
                             end_year: int,
                             min_half_pairs: int) -> pd.DataFrame:
        """Build index for a single CBSA (worker function).
        
        Args:
            pairs_df: Pandas DataFrame with repeat sales pairs
            census_tracts: List of census tracts for this CBSA
            weight_type: Type of weighting to use
            start_year: Start year for index
            end_year: End year for index
            min_half_pairs: Minimum half-pairs for supertract
            
        Returns:
            DataFrame with index values
        """
        # Create builder
        builder = CityLevelIndexBuilder(min_half_pairs=min_half_pairs)
        
        # Build index
        index = builder.build_annual_index(
            pairs_df,
            census_tracts,
            weight_type,
            start_year,
            end_year
        )
        
        return index.to_dataframe()
    
    def close(self):
        """Close the Dask client."""
        if self.client:
            self.client.close()


def is_dask_available() -> bool:
    """Check if Dask is available."""
    return DASK_AVAILABLE


def process_multiple_cbsas(transaction_files: Dict[str, str],
                          census_tract_files: Dict[str, str],
                          output_dir: str,
                          start_year: int = 2015,
                          end_year: int = 2021,
                          n_workers: int = 4) -> None:
    """Process multiple CBSAs in parallel using Dask.
    
    Args:
        transaction_files: Dict mapping CBSA codes to transaction file paths
        census_tract_files: Dict mapping CBSA codes to tract file paths
        output_dir: Directory to save results
        start_year: Start year for indices
        end_year: End year for indices
        n_workers: Number of Dask workers
    """
    if not DASK_AVAILABLE:
        raise ImportError("Dask is not installed")
    
    # Create client
    with Client(n_workers=n_workers) as client:
        processor = DaskHPIProcessor(client)
        
        logger.info(f"Processing {len(transaction_files)} CBSAs with {n_workers} workers")
        
        # Process each CBSA
        for cbsa_code, trans_file in transaction_files.items():
            try:
                logger.info(f"Processing CBSA {cbsa_code}")
                
                # Load census tracts (implement based on your data format)
                # census_tracts = load_census_tracts(census_tract_files[cbsa_code])
                
                # Process transactions
                # ddf = processor.process_large_transactions(
                #     trans_file, census_tracts, cbsa_code
                # )
                
                # Build repeat sales
                # pairs_ddf = processor.construct_repeat_sales_distributed(ddf)
                
                # Build indices for all weight types
                # for weight_type in WeightType:
                #     results = processor.build_indices_distributed(
                #         pairs_ddf, census_tracts, weight_type,
                #         start_year, end_year
                #     )
                
                logger.info(f"Completed CBSA {cbsa_code}")
                
            except Exception as e:
                logger.error(f"Failed to process CBSA {cbsa_code}: {e}")
                continue