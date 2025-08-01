"""Data loading utilities for HPI-FHFA implementation."""

import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging

from .schemas import validate_transactions, validate_census_tracts
from ..config import constants

logger = logging.getLogger(__name__)


def load_transactions(
    filepath: Union[str, Path],
    validate: bool = True,
    chunksize: Optional[int] = None,
    **kwargs
) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
    """
    Load property transaction data from file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to transaction data file
    validate : bool, default True
        Whether to validate data against schema
    chunksize : int, optional
        If specified, return an iterator of DataFrames
    **kwargs
        Additional arguments passed to pandas read function
        
    Returns
    -------
    pd.DataFrame or iterator
        Transaction data (validated if requested)
        
    Raises
    ------
    ValueError
        If file format is not supported
    pa.errors.SchemaError
        If validation fails
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Transaction file not found: {filepath}")
    
    # Determine file format and read
    file_ext = filepath.suffix.lower()
    
    if file_ext == '.parquet':
        df = pd.read_parquet(filepath, **kwargs)
    elif file_ext == '.csv':
        df = pd.read_csv(filepath, chunksize=chunksize, **kwargs)
    elif file_ext == '.feather':
        df = pd.read_feather(filepath, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file format: {file_ext}. "
            f"Supported formats: {constants.SUPPORTED_INPUT_FORMATS}"
        )
    
    # Handle chunked reading
    if chunksize is not None and file_ext == '.csv':
        if validate:
            logger.warning(
                "Validation skipped for chunked reading. "
                "Validate each chunk separately."
            )
        return df
    
    # Validate if requested
    if validate:
        df = validate_transactions(df)
        logger.info(f"Loaded and validated {len(df):,} transactions")
    else:
        logger.info(f"Loaded {len(df):,} transactions (unvalidated)")
    
    # Optimize memory usage
    df = optimize_dtypes(df)
    
    return df


def load_census_data(
    filepath: Union[str, Path],
    validate: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Load census tract geographic and demographic data.
    
    Parameters
    ----------
    filepath : str or Path
        Path to census data file
    validate : bool, default True
        Whether to validate data against schema
    **kwargs
        Additional arguments passed to pandas read function
        
    Returns
    -------
    pd.DataFrame
        Census tract data (validated if requested)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Census data file not found: {filepath}")
    
    # Read file based on format
    file_ext = filepath.suffix.lower()
    
    if file_ext == '.parquet':
        df = pd.read_parquet(filepath, **kwargs)
    elif file_ext == '.csv':
        df = pd.read_csv(filepath, **kwargs)
    elif file_ext == '.feather':
        df = pd.read_feather(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Validate if requested
    if validate:
        df = validate_census_tracts(df)
        logger.info(f"Loaded and validated {len(df):,} census tracts")
    else:
        logger.info(f"Loaded {len(df):,} census tracts (unvalidated)")
    
    # Set census_tract as index for efficient lookups
    df = df.set_index('census_tract')
    
    return df


def save_results(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save results to file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to save
    filepath : str or Path
        Output file path
    format : str, optional
        Output format. If None, inferred from filepath extension
    **kwargs
        Additional arguments passed to pandas write function
    """
    filepath = Path(filepath)
    
    # Create directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format
    if format is None:
        format = filepath.suffix.lower()
    else:
        format = format.lower()
        if not format.startswith('.'):
            format = f'.{format}'
    
    # Save based on format
    if format == '.parquet':
        df.to_parquet(filepath, **kwargs)
    elif format == '.csv':
        df.to_csv(filepath, **kwargs)
    elif format == '.feather':
        df.to_feather(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported output format: {format}")
    
    logger.info(f"Saved {len(df):,} rows to {filepath}")


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes to reduce memory usage.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to optimize
        
    Returns
    -------
    pd.DataFrame
        Optimized DataFrame
    """
    # Convert string columns that repeat values to categorical
    for col in ['property_id', 'census_tract', 'cbsa_code']:
        if col in df.columns:
            nunique = df[col].nunique()
            nrows = len(df)
            # Convert to categorical if < 50% unique values
            if nunique / nrows < 0.5:
                df[col] = df[col].astype('category')
                logger.debug(f"Converted {col} to categorical dtype")
    
    # Downcast numeric types where possible
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df


def load_weight_data(
    filepath: Union[str, Path],
    weight_type: str,
    year: Optional[int] = None
) -> pd.DataFrame:
    """
    Load data for weight calculations.
    
    Parameters
    ----------
    filepath : str or Path
        Path to weight data file
    weight_type : str
        Type of weight data to load
    year : int, optional
        Specific year to load (for time-varying weights)
        
    Returns
    -------
    pd.DataFrame
        Weight data
    """
    df = pd.read_parquet(filepath)
    
    if year is not None and 'year' in df.columns:
        df = df[df['year'] == year]
    
    return df