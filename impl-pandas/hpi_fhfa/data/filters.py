"""Transaction filtering functions for HPI-FHFA implementation."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

from ..config import constants

logger = logging.getLogger(__name__)


def filter_transactions(
    df: pd.DataFrame,
    min_cagr: Optional[float] = None,
    max_cagr: Optional[float] = None,
    filter_same_period: bool = True,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Apply all standard filters to repeat sales pairs.
    
    Parameters
    ----------
    df : pd.DataFrame
        Repeat sales pairs with required columns
    min_cagr : float, optional
        Minimum CAGR threshold (default: -max_cagr)
    max_cagr : float, optional  
        Maximum CAGR threshold (default: from constants)
    filter_same_period : bool, default True
        Whether to filter same 12-month period transactions
    inplace : bool, default False
        Whether to modify DataFrame in place
        
    Returns
    -------
    pd.DataFrame
        Filtered repeat sales pairs
    """
    if not inplace:
        df = df.copy()
    
    initial_count = len(df)
    
    # Apply same period filter
    if filter_same_period:
        df = apply_same_period_filter(df, inplace=True)
        same_period_removed = initial_count - len(df)
        logger.info(f"Removed {same_period_removed:,} same-period transactions")
        initial_count = len(df)
    
    # Apply CAGR filter
    if max_cagr is None:
        max_cagr = constants.MAX_CAGR
    if min_cagr is None:
        min_cagr = -max_cagr
        
    df = apply_cagr_filter(df, min_cagr, max_cagr, inplace=True)
    cagr_removed = initial_count - len(df)
    logger.info(f"Removed {cagr_removed:,} transactions exceeding CAGR thresholds")
    initial_count = len(df)
    
    # Apply cumulative appreciation filter
    df = apply_cumulative_filter(df, inplace=True)
    cumulative_removed = initial_count - len(df)
    logger.info(f"Removed {cumulative_removed:,} transactions exceeding cumulative thresholds")
    
    logger.info(f"Total transactions after filtering: {len(df):,}")
    
    return df


def apply_same_period_filter(
    df: pd.DataFrame,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Remove pairs where both transactions occur in same 12-month period.
    
    Parameters
    ----------
    df : pd.DataFrame
        Repeat sales pairs with sale1_date and sale2_date columns
    inplace : bool, default False
        Whether to modify DataFrame in place
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if not inplace:
        df = df.copy()
    
    # Extract year from dates
    year1 = df['sale1_date'].dt.year
    year2 = df['sale2_date'].dt.year
    
    # Keep only pairs with different years
    mask = year1 != year2
    
    return df[mask]


def apply_cagr_filter(
    df: pd.DataFrame,
    min_cagr: float = -0.30,
    max_cagr: float = 0.30,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Filter based on compound annual growth rate.
    
    Removes pairs where: |(V1/V0)^(1/(t1-t0)) - 1| > threshold
    
    Parameters
    ----------
    df : pd.DataFrame
        Repeat sales pairs with required columns
    min_cagr : float, default -0.30
        Minimum CAGR (-30%)
    max_cagr : float, default 0.30
        Maximum CAGR (30%)
    inplace : bool, default False
        Whether to modify DataFrame in place
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if not inplace:
        df = df.copy()
    
    # Calculate CAGR if not already present
    if 'cagr' not in df.columns:
        time_diff = (df['sale2_date'] - df['sale1_date']).dt.days / 365.25
        price_ratio = df['sale2_price'] / df['sale1_price']
        df['cagr'] = np.power(price_ratio, 1 / time_diff) - 1
    
    # Apply filter
    mask = (df['cagr'] >= min_cagr) & (df['cagr'] <= max_cagr)
    
    return df[mask]


def apply_cumulative_filter(
    df: pd.DataFrame,
    min_ratio: float = None,
    max_ratio: float = None,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Filter based on cumulative appreciation.
    
    Removes pairs where price ratio is > 10x or < 0.25x
    
    Parameters
    ----------
    df : pd.DataFrame
        Repeat sales pairs with price columns
    min_ratio : float, optional
        Minimum price ratio (default: from constants)
    max_ratio : float, optional
        Maximum price ratio (default: from constants)
    inplace : bool, default False
        Whether to modify DataFrame in place
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if not inplace:
        df = df.copy()
    
    if min_ratio is None:
        min_ratio = constants.MIN_CUMULATIVE_APPRECIATION
    if max_ratio is None:
        max_ratio = constants.MAX_CUMULATIVE_APPRECIATION
    
    # Calculate price ratio
    price_ratio = df['sale2_price'] / df['sale1_price']
    
    # Apply filter
    mask = (price_ratio >= min_ratio) & (price_ratio <= max_ratio)
    
    return df[mask]


def filter_outliers_by_zscore(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3.0,
    by_group: Optional[str] = None,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Remove outliers based on z-score.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to filter
    column : str
        Column to check for outliers
    threshold : float, default 3.0
        Z-score threshold
    by_group : str, optional
        Column to group by before calculating z-scores
    inplace : bool, default False
        Whether to modify DataFrame in place
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if not inplace:
        df = df.copy()
    
    if by_group is None:
        # Global z-score
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        mask = z_scores <= threshold
    else:
        # Group-wise z-score
        def zscore_filter(group):
            z_scores = np.abs((group[column] - group[column].mean()) / group[column].std())
            return z_scores <= threshold
        
        mask = df.groupby(by_group, group_keys=False).apply(zscore_filter)
    
    return df[mask]


def validate_filtered_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate filtered data meets requirements.
    
    Parameters
    ----------
    df : pd.DataFrame
        Filtered repeat sales data
        
    Returns
    -------
    tuple
        (is_valid, message)
    """
    if len(df) == 0:
        return False, "No transactions remain after filtering"
    
    # Check for required columns
    required_cols = [
        'property_id', 'sale1_date', 'sale2_date', 
        'sale1_price', 'sale2_price', 'census_tract'
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check date ordering
    if not (df['sale2_date'] > df['sale1_date']).all():
        return False, "sale2_date must be after sale1_date for all pairs"
    
    # Check price validity
    if (df['sale1_price'] <= 0).any() or (df['sale2_price'] <= 0).any():
        return False, "All prices must be positive"
    
    return True, "Data validation passed"