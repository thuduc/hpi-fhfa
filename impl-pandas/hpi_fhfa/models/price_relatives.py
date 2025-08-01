"""Price relative calculations for repeat sales analysis."""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


def calculate_price_relative(
    price1: float,
    price2: float,
    log_transform: bool = True
) -> float:
    """
    Calculate price relative between two transactions.
    
    For repeat-sales pair of property i between times τ and t:
    p_itτ = log(price_t) - log(price_τ)
    
    Parameters
    ----------
    price1 : float
        First (earlier) transaction price
    price2 : float
        Second (later) transaction price
    log_transform : bool, default True
        Whether to return log price difference (True) or price ratio (False)
        
    Returns
    -------
    float
        Log price difference if log_transform=True, else price ratio
    """
    if price1 <= 0 or price2 <= 0:
        raise ValueError("Prices must be positive")
    
    if log_transform:
        return np.log(price2) - np.log(price1)
    else:
        return price2 / price1


def calculate_price_relatives(
    df: pd.DataFrame,
    price1_col: str = 'sale1_price',
    price2_col: str = 'sale2_price',
    output_col: str = 'price_relative'
) -> pd.DataFrame:
    """
    Calculate price relatives for all repeat sales pairs.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with repeat sales pairs
    price1_col : str
        Column name for first sale price
    price2_col : str  
        Column name for second sale price
    output_col : str
        Column name for output price relatives
        
    Returns
    -------
    pd.DataFrame
        DataFrame with price relative column added
    """
    # Validate prices are positive
    if (df[price1_col] <= 0).any() or (df[price2_col] <= 0).any():
        raise ValueError("All prices must be positive")
    
    # Calculate log price differences
    df[output_col] = np.log(df[price2_col]) - np.log(df[price1_col])
    
    logger.info(f"Calculated {len(df):,} price relatives")
    
    return df


def calculate_half_pairs(
    transactions: pd.DataFrame,
    property_col: str = 'property_id',
    date_col: str = 'transaction_date',
    tract_col: str = 'census_tract',
    by_period: bool = True
) -> pd.DataFrame:
    """
    Calculate half-pairs count for each tract/period.
    
    For a property with transactions at times [t1, t2, t3]:
    - Half-pairs at t1: 1
    - Half-pairs at t2: 2  
    - Half-pairs at t3: 1
    
    Parameters
    ----------
    transactions : pd.DataFrame
        Transaction data
    property_col : str
        Column with property identifier
    date_col : str
        Column with transaction date
    tract_col : str
        Column with census tract
    by_period : bool, default True
        Whether to calculate by year/period
        
    Returns
    -------
    pd.DataFrame
        Half-pairs count by tract (and period if requested)
    """
    # Sort by property and date
    df = transactions.sort_values([property_col, date_col])
    
    # Count transactions per property
    property_counts = df.groupby(property_col).size()
    
    # Only keep properties with repeat sales
    repeat_properties = property_counts[property_counts > 1].index
    df_repeat = df[df[property_col].isin(repeat_properties)].copy()
    
    # Calculate position of each transaction within property
    df_repeat['transaction_order'] = df_repeat.groupby(property_col).cumcount()
    df_repeat['total_transactions'] = df_repeat[property_col].map(property_counts)
    
    # Calculate half-pairs for each transaction
    # First transaction: 1 half-pair
    # Middle transactions: 2 half-pairs each
    # Last transaction: 1 half-pair
    df_repeat['half_pairs'] = 2  # Default for middle transactions
    
    # First transactions
    first_mask = df_repeat['transaction_order'] == 0
    df_repeat.loc[first_mask, 'half_pairs'] = 1
    
    # Last transactions
    last_mask = df_repeat['transaction_order'] == df_repeat['total_transactions'] - 1
    df_repeat.loc[last_mask, 'half_pairs'] = 1
    
    # Aggregate by tract (and period if requested)
    if by_period:
        # Extract year from date
        df_repeat['year'] = pd.to_datetime(df_repeat[date_col]).dt.year
        
        # Group by tract and year
        half_pairs_summary = df_repeat.groupby([tract_col, 'year'])['half_pairs'].sum().reset_index()
        half_pairs_summary.columns = [tract_col, 'year', 'half_pairs_count']
    else:
        # Group by tract only
        half_pairs_summary = df_repeat.groupby(tract_col)['half_pairs'].sum().reset_index()
        half_pairs_summary.columns = [tract_col, 'half_pairs_count']
    
    logger.info(f"Calculated half-pairs for {len(half_pairs_summary)} tract-periods")
    
    return half_pairs_summary


def calculate_appreciation_rate(
    price_relative: float,
    time_diff_years: float
) -> float:
    """
    Calculate annualized appreciation rate from price relative.
    
    Parameters
    ----------
    price_relative : float
        Log price difference
    time_diff_years : float
        Time difference in years
        
    Returns
    -------
    float
        Annualized appreciation rate
    """
    if time_diff_years <= 0:
        raise ValueError("Time difference must be positive")
    
    return price_relative / time_diff_years


def calculate_cagr(
    price1: float,
    price2: float,
    time_diff_years: float
) -> float:
    """
    Calculate compound annual growth rate (CAGR).
    
    CAGR = (price2/price1)^(1/years) - 1
    
    Parameters
    ----------
    price1 : float
        First (earlier) price
    price2 : float
        Second (later) price
    time_diff_years : float
        Time difference in years
        
    Returns
    -------
    float
        Compound annual growth rate
    """
    if price1 <= 0 or price2 <= 0:
        raise ValueError("Prices must be positive")
    if time_diff_years <= 0:
        raise ValueError("Time difference must be positive")
    
    price_ratio = price2 / price1
    cagr = np.power(price_ratio, 1 / time_diff_years) - 1
    
    return cagr


def add_time_variables(
    df: pd.DataFrame,
    date1_col: str = 'sale1_date',
    date2_col: str = 'sale2_date'
) -> pd.DataFrame:
    """
    Add time-related variables to repeat sales DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Repeat sales pairs
    date1_col : str
        Column with first sale date
    date2_col : str
        Column with second sale date
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added time variables
    """
    # Ensure datetime
    df[date1_col] = pd.to_datetime(df[date1_col])
    df[date2_col] = pd.to_datetime(df[date2_col])
    
    # Time difference in days and years
    df['time_diff_days'] = (df[date2_col] - df[date1_col]).dt.days
    df['time_diff_years'] = df['time_diff_days'] / 365.25
    
    # Extract year and quarter
    df['sale1_year'] = df[date1_col].dt.year
    df['sale2_year'] = df[date2_col].dt.year
    df['sale1_quarter'] = df[date1_col].dt.quarter
    df['sale2_quarter'] = df[date2_col].dt.quarter
    
    # Create period indices (for BMN regression)
    # Map each unique year-quarter to an integer
    all_periods = pd.concat([
        df[date1_col].dt.to_period('Q'),
        df[date2_col].dt.to_period('Q')
    ]).unique()
    all_periods = sorted(all_periods)
    
    period_map = {period: idx for idx, period in enumerate(all_periods)}
    
    df['sale1_period'] = df[date1_col].dt.to_period('Q').map(period_map)
    df['sale2_period'] = df[date2_col].dt.to_period('Q').map(period_map)
    
    return df


def summarize_price_relatives(
    df: pd.DataFrame,
    price_relative_col: str = 'price_relative',
    group_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate summary statistics for price relatives.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price relatives
    price_relative_col : str
        Column containing price relatives
    group_col : str, optional
        Column to group by (e.g., 'census_tract', 'cbsa_code')
        
    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    if group_col:
        summary = df.groupby(group_col)[price_relative_col].agg([
            'count', 'mean', 'std', 'min', 'max',
            ('q25', lambda x: x.quantile(0.25)),
            ('median', lambda x: x.quantile(0.50)),
            ('q75', lambda x: x.quantile(0.75))
        ]).reset_index()
    else:
        stats = {
            'count': len(df),
            'mean': df[price_relative_col].mean(),
            'std': df[price_relative_col].std(),
            'min': df[price_relative_col].min(),
            'max': df[price_relative_col].max(),
            'q25': df[price_relative_col].quantile(0.25),
            'median': df[price_relative_col].quantile(0.50),
            'q75': df[price_relative_col].quantile(0.75)
        }
        summary = pd.DataFrame([stats])
    
    return summary