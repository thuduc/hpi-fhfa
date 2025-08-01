"""Repeat sales pair construction and handling."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import logging

from .price_relatives import calculate_price_relative, calculate_cagr, add_time_variables
from ..data.filters import filter_transactions

logger = logging.getLogger(__name__)


@dataclass
class RepeatSalesPair:
    """Container for a repeat sales transaction pair."""
    
    property_id: str
    sale1_date: pd.Timestamp
    sale1_price: float
    sale2_date: pd.Timestamp
    sale2_price: float
    census_tract: str
    cbsa_code: str
    distance_to_cbd: float
    
    @property
    def price_relative(self) -> float:
        """Calculate log price difference."""
        return calculate_price_relative(self.sale1_price, self.sale2_price)
    
    @property
    def time_diff_years(self) -> float:
        """Calculate time difference in years."""
        return (self.sale2_date - self.sale1_date).days / 365.25
    
    @property
    def cagr(self) -> float:
        """Calculate compound annual growth rate."""
        return calculate_cagr(self.sale1_price, self.sale2_price, self.time_diff_years)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'property_id': self.property_id,
            'sale1_date': self.sale1_date,
            'sale1_price': self.sale1_price,
            'sale2_date': self.sale2_date,
            'sale2_price': self.sale2_price,
            'census_tract': self.census_tract,
            'cbsa_code': self.cbsa_code,
            'distance_to_cbd': self.distance_to_cbd,
            'price_relative': self.price_relative,
            'time_diff_years': self.time_diff_years,
            'cagr': self.cagr
        }


def construct_repeat_sales_pairs(
    transactions: pd.DataFrame,
    property_col: str = 'property_id',
    date_col: str = 'transaction_date',
    price_col: str = 'transaction_price',
    tract_col: str = 'census_tract',
    cbsa_col: str = 'cbsa_code',
    cbd_dist_col: str = 'distance_to_cbd',
    apply_filters: bool = True,
    filter_kwargs: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Construct all valid repeat sales pairs from transaction data.
    
    Parameters
    ----------
    transactions : pd.DataFrame
        Raw transaction data
    property_col : str
        Column with property identifier
    date_col : str
        Column with transaction date
    price_col : str
        Column with transaction price
    tract_col : str
        Column with census tract
    cbsa_col : str
        Column with CBSA code
    cbd_dist_col : str
        Column with distance to CBD
    apply_filters : bool, default True
        Whether to apply standard filters
    filter_kwargs : dict, optional
        Additional arguments for filter_transactions
        
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per repeat sales pair
    """
    # Sort transactions by property and date
    transactions = transactions.sort_values([property_col, date_col])
    
    # Identify properties with multiple transactions
    property_counts = transactions.groupby(property_col).size()
    repeat_properties = property_counts[property_counts > 1].index
    
    logger.info(f"Found {len(repeat_properties):,} properties with repeat sales")
    
    # Filter to only repeat sale properties
    repeat_trans = transactions[transactions[property_col].isin(repeat_properties)]
    
    # Create all pairs for each property
    pairs_list = []
    
    for prop_id, group in repeat_trans.groupby(property_col):
        # Sort by date to ensure correct ordering
        group = group.sort_values(date_col)
        
        # Create all consecutive pairs
        # For a property with n transactions, create n-1 pairs
        for i in range(len(group) - 1):
            row1 = group.iloc[i]
            row2 = group.iloc[i + 1]
            
            pair_data = {
                'property_id': prop_id,
                'sale1_date': row1[date_col],
                'sale1_price': row1[price_col],
                'sale2_date': row2[date_col],
                'sale2_price': row2[price_col],
                'census_tract': row1[tract_col],  # Assuming tract doesn't change
                'cbsa_code': row1[cbsa_col],
                'distance_to_cbd': row1[cbd_dist_col]
            }
            
            pairs_list.append(pair_data)
    
    # Convert to DataFrame
    pairs_df = pd.DataFrame(pairs_list)
    
    logger.info(f"Created {len(pairs_df):,} initial repeat sales pairs")
    
    # Handle empty dataframe case
    if pairs_df.empty:
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=[
            'property_id', 'sale1_date', 'sale1_price', 'sale2_date', 'sale2_price',
            'census_tract', 'cbsa_code', 'distance_to_cbd', 'price_relative',
            'time_diff_years', 'days_diff', 'cagr', 'sale1_year', 'sale2_year',
            'sale1_quarter', 'sale2_quarter', 'sale1_period', 'sale2_period',
            'period_1', 'period_2', 'time_diff_days'
        ])
    
    # Add calculated fields
    pairs_df['price_relative'] = np.log(pairs_df['sale2_price']) - np.log(pairs_df['sale1_price'])
    pairs_df['time_diff_years'] = (pairs_df['sale2_date'] - pairs_df['sale1_date']).dt.days / 365.25
    pairs_df['days_diff'] = (pairs_df['sale2_date'] - pairs_df['sale1_date']).dt.days
    
    # Calculate CAGR
    price_ratio = pairs_df['sale2_price'] / pairs_df['sale1_price']
    pairs_df['cagr'] = np.power(price_ratio, 1 / pairs_df['time_diff_years']) - 1
    
    # Add time variables
    pairs_df = add_time_variables(pairs_df)
    
    # Add period_1 and period_2 columns for compatibility with supertract algorithm
    pairs_df['period_1'] = pairs_df['sale1_year']
    pairs_df['period_2'] = pairs_df['sale2_year']
    
    # Apply filters if requested
    if apply_filters:
        filter_kwargs = filter_kwargs or {}
        pairs_df = filter_transactions(pairs_df, **filter_kwargs)
        logger.info(f"After filtering: {len(pairs_df):,} repeat sales pairs remain")
    
    return pairs_df


def create_time_dummies(
    pairs_df: pd.DataFrame,
    period1_col: str = 'sale1_period',
    period2_col: str = 'sale2_period',
    sparse: bool = True
) -> pd.DataFrame:
    """
    Create time dummy variables for BMN regression.
    
    Parameters
    ----------
    pairs_df : pd.DataFrame
        Repeat sales pairs with period columns
    period1_col : str
        Column with first sale period
    period2_col : str
        Column with second sale period
    sparse : bool, default True
        Whether to return sparse DataFrame
        
    Returns
    -------
    pd.DataFrame
        DataFrame with dummy variables
    """
    # Get unique periods
    all_periods = np.unique(np.concatenate([
        pairs_df[period1_col].values,
        pairs_df[period2_col].values
    ]))
    
    n_periods = len(all_periods)
    n_pairs = len(pairs_df)
    
    # Create dummy DataFrame
    dummy_cols = [f'period_{p}' for p in all_periods]
    
    if sparse:
        # Create dense DataFrame first, then convert to sparse
        dummies = pd.DataFrame(
            0, 
            index=pairs_df.index,
            columns=dummy_cols,
            dtype='int8'
        )
    else:
        dummies = pd.DataFrame(
            0,
            index=pairs_df.index, 
            columns=dummy_cols,
            dtype='int8'
        )
    
    # Set dummy values
    for idx, row in pairs_df.iterrows():
        period1 = row[period1_col]
        period2 = row[period2_col]
        
        dummies.loc[idx, f'period_{period1}'] = -1
        dummies.loc[idx, f'period_{period2}'] = 1
    
    # Convert to sparse if requested
    if sparse:
        for col in dummies.columns:
            dummies[col] = pd.arrays.SparseArray(dummies[col], fill_value=0)
    
    return dummies


def split_by_tract(
    pairs_df: pd.DataFrame,
    tract_col: str = 'census_tract'
) -> Dict[str, pd.DataFrame]:
    """
    Split repeat sales pairs by census tract.
    
    Parameters
    ----------
    pairs_df : pd.DataFrame
        All repeat sales pairs
    tract_col : str
        Column with census tract identifier
        
    Returns
    -------
    dict
        Dictionary mapping tract ID to DataFrame of pairs
    """
    tract_groups = {}
    
    for tract, group in pairs_df.groupby(tract_col):
        tract_groups[tract] = group.reset_index(drop=True)
    
    logger.info(f"Split pairs into {len(tract_groups)} census tracts")
    
    return tract_groups


def calculate_pair_statistics(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics for repeat sales pairs.
    
    Parameters
    ----------
    pairs_df : pd.DataFrame
        Repeat sales pairs
        
    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    stats = {
        'n_pairs': len(pairs_df),
        'n_properties': pairs_df['property_id'].nunique(),
        'avg_time_between_sales': pairs_df['time_diff_years'].mean(),
        'median_time_between_sales': pairs_df['time_diff_years'].median(),
        'avg_price_relative': pairs_df['price_relative'].mean(),
        'std_price_relative': pairs_df['price_relative'].std(),
        'avg_cagr': pairs_df['cagr'].mean(),
        'median_cagr': pairs_df['cagr'].median(),
        'date_range_start': pairs_df['sale1_date'].min(),
        'date_range_end': pairs_df['sale2_date'].max()
    }
    
    # Add statistics by year if available
    if 'sale1_year' in pairs_df.columns:
        yearly_counts = pairs_df.groupby('sale1_year').size()
        stats['pairs_by_year'] = yearly_counts.to_dict()
    
    # Add statistics by tract if available
    if 'census_tract' in pairs_df.columns:
        tract_counts = pairs_df.groupby('census_tract').size()
        stats['n_tracts'] = len(tract_counts)
        stats['avg_pairs_per_tract'] = tract_counts.mean()
        stats['median_pairs_per_tract'] = tract_counts.median()
    
    return pd.Series(stats).to_frame('value')


def validate_repeat_sales_pairs(pairs_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate repeat sales pairs data.
    
    Parameters
    ----------
    pairs_df : pd.DataFrame
        Repeat sales pairs to validate
        
    Returns
    -------
    tuple
        (is_valid, list of error messages)
    """
    errors = []
    
    # Check required columns
    required_cols = [
        'property_id', 'sale1_date', 'sale2_date',
        'sale1_price', 'sale2_price', 'census_tract'
    ]
    missing_cols = set(required_cols) - set(pairs_df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check date ordering
    if 'sale1_date' in pairs_df.columns and 'sale2_date' in pairs_df.columns:
        bad_dates = pairs_df['sale1_date'] >= pairs_df['sale2_date']
        if bad_dates.any():
            errors.append(f"{bad_dates.sum()} pairs have sale1_date >= sale2_date")
    
    # Check positive prices
    if 'sale1_price' in pairs_df.columns:
        if (pairs_df['sale1_price'] <= 0).any():
            errors.append("Some sale1_price values are not positive")
    
    if 'sale2_price' in pairs_df.columns:
        if (pairs_df['sale2_price'] <= 0).any():
            errors.append("Some sale2_price values are not positive")
    
    # Check price relatives
    if 'price_relative' in pairs_df.columns:
        if pairs_df['price_relative'].isna().any():
            errors.append("Some price_relative values are NaN")
    
    # Check time differences
    if 'time_diff_years' in pairs_df.columns:
        if (pairs_df['time_diff_years'] <= 0).any():
            errors.append("Some time_diff_years values are not positive")
    
    is_valid = len(errors) == 0
    
    return is_valid, errors