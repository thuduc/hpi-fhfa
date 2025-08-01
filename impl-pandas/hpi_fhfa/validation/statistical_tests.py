"""Statistical tests for HPI validation."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

# Check if statsmodels is available
try:
    from statsmodels.tsa.stattools import adfuller, kpss, coint
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    adfuller = None
    kpss = None
    coint = None
    acorr_ljungbox = None

logger = logging.getLogger(__name__)


def test_index_stationarity(index_values: pd.Series,
                          significance_level: float = 0.05) -> Tuple[bool, dict]:
    """Test if index returns are stationary using ADF and KPSS tests.
    
    Args:
        index_values: Series of index values
        significance_level: Significance level for tests
        
    Returns:
        Tuple of (is_stationary, test_results)
    """
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available, skipping stationarity tests")
        return True, {"message": "statsmodels not installed"}
    
    # Calculate returns
    returns = index_values.pct_change().dropna()
    
    results = {}
    
    # ADF test (null: non-stationary)
    adf_result = adfuller(returns, autolag='AIC')
    results['adf_statistic'] = adf_result[0]
    results['adf_pvalue'] = adf_result[1]
    results['adf_critical_values'] = adf_result[4]
    
    # KPSS test (null: stationary)
    kpss_result = kpss(returns, regression='c', nlags="auto")
    results['kpss_statistic'] = kpss_result[0]
    results['kpss_pvalue'] = kpss_result[1]
    results['kpss_critical_values'] = kpss_result[3]
    
    # Both tests should agree on stationarity
    adf_stationary = adf_result[1] < significance_level
    kpss_stationary = kpss_result[1] > significance_level
    
    is_stationary = adf_stationary and kpss_stationary
    results['is_stationary'] = is_stationary
    results['adf_rejects_null'] = adf_stationary
    results['kpss_fails_to_reject_null'] = kpss_stationary
    
    return is_stationary, results


def test_cointegration(index1: pd.Series,
                      index2: pd.Series,
                      significance_level: float = 0.05) -> Tuple[bool, dict]:
    """Test if two indices are cointegrated.
    
    Args:
        index1: First index series
        index2: Second index series
        significance_level: Significance level
        
    Returns:
        Tuple of (are_cointegrated, test_results)
    """
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available, skipping cointegration test")
        return False, {"message": "statsmodels not installed"}
    
    # Align series
    aligned = pd.concat([index1, index2], axis=1).dropna()
    if len(aligned) < 20:
        return False, {"message": "Insufficient data for cointegration test"}
    
    # Engle-Granger test
    coint_result = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])
    
    results = {
        'test_statistic': coint_result[0],
        'pvalue': coint_result[1],
        'critical_values': {
            '1%': coint_result[2][0],
            '5%': coint_result[2][1],
            '10%': coint_result[2][2]
        }
    }
    
    are_cointegrated = coint_result[1] < significance_level
    results['are_cointegrated'] = are_cointegrated
    
    return are_cointegrated, results


def calculate_tracking_error(index: pd.Series,
                           benchmark: pd.Series,
                           annualize: bool = True) -> float:
    """Calculate tracking error between index and benchmark.
    
    Args:
        index: Index series
        benchmark: Benchmark series
        annualize: Whether to annualize the tracking error
        
    Returns:
        Tracking error
    """
    # Align series
    aligned = pd.concat([index, benchmark], axis=1).dropna()
    
    # Calculate returns
    index_returns = aligned.iloc[:, 0].pct_change().dropna()
    benchmark_returns = aligned.iloc[:, 1].pct_change().dropna()
    
    # Tracking error is std of return differences
    tracking_error = np.std(index_returns - benchmark_returns)
    
    if annualize:
        # Assume annual data
        tracking_error = tracking_error * np.sqrt(1)
    
    return tracking_error


def test_return_autocorrelation(index_values: pd.Series,
                              lags: int = 10,
                              significance_level: float = 0.05) -> Tuple[bool, dict]:
    """Test for autocorrelation in index returns.
    
    Args:
        index_values: Series of index values
        lags: Number of lags to test
        significance_level: Significance level
        
    Returns:
        Tuple of (no_autocorrelation, test_results)
    """
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available, skipping autocorrelation test")
        return True, {"message": "statsmodels not installed"}
    
    # Calculate returns
    returns = index_values.pct_change().dropna()
    
    # Ljung-Box test
    lb_result = acorr_ljungbox(returns, lags=lags, return_df=True)
    
    results = {
        'ljung_box_stats': lb_result['lb_stat'].to_dict(),
        'ljung_box_pvalues': lb_result['lb_pvalue'].to_dict(),
        'min_pvalue': lb_result['lb_pvalue'].min(),
        'significant_lags': lb_result[lb_result['lb_pvalue'] < significance_level].index.tolist()
    }
    
    no_autocorrelation = results['min_pvalue'] > significance_level
    results['no_autocorrelation'] = no_autocorrelation
    
    return no_autocorrelation, results


def calculate_index_volatility(index_values: pd.Series,
                             window: Optional[int] = None) -> pd.Series:
    """Calculate rolling volatility of index returns.
    
    Args:
        index_values: Series of index values
        window: Rolling window size (None for expanding)
        
    Returns:
        Series of volatility values
    """
    # Calculate returns
    returns = index_values.pct_change().dropna()
    
    if window:
        volatility = returns.rolling(window=window).std()
    else:
        volatility = returns.expanding().std()
    
    return volatility


def test_index_efficiency(index_values: pd.Series,
                        benchmark_values: pd.Series,
                        risk_free_rate: float = 0.02) -> dict:
    """Calculate efficiency metrics for index.
    
    Args:
        index_values: Series of index values
        benchmark_values: Series of benchmark values
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary of efficiency metrics
    """
    # Align series
    aligned = pd.concat([index_values, benchmark_values], axis=1).dropna()
    
    # Calculate returns
    index_returns = aligned.iloc[:, 0].pct_change().dropna()
    benchmark_returns = aligned.iloc[:, 1].pct_change().dropna()
    
    # Calculate metrics
    metrics = {}
    
    # Sharpe ratio
    excess_returns = index_returns - risk_free_rate
    metrics['sharpe_ratio'] = excess_returns.mean() / excess_returns.std()
    
    # Information ratio
    active_returns = index_returns - benchmark_returns
    metrics['information_ratio'] = active_returns.mean() / active_returns.std()
    
    # Beta
    covariance = np.cov(index_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    metrics['beta'] = covariance / benchmark_variance
    
    # Alpha (Jensen's alpha)
    metrics['alpha'] = index_returns.mean() - (
        risk_free_rate + metrics['beta'] * (benchmark_returns.mean() - risk_free_rate)
    )
    
    # Correlation
    metrics['correlation'] = index_returns.corr(benchmark_returns)
    
    # R-squared
    metrics['r_squared'] = metrics['correlation'] ** 2
    
    return metrics