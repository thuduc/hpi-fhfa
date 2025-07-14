"""Bailey, Muth, and Nourse (BMN) regression implementation for RSAI"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional
import logging
from scipy import sparse

logger = logging.getLogger(__name__)


class BMNRegression:
    """
    Implements the Bailey, Muth, and Nourse (1963) repeat-sales regression.
    
    The BMN method estimates price indices by regressing log price relatives
    on time dummy variables, using the full time sample of data.
    """
    
    def __init__(self):
        self.results = None
        self.time_periods = None
        self.base_period = None
    
    def prepare_regression_data(self, repeat_sales_df: pd.DataFrame,
                              start_year: int = None, 
                              end_year: int = None) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Prepare data for BMN regression by creating time dummy variables.
        
        Parameters:
        -----------
        repeat_sales_df: pd.DataFrame
            DataFrame with repeat sales pairs
        start_year: int, optional
            Start year for the analysis (if None, uses minimum year in data)
        end_year: int, optional
            End year for the analysis (if None, uses maximum year in data)
            
        Returns:
        --------
        tuple
            (X matrix of time dummies, y vector of log price relatives, list of years)
        """
        df = repeat_sales_df.copy()
        
        # Extract years from sale dates
        df['first_year'] = df['first_sale_date'].dt.year
        df['second_year'] = df['second_sale_date'].dt.year
        
        # Determine time range
        if start_year is None:
            start_year = df['first_year'].min()
        if end_year is None:
            end_year = df['second_year'].max()
        
        years = list(range(start_year, end_year + 1))
        n_years = len(years)
        n_obs = len(df)
        
        # Create sparse matrix for efficiency with large datasets
        # Each row represents a repeat sale pair
        # Columns are time dummies (excluding base period)
        X = sparse.lil_matrix((n_obs, n_years - 1))
        
        # Set base period as first year
        self.base_period = start_year
        
        # Fill the design matrix
        for i, row in df.iterrows():
            first_year_idx = row['first_year'] - start_year
            second_year_idx = row['second_year'] - start_year
            
            # Skip if outside our time range
            if first_year_idx < 0 or second_year_idx >= n_years:
                continue
            
            # Set dummy variables
            # -1 for first sale period, +1 for second sale period
            if first_year_idx > 0:  # Not base period
                X[i, first_year_idx - 1] = -1
            if second_year_idx > 0:  # Not base period
                X[i, second_year_idx - 1] = 1
        
        # Convert to dense array for statsmodels
        X_dense = X.toarray()
        
        # Log price relatives
        y = df['log_price_relative'].values
        
        self.time_periods = years
        
        return X_dense, y, years
    
    def run_regression(self, repeat_sales_df: pd.DataFrame,
                      start_year: int = None,
                      end_year: int = None) -> sm.regression.linear_model.RegressionResults:
        """
        Run the BMN regression on repeat sales data.
        
        Parameters:
        -----------
        repeat_sales_df: pd.DataFrame
            DataFrame with repeat sales pairs
        start_year: int, optional
            Start year for the analysis
        end_year: int, optional
            End year for the analysis
            
        Returns:
        --------
        RegressionResults
            Statsmodels regression results object
        """
        logger.info("Running BMN regression")
        
        # Prepare data
        X, y, years = self.prepare_regression_data(repeat_sales_df, start_year, end_year)
        
        # Check for sufficient observations
        if len(y) == 0:
            raise ValueError("No observations available for regression")
        
        if len(y) < X.shape[1]:
            logger.warning(f"Few observations ({len(y)}) relative to parameters ({X.shape[1]})")
        
        # Run OLS regression (no constant term in BMN)
        try:
            model = sm.OLS(y, X)
            self.results = model.fit()
            
            logger.info(f"Regression completed. R-squared: {self.results.rsquared:.4f}")
            
            return self.results
            
        except np.linalg.LinAlgError:
            logger.error("Regression failed due to singular matrix")
            raise
        except Exception as e:
            logger.error(f"Regression failed: {str(e)}")
            raise
    
    def get_index_values(self, base_value: float = 100.0) -> pd.DataFrame:
        """
        Extract index values from regression results.
        
        Parameters:
        -----------
        base_value: float
            Base index value (default 100)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: year, index_value, coefficient, std_error
        """
        if self.results is None:
            raise ValueError("Must run regression before extracting index values")
        
        # Get coefficients and standard errors
        coefficients = self.results.params
        std_errors = self.results.bse
        
        # Build index series
        index_data = []
        
        # Base period
        index_data.append({
            'year': self.base_period,
            'coefficient': 0.0,  # Base period coefficient is 0
            'std_error': 0.0,
            'index_value': base_value
        })
        
        # Other periods
        for i, year in enumerate(self.time_periods[1:]):
            coef = coefficients[i]
            se = std_errors[i]
            
            # Index value is base * exp(coefficient)
            index_value = base_value * np.exp(coef)
            
            index_data.append({
                'year': year,
                'coefficient': coef,
                'std_error': se,
                'index_value': index_value
            })
        
        return pd.DataFrame(index_data)
    
    def get_appreciation_rates(self) -> pd.DataFrame:
        """
        Calculate annual appreciation rates from regression coefficients.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: year, appreciation_rate
        """
        if self.results is None:
            raise ValueError("Must run regression before calculating appreciation rates")
        
        # Get coefficients
        coefficients = np.concatenate([[0.0], self.results.params])  # Include base period
        
        # Calculate year-over-year differences
        appreciation_rates = []
        
        for i in range(1, len(self.time_periods)):
            rate = coefficients[i] - coefficients[i-1]
            
            appreciation_rates.append({
                'year': self.time_periods[i],
                'appreciation_rate': rate
            })
        
        return pd.DataFrame(appreciation_rates)
    
    def get_coefficient_for_year(self, year: int) -> float:
        """
        Get the regression coefficient for a specific year.
        
        Parameters:
        -----------
        year: int
            Year to get coefficient for
            
        Returns:
        --------
        float
            Regression coefficient (0 for base period)
        """
        if self.results is None:
            raise ValueError("Must run regression before extracting coefficients")
        
        if year == self.base_period:
            return 0.0
        
        if year not in self.time_periods:
            raise ValueError(f"Year {year} not in regression time periods")
        
        # Find index in coefficient array
        year_idx = self.time_periods.index(year)
        
        if year_idx == 0:  # Base period
            return 0.0
        else:
            return self.results.params[year_idx - 1]
    
    def diagnostic_summary(self) -> Dict[str, float]:
        """
        Get regression diagnostic statistics.
        
        Returns:
        --------
        dict
            Dictionary of diagnostic statistics
        """
        if self.results is None:
            raise ValueError("Must run regression before getting diagnostics")
        
        return {
            'r_squared': self.results.rsquared,
            'adj_r_squared': self.results.rsquared_adj,
            'f_statistic': self.results.fvalue,
            'f_pvalue': self.results.f_pvalue,
            'n_observations': self.results.nobs,
            'n_parameters': len(self.results.params),
            'aic': self.results.aic,
            'bic': self.results.bic,
            'mse': self.results.mse_resid,
            'rmse': np.sqrt(self.results.mse_resid)
        }


def run_bmn_for_supertract(repeat_sales_df: pd.DataFrame,
                          supertract_tracts: List[str],
                          year: int) -> Tuple[float, float]:
    """
    Convenience function to run BMN regression for a supertract and extract appreciation.
    
    Parameters:
    -----------
    repeat_sales_df: pd.DataFrame
        All repeat sales data
    supertract_tracts: List[str]
        List of census tracts in the supertract
    year: int
        Year to calculate appreciation for
        
    Returns:
    --------
    tuple
        (appreciation_rate, coefficient_year_t)
    """
    # Filter for supertract
    supertract_sales = repeat_sales_df[
        repeat_sales_df['census_tract_2010'].isin(supertract_tracts)
    ]
    
    if len(supertract_sales) == 0:
        logger.warning(f"No sales data for supertract with tracts {supertract_tracts}")
        return 0.0, 0.0
    
    # Run regression
    bmn = BMNRegression()
    
    try:
        bmn.run_regression(supertract_sales)
        
        # Get coefficients for current and previous year
        coef_t = bmn.get_coefficient_for_year(year)
        coef_t_minus_1 = bmn.get_coefficient_for_year(year - 1)
        
        # Calculate appreciation rate
        appreciation_rate = coef_t - coef_t_minus_1
        
        return appreciation_rate, coef_t
        
    except Exception as e:
        logger.error(f"Regression failed for supertract: {str(e)}")
        return 0.0, 0.0