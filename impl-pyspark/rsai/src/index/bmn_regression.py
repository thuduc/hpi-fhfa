"""
Bailey-Muth-Nourse (BMN) regression implementation using PySpark MLlib.

This module implements the BMN repeat sales regression method for calculating
price indices using distributed computing.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import date, datetime
from collections import defaultdict

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
import numpy as np

from rsai.src.data.models import (
    BMNRegressionResult,
    IndexValue,
    GeographyLevel,
    WeightingScheme
)

logger = logging.getLogger(__name__)


class BMNRegression:
    """
    Implements Bailey-Muth-Nourse repeat sales regression using PySpark MLlib.
    
    The BMN method estimates price indices by regressing log price ratios
    on time dummy variables.
    """
    
    def __init__(
        self,
        spark: SparkSession,
        base_period: Optional[date] = None,
        frequency: str = "monthly",
        min_pairs_per_period: int = 10
    ):
        """
        Initialize BMN regression.
        
        Args:
            spark: SparkSession instance
            base_period: Base period for index (default: first period)
            frequency: Time frequency ('daily', 'monthly', 'quarterly')
            min_pairs_per_period: Minimum pairs required per period
        """
        self.spark = spark
        self.base_period = base_period
        self.frequency = frequency
        self.min_pairs_per_period = min_pairs_per_period
        self.results: Dict[str, BMNRegressionResult] = {}
        
    def fit(
        self,
        repeat_sales_df: DataFrame,
        geography_level: GeographyLevel,
        geography_id: str,
        weights_df: Optional[DataFrame] = None
    ) -> BMNRegressionResult:
        """
        Fit BMN regression model to repeat sales data.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales pairs
            geography_level: Geographic level for index
            geography_id: Geographic identifier
            weights_df: Optional DataFrame with observation weights
            
        Returns:
            BMNRegressionResult with regression output and index values
        """
        # Handle both string and enum geography_level
        if isinstance(geography_level, str):
            geography_level = GeographyLevel(geography_level)
        level_name = geography_level.value
        logger.info(f"Fitting BMN regression for {level_name} {geography_id}")
        
        # Prepare time periods
        periods_df = self._create_time_periods(repeat_sales_df)
        
        # Create features for regression
        regression_df = self._prepare_regression_data(
            repeat_sales_df, periods_df, weights_df
        )
        
        # Check for sufficient data
        n_obs = regression_df.count()
        if n_obs < self.min_pairs_per_period:
            raise ValueError(
                f"Insufficient data: {n_obs} pairs < {self.min_pairs_per_period} minimum"
            )
            
        # Fit regression model
        model, metrics = self._fit_regression(regression_df)
        
        # Extract coefficients
        coefficients = self._extract_coefficients(model, periods_df)
        
        # Calculate index values
        index_values = self._calculate_index_values(
            coefficients,
            periods_df,
            repeat_sales_df,
            geography_level,
            geography_id
        )
        
        # Create result object
        result = BMNRegressionResult(
            geography_level=geography_level,
            geography_id=geography_id,
            start_period=periods_df.agg(F.min("period")).collect()[0][0],
            end_period=periods_df.agg(F.max("period")).collect()[0][0],
            num_periods=periods_df.count(),
            num_observations=n_obs,
            r_squared=metrics.get("r2", 0.0),
            adj_r_squared=metrics.get("adj_r2", 0.0),
            coefficients=coefficients,
            standard_errors=metrics.get("standard_errors", {}),
            t_statistics=metrics.get("t_statistics", {}),
            p_values=metrics.get("p_values", {}),
            index_values=index_values
        )
        
        self.results[f"{geography_level.value}_{geography_id}"] = result
        return result
        
    def fit_multiple_geographies(
        self,
        repeat_sales_df: DataFrame,
        geography_col: str,
        geography_level: GeographyLevel,
        weights_df: Optional[DataFrame] = None,
        min_pairs: int = 30
    ) -> Dict[str, BMNRegressionResult]:
        """
        Fit BMN regression for multiple geographic areas in parallel.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales pairs
            geography_col: Column name for geographic identifier
            geography_level: Geographic level
            weights_df: Optional DataFrame with weights
            min_pairs: Minimum pairs required per geography
            
        Returns:
            Dictionary mapping geography ID to regression results
        """
        # Handle both string and enum geography_level
        if isinstance(geography_level, str):
            geography_level = GeographyLevel(geography_level)
        level_name = geography_level.value
        logger.info(f"Fitting BMN regression for multiple {level_name} areas")
        
        results = {}
        
        # Get unique geographies with sufficient data
        geo_counts = repeat_sales_df.groupBy(geography_col).agg(
            F.count("*").alias("num_pairs")
        ).filter(F.col("num_pairs") >= min_pairs).collect()
        
        logger.info(f"Processing {len(geo_counts)} geographic areas")
        
        # Process each geography
        for row in geo_counts:
            geo_id = row[geography_col]
            
            try:
                # Filter data for this geography
                geo_data = repeat_sales_df.filter(
                    F.col(geography_col) == geo_id
                )
                
                # Filter weights if provided
                geo_weights = None
                if weights_df is not None:
                    geo_weights = weights_df.filter(
                        F.col(geography_col) == geo_id
                    )
                    
                # Fit model
                result = self.fit(
                    geo_data,
                    geography_level,
                    str(geo_id),
                    geo_weights
                )
                
                results[str(geo_id)] = result
                
            except Exception as e:
                logger.error(f"Failed to fit model for {geo_id}: {str(e)}")
                continue
                
        logger.info(f"Successfully fitted {len(results)} models")
        return results
        
    def _create_time_periods(self, repeat_sales_df: DataFrame) -> DataFrame:
        """
        Create time period mapping based on frequency.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales
            
        Returns:
            DataFrame with unique time periods
        """
        # Extract all unique dates
        all_dates = repeat_sales_df.select("sale1_date").union(
            repeat_sales_df.select("sale2_date")
        ).distinct()
        
        # Convert to periods based on frequency
        if self.frequency == "monthly":
            periods_df = all_dates.withColumn(
                "period",
                F.date_trunc("month", F.col("sale1_date"))
            )
        elif self.frequency == "quarterly":
            periods_df = all_dates.withColumn(
                "period",
                F.date_trunc("quarter", F.col("sale1_date"))
            )
        else:  # daily
            periods_df = all_dates.withColumn(
                "period",
                F.col("sale1_date")
            )
            
        # Get unique periods and add index
        periods_df = periods_df.select("period").distinct().orderBy("period")
        periods_df = periods_df.withColumn(
            "period_index",
            F.row_number().over(Window.orderBy("period")) - 1
        )
        
        # Set base period if not specified
        if self.base_period is None:
            self.base_period = periods_df.first()["period"]
            
        return periods_df
        
    def _prepare_regression_data(
        self,
        repeat_sales_df: DataFrame,
        periods_df: DataFrame,
        weights_df: Optional[DataFrame] = None
    ) -> DataFrame:
        """
        Prepare data for regression with time dummies.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales
            periods_df: DataFrame with time periods
            weights_df: Optional weights
            
        Returns:
            DataFrame ready for regression
        """
        # Convert dates to periods
        if self.frequency == "monthly":
            data_df = repeat_sales_df.withColumn(
                "period1", F.date_trunc("month", F.col("sale1_date"))
            ).withColumn(
                "period2", F.date_trunc("month", F.col("sale2_date"))
            )
        elif self.frequency == "quarterly":
            data_df = repeat_sales_df.withColumn(
                "period1", F.date_trunc("quarter", F.col("sale1_date"))
            ).withColumn(
                "period2", F.date_trunc("quarter", F.col("sale2_date"))
            )
        else:
            data_df = repeat_sales_df.withColumn(
                "period1", F.col("sale1_date")
            ).withColumn(
                "period2", F.col("sale2_date")
            )
            
        # Join with period indices
        data_df = data_df.join(
            periods_df.select(
                F.col("period").alias("period1"),
                F.col("period_index").alias("period1_idx")
            ),
            on="period1"
        ).join(
            periods_df.select(
                F.col("period").alias("period2"),
                F.col("period_index").alias("period2_idx")
            ),
            on="period2"
        )
        
        # Add weight column if needed
        if weights_df is not None:
            data_df = data_df.join(
                weights_df.select("pair_id", "weight"),
                on="pair_id",
                how="left"
            ).fillna({"weight": 1.0})
        elif "weight" not in data_df.columns:
            data_df = data_df.withColumn("weight", F.lit(1.0))
            
        # Select necessary columns
        data_df = data_df.select(
            "pair_id",
            "log_price_ratio",
            "period1_idx",
            "period2_idx",
            "weight"
        )
        
        return data_df
        
    def _fit_regression(
        self,
        regression_df: DataFrame
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Fit the regression model using MLlib.
        
        Args:
            regression_df: Prepared regression data
            
        Returns:
            Tuple of (fitted model, metrics dictionary)
        """
        # Get number of periods
        max_period = regression_df.agg(
            F.max(F.greatest("period1_idx", "period2_idx"))
        ).collect()[0][0]
        
        # Create time dummy variables
        # For BMN, we need dummies for each period except the base
        # The design matrix has -1 for sale1 period and +1 for sale2 period
        
        # Create feature columns
        feature_exprs = []
        for i in range(max_period + 1):
            # Skip base period (assumed to be 0)
            if i == 0:
                continue
                
            feature_name = f"period_{i}"
            feature_expr = F.when(
                F.col("period2_idx") == i, 1.0
            ).when(
                F.col("period1_idx") == i, -1.0
            ).otherwise(0.0).alias(feature_name)
            
            feature_exprs.append(feature_expr)
            
        # Add features to DataFrame
        feature_cols = [f"period_{i}" for i in range(1, max_period + 1)]
        regression_df = regression_df.select("*", *feature_exprs)
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features"
        )
        
        assembled_df = assembler.transform(regression_df)
        
        # Create and fit regression model
        lr = LinearRegression(
            featuresCol="features",
            labelCol="log_price_ratio",
            weightCol="weight",
            regParam=0.0,  # No regularization for standard BMN
            elasticNetParam=0.0,
            standardization=False  # BMN doesn't standardize features
        )
        
        model = lr.fit(assembled_df)
        
        # Calculate metrics
        summary = model.summary
        
        # Handle NaN values in summary statistics
        import math
        r2 = summary.r2 if not math.isnan(summary.r2) else 0.0
        adj_r2 = summary.r2adj if hasattr(summary, 'r2adj') and not math.isnan(summary.r2adj) else r2
        
        metrics = {
            "r2": r2,
            "adj_r2": adj_r2,
            "rmse": summary.rootMeanSquaredError,
            "coefficients": model.coefficients.toArray().tolist(),
            "intercept": model.intercept
        }
        
        # Calculate standard errors if available
        try:
            if hasattr(summary, 'coefficientStandardErrors'):
                std_errors = summary.coefficientStandardErrors
                t_stats = [coef / se if se > 0 else 0.0 for coef, se in 
                          zip(model.coefficients.toArray(), std_errors)]
            else:
                raise Exception("No coefficientStandardErrors attribute")
        except Exception:
            # Standard errors not available (common with small datasets)
            std_errors = [0.0] * len(model.coefficients.toArray())
            t_stats = [0.0] * len(model.coefficients.toArray())
            
        metrics["standard_errors"] = {
            feature_cols[i]: std_errors[i] 
            for i in range(len(feature_cols))
        }
        metrics["t_statistics"] = {
            feature_cols[i]: t_stats[i] 
            for i in range(len(feature_cols))
        }
            
        return model, metrics
        
    def _extract_coefficients(
        self,
        model: Any,
        periods_df: DataFrame
    ) -> Dict[str, float]:
        """
        Extract coefficients for each time period.
        
        Args:
            model: Fitted regression model
            periods_df: DataFrame with periods
            
        Returns:
            Dictionary mapping period to coefficient
        """
        coefficients = {}
        
        # Get periods in order
        periods = periods_df.orderBy("period_index").collect()
        
        # Base period has coefficient 0
        base_period_str = self.base_period.strftime("%Y-%m-%d")
        coefficients[base_period_str] = 0.0
        
        # Extract other coefficients
        coef_array = model.coefficients.toArray()
        coef_idx = 0
        
        for row in periods:
            period = row["period"]
            period_idx = row["period_index"]
            
            # Skip base period
            if period == self.base_period:
                continue
                
            period_str = period.strftime("%Y-%m-%d")
            
            if coef_idx < len(coef_array):
                coefficients[period_str] = float(coef_array[coef_idx])
                coef_idx += 1
        
        # Ensure base period coefficient is exactly 0.0
        coefficients[base_period_str] = 0.0
                
        return coefficients
        
    def _calculate_index_values(
        self,
        coefficients: Dict[str, float],
        periods_df: DataFrame,
        repeat_sales_df: DataFrame,
        geography_level: GeographyLevel,
        geography_id: str
    ) -> List[IndexValue]:
        """
        Calculate index values from regression coefficients.
        
        Args:
            coefficients: Period coefficients
            periods_df: DataFrame with periods
            repeat_sales_df: Original repeat sales data
            geography_level: Geographic level
            geography_id: Geographic identifier
            
        Returns:
            List of IndexValue objects
        """
        index_values = []
        
        # Convert dates to periods for aggregation
        if self.frequency == "monthly":
            period_col = F.date_trunc("month", F.col("sale2_date"))
        elif self.frequency == "quarterly":
            period_col = F.date_trunc("quarter", F.col("sale2_date"))
        else:
            period_col = F.col("sale2_date")
            
        # Calculate statistics by period
        period_stats = repeat_sales_df.withColumn(
            "period", period_col
        ).groupBy("period").agg(
            F.count("*").alias("num_pairs"),
            F.countDistinct("property_id").alias("num_properties"),
            F.expr("percentile_approx(sale2_price, 0.5)").alias("median_price")
        ).collect()
        
        # Create lookup dictionary
        stats_dict = {row["period"]: row for row in period_stats}
        
        # Create index values for each period
        periods = periods_df.orderBy("period").collect()
        
        for row in periods:
            period = row["period"]
            period_str = period.strftime("%Y-%m-%d")
            
            # Get statistics for this period
            stats = stats_dict.get(period)
            if stats and stats["num_pairs"] > 0:
                # Only create index values for periods with actual data
                num_pairs = stats["num_pairs"]
                num_properties = stats["num_properties"] 
                median_price = stats["median_price"]
                
                # Calculate index value
                if period_str in coefficients:
                    # Index = 100 * exp(coefficient)
                    index_value = 100.0 * np.exp(coefficients[period_str])
                else:
                    index_value = 100.0  # Base period
                    
                index_values.append(IndexValue(
                    geography_level=geography_level,
                    geography_id=geography_id,
                    period=period,
                    index_value=index_value,
                    num_pairs=num_pairs,
                    num_properties=num_properties,
                    median_price=median_price,
                    standard_error=None,  # TODO: Calculate from regression
                    confidence_lower=None,
                    confidence_upper=None
                ))
            # Skip periods with no data (num_pairs = 0)
            
        return index_values
        
    def calculate_returns(
        self,
        index_values: List[IndexValue],
        return_type: str = "log"
    ) -> DataFrame:
        """
        Calculate returns from index values.
        
        Args:
            index_values: List of IndexValue objects
            return_type: Type of return ('log' or 'simple')
            
        Returns:
            DataFrame with returns
        """
        # Convert to DataFrame
        data = [
            {
                "period": iv.period,
                "index_value": iv.index_value,
                "geography_id": iv.geography_id
            }
            for iv in index_values
        ]
        
        index_df = self.spark.createDataFrame(data).orderBy("period")
        
        # Calculate returns
        window = Window.partitionBy("geography_id").orderBy("period")
        
        if return_type == "log":
            index_df = index_df.withColumn(
                "return",
                F.log(F.col("index_value")) - 
                F.lag(F.log(F.col("index_value"))).over(window)
            )
        else:
            index_df = index_df.withColumn(
                "return",
                F.when(F.lag(F.col("index_value")).over(window) > 0,
                    (F.col("index_value") / 
                     F.lag(F.col("index_value")).over(window)) - 1
                ).otherwise(F.lit(None))
            )
            
        # Add annualized returns
        if self.frequency == "monthly":
            periods_per_year = 12
        elif self.frequency == "quarterly":
            periods_per_year = 4
        else:
            periods_per_year = 365
            
        index_df = index_df.withColumn(
            "annualized_return",
            F.col("return") * periods_per_year
        )
        
        return index_df