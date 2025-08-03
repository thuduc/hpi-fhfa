"""Bailey-Muth-Nourse (BMN) regression implementation using PySpark MLlib"""

from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

from ..utils.logging_config import setup_logging


class BMNRegression:
    """
    Implements Bailey-Muth-Nourse regression for repeat-sales index estimation
    using PySpark MLlib's distributed linear regression
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = setup_logging(self.__class__.__name__)
        
    def prepare_regression_data(
        self, 
        repeat_sales: DataFrame,
        supertract_id: str,
        start_year: int,
        end_year: int
    ) -> Tuple[DataFrame, List[int]]:
        """
        Prepare data for BMN regression with dummy variables
        
        Args:
            repeat_sales: DataFrame with repeat-sales data
            supertract_id: ID of supertract to process
            start_year: Start year for analysis
            end_year: End year for analysis
            
        Returns:
            Tuple of (regression DataFrame, list of periods)
        """
        self.logger.debug(f"Preparing regression data for supertract {supertract_id}")
        
        # Filter for supertract and time period
        supertract_data = repeat_sales.filter(
            (F.col("supertract_id") == supertract_id) &
            (F.year("sale_date_1") >= start_year) &
            (F.year("sale_date_2") <= end_year)
        )
        
        # Create time period list
        periods = list(range(start_year, end_year + 1))
        num_periods = len(periods)
        
        # Create UDF for dummy variable generation
        @F.udf(returnType=VectorUDT())
        def create_dummy_vector(year1: int, year2: int):
            """Create sparse vector of time dummies for BMN regression"""
            indices = []
            values = []
            
            # Find indices for sale years
            if year1 in periods:
                idx1 = periods.index(year1)
                indices.append(idx1)
                values.append(-1.0)
                
            if year2 in periods:
                idx2 = periods.index(year2)
                indices.append(idx2)
                values.append(1.0)
            
            return Vectors.sparse(num_periods, indices, values)
        
        # Prepare regression dataset
        regression_data = supertract_data.withColumn(
            "features",
            create_dummy_vector(
                F.year("sale_date_1"),
                F.year("sale_date_2")
            )
        ).select(
            F.col("price_relative").alias("label"),
            F.col("features"),
            F.col("property_id"),  # Keep for diagnostics
            F.year("sale_date_1").alias("year1"),
            F.year("sale_date_2").alias("year2")
        )
        
        return regression_data, periods
    
    def estimate_bmn(
        self, 
        regression_data: DataFrame,
        elastic_net_param: float = 0.0,
        reg_param: float = 0.0,
        max_iter: int = 100
    ) -> Dict[str, any]:
        """
        Estimate BMN regression using MLlib LinearRegression
        
        Args:
            regression_data: Prepared regression DataFrame
            elastic_net_param: Elastic net mixing parameter (0 = ridge, 1 = lasso)
            reg_param: Regularization parameter
            max_iter: Maximum iterations for solver
            
        Returns:
            Dictionary with regression results
        """
        obs_count = regression_data.count()
        
        if obs_count < 10:
            self.logger.warning(
                f"Insufficient observations ({obs_count}) for regression"
            )
            return None
            
        self.logger.debug(f"Estimating BMN regression with {obs_count} observations")
        
        # Configure regression
        lr = LinearRegression(
            elasticNetParam=elastic_net_param,
            regParam=reg_param,
            standardization=False,
            fitIntercept=False,  # No intercept in BMN
            maxIter=max_iter,
            solver="normal"  # Use normal equations for exact solution when possible
        )
        
        try:
            # Fit model
            model = lr.fit(regression_data)
            
            # Extract coefficients
            coefficients = model.coefficients.toArray()
            
            # Get model summary for diagnostics
            summary = model.summary
            
            results = {
                "coefficients": coefficients,
                "r2": summary.r2,
                "rmse": summary.rootMeanSquaredError,
                "num_observations": obs_count,
                "num_parameters": len(coefficients),
                "degrees_of_freedom": obs_count - len(coefficients)
            }
            
            # Add coefficient standard errors if available
            try:
                # This might fail for some solver configurations
                std_errors = summary.coefficientStandardErrors
                results["std_errors"] = std_errors
            except:
                self.logger.debug("Standard errors not available")
                results["std_errors"] = None
            
            self.logger.debug(
                f"Regression complete: R² = {results['r2']:.4f}, "
                f"RMSE = {results['rmse']:.4f}"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Regression failed: {str(e)}")
            return None
    
    def calculate_appreciation_rates(
        self, 
        bmn_results: Dict[str, any],
        periods: List[int]
    ) -> DataFrame:
        """
        Calculate period-to-period appreciation rates from BMN coefficients
        
        Args:
            bmn_results: Results from BMN regression
            periods: List of time periods
            
        Returns:
            DataFrame with appreciation rates by period
        """
        if not bmn_results or "coefficients" not in bmn_results:
            return None
            
        coefficients = bmn_results["coefficients"]
        
        # Calculate appreciation rates (difference in adjacent coefficients)
        appreciation_data = []
        
        # First period has no appreciation (base period)
        appreciation_data.append({
            "year": periods[0],
            "coefficient": float(coefficients[0]) if len(coefficients) > 0 else 0.0,
            "appreciation_rate": 0.0,
            "cumulative_index": 100.0  # Base = 100
        })
        
        # Calculate for remaining periods
        cumulative_index = 100.0
        for i in range(1, len(periods)):
            if i < len(coefficients):
                appreciation = float(coefficients[i] - coefficients[i-1])
                cumulative_index *= np.exp(appreciation)
                
                appreciation_data.append({
                    "year": periods[i],
                    "coefficient": float(coefficients[i]),
                    "appreciation_rate": appreciation,
                    "cumulative_index": float(cumulative_index)
                })
        
        return self.spark.createDataFrame(appreciation_data)
    
    def run_bmn_for_supertract(
        self,
        repeat_sales: DataFrame,
        supertract_id: str,
        start_year: int,
        end_year: int,
        **regression_params
    ) -> Optional[Dict[str, any]]:
        """
        Complete BMN regression pipeline for a single supertract
        
        Args:
            repeat_sales: DataFrame with repeat-sales data
            supertract_id: ID of supertract to process
            start_year: Start year for analysis
            end_year: End year for analysis
            **regression_params: Additional parameters for regression
            
        Returns:
            Dictionary with complete results or None if failed
        """
        # Prepare data
        regression_data, periods = self.prepare_regression_data(
            repeat_sales, supertract_id, start_year, end_year
        )
        
        # Check if we have enough data
        obs_count = regression_data.count()
        if obs_count < 10:
            self.logger.warning(
                f"Supertract {supertract_id}: Insufficient data ({obs_count} obs)"
            )
            return None
        
        # Estimate regression
        bmn_results = self.estimate_bmn(regression_data, **regression_params)
        
        if bmn_results:
            # Add appreciation rates
            appreciation_df = self.calculate_appreciation_rates(bmn_results, periods)
            
            return {
                "supertract_id": supertract_id,
                "regression_results": bmn_results,
                "appreciation_rates": appreciation_df,
                "periods": periods
            }
        
        return None
    
    def batch_process_supertracts(
        self,
        repeat_sales: DataFrame,
        supertracts: DataFrame,
        year: int,
        **regression_params
    ) -> DataFrame:
        """
        Process multiple supertracts in batch for a given year
        
        Args:
            repeat_sales: DataFrame with repeat-sales data
            supertracts: DataFrame with supertract definitions
            year: Year to process
            **regression_params: Additional parameters for regression
            
        Returns:
            DataFrame with appreciation rates for all supertracts
        """
        self.logger.info(f"Batch processing BMN regressions for year {year}")
        
        # Get list of supertracts with cbsa_code
        supertract_list = supertracts.select("supertract_id", "cbsa_code").distinct().collect()
        
        results = []
        successful = 0
        failed = 0
        
        for row in supertract_list:
            supertract_id = row["supertract_id"]
            cbsa_code = row["cbsa_code"]
            
            # Run regression for period [year-1, year]
            result = self.run_bmn_for_supertract(
                repeat_sales,
                supertract_id,
                year - 1,
                year,
                **regression_params
            )
            
            if result and result["appreciation_rates"]:
                # Extract appreciation for the target year
                appreciation_df = result["appreciation_rates"]
                year_appreciation = appreciation_df.filter(
                    F.col("year") == year
                ).first()
                
                if year_appreciation:
                    results.append({
                        "supertract_id": supertract_id,
                        "cbsa_code": cbsa_code,
                        "year": year,
                        "appreciation_rate": year_appreciation["appreciation_rate"],
                        "r2": result["regression_results"]["r2"],
                        "rmse": result["regression_results"]["rmse"],
                        "num_observations": result["regression_results"]["num_observations"]
                    })
                    successful += 1
                else:
                    failed += 1
            else:
                failed += 1
        
        self.logger.info(
            f"Completed BMN regressions: {successful} successful, {failed} failed"
        )
        
        if results:
            return self.spark.createDataFrame(results)
        else:
            # Return empty DataFrame with proper schema
            from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
            schema = StructType([
                StructField("supertract_id", StringType(), False),
                StructField("cbsa_code", StringType(), False),
                StructField("year", IntegerType(), False),
                StructField("appreciation_rate", DoubleType(), True),
                StructField("r2", DoubleType(), True),
                StructField("rmse", DoubleType(), True),
                StructField("num_observations", IntegerType(), True)
            ])
            return self.spark.createDataFrame([], schema)