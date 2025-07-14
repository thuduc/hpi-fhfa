"""
Weighting schemes for index aggregation using PySpark.

This module implements various weighting schemes including equal weights,
value weights, Case-Shiller weights, and custom weighting functions.
"""

import logging
from typing import Optional, Dict, Any, Callable
from datetime import date
import numpy as np

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

from rsai.src.data.models import WeightingScheme

logger = logging.getLogger(__name__)


class WeightCalculator:
    """Calculate weights for repeat sales observations using PySpark."""
    
    def __init__(
        self,
        spark: SparkSession,
        scheme: WeightingScheme = WeightingScheme.EQUAL
    ):
        """
        Initialize weight calculator.
        
        Args:
            spark: SparkSession instance
            scheme: Weighting scheme to use
        """
        self.spark = spark
        # Convert string to enum if needed
        if isinstance(scheme, str):
            self.scheme = WeightingScheme(scheme)
        else:
            self.scheme = scheme
        
        # Register weight calculation UDFs
        self._register_udfs()
        
    def _register_udfs(self):
        """Register User Defined Functions for weight calculations."""
        # BMN temporal weight function
        def bmn_temporal_weight(holding_days, age_years=None):
            """Calculate BMN weight based on holding period and age."""
            if holding_days is None or holding_days <= 0:
                return 0.0
                
            # Base weight decreases with holding period
            base_weight = 1.0 / np.sqrt(holding_days / 365.0)
            
            # Age adjustment if provided
            if age_years is not None and age_years > 0:
                age_factor = 1.0 / (1.0 + age_years / 20.0)
                base_weight *= age_factor
                
            return float(base_weight)
            
        self.bmn_weight_udf = F.udf(bmn_temporal_weight, DoubleType())
        
        # Case-Shiller interval weight function
        def cs_interval_weight(holding_days):
            """Calculate Case-Shiller interval weight."""
            if holding_days is None or holding_days <= 0:
                return 0.0
            return float(1.0 / (holding_days / 365.0))
            
        self.cs_interval_udf = F.udf(cs_interval_weight, DoubleType())
        
    def calculate_weights(
        self,
        repeat_sales_df: DataFrame,
        weight_col: str = "weight",
        **kwargs
    ) -> DataFrame:
        """
        Calculate weights based on the configured scheme.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales
            weight_col: Name for weight column
            **kwargs: Additional arguments for specific schemes
            
        Returns:
            DataFrame with weights added
        """
        logger.info(f"Calculating {self.scheme.value} weights")
        
        if self.scheme == WeightingScheme.EQUAL:
            return self._equal_weights(repeat_sales_df, weight_col)
        elif self.scheme == WeightingScheme.VALUE:
            return self._value_weights(repeat_sales_df, weight_col)
        elif self.scheme == WeightingScheme.CASE_SHILLER:
            return self._case_shiller_weights(repeat_sales_df, weight_col, **kwargs)
        elif self.scheme == WeightingScheme.BMN:
            return self._bmn_weights(repeat_sales_df, weight_col, **kwargs)
        elif self.scheme == WeightingScheme.CUSTOM:
            return self._custom_weights(repeat_sales_df, weight_col, **kwargs)
        else:
            raise ValueError(f"Unknown weighting scheme: {self.scheme}")
            
    def _equal_weights(
        self,
        repeat_sales_df: DataFrame,
        weight_col: str
    ) -> DataFrame:
        """Apply equal weights to all observations."""
        return repeat_sales_df.withColumn(
            weight_col, F.lit(1.0)
        ).withColumn(
            "weight_type", F.lit("equal")
        )
        
    def _value_weights(
        self,
        repeat_sales_df: DataFrame,
        weight_col: str
    ) -> DataFrame:
        """
        Apply value-based weights.
        
        Higher value properties get higher weights.
        """
        # Calculate average sale value
        df = repeat_sales_df.withColumn(
            "avg_value",
            (F.col("sale1_price") + F.col("sale2_price")) / 2.0
        )
        
        # Calculate median value for normalization
        median_value = df.agg(
            F.expr("percentile_approx(avg_value, 0.5)")
        ).collect()[0][0]
        
        # Weight proportional to value, normalized to median (safe division)
        if median_value is not None and median_value > 0:
            df = df.withColumn(
                weight_col,
                F.col("avg_value") / median_value
            )
        else:
            df = df.withColumn(
                weight_col,
                F.lit(1.0)
            )
        
        # Cap extreme weights
        df = df.withColumn(
            weight_col,
            F.when(F.col(weight_col) > 5.0, 5.0)
            .when(F.col(weight_col) < 0.2, 0.2)
            .otherwise(F.col(weight_col))
        ).withColumn(
            "weight_type", F.lit("value")
        )
        
        return df.drop("avg_value")
        
    def _case_shiller_weights(
        self,
        repeat_sales_df: DataFrame,
        weight_col: str,
        interval_correction: bool = True,
        heteroscedasticity_correction: bool = True
    ) -> DataFrame:
        """
        Apply Case-Shiller three-stage weighting.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales
            weight_col: Name for weight column
            interval_correction: Apply interval weighting
            heteroscedasticity_correction: Apply heteroscedasticity correction
            
        Returns:
            DataFrame with Case-Shiller weights
        """
        df = repeat_sales_df
        
        # Stage 1: Interval weights (inverse of time between sales)
        if interval_correction:
            df = df.withColumn(
                "interval_weight",
                self.cs_interval_udf(F.col("holding_period_days"))
            )
        else:
            df = df.withColumn("interval_weight", F.lit(1.0))
            
        # Stage 2: Heteroscedasticity correction
        if heteroscedasticity_correction:
            # Estimate variance as function of price level
            df = df.withColumn(
                "log_avg_price",
                F.log((F.col("sale1_price") + F.col("sale2_price")) / 2.0)
            )
            
            # Bin by price level and calculate variance
            price_bins = df.select(
                F.expr("percentile_approx(log_avg_price, 0.1)").alias("p10"),
                F.expr("percentile_approx(log_avg_price, 0.9)").alias("p90")
            ).collect()[0]
            
            df = df.withColumn(
                "price_bin",
                F.when(F.col("log_avg_price") < price_bins["p10"], 0)
                .when(F.col("log_avg_price") > price_bins["p90"], 2)
                .otherwise(1)
            )
            
            # Calculate variance by bin
            variance_df = df.groupBy("price_bin").agg(
                F.variance("log_price_ratio").alias("variance")
            )
            
            df = df.join(variance_df, on="price_bin", how="left")
            
            # Heteroscedasticity weight is inverse of variance
            df = df.withColumn(
                "hetero_weight",
                F.when(F.col("variance") > 0, 1.0 / F.sqrt(F.col("variance")))
                .otherwise(1.0)
            )
            
            df = df.drop("log_avg_price", "price_bin", "variance")
        else:
            df = df.withColumn("hetero_weight", F.lit(1.0))
            
        # Combine weights
        df = df.withColumn(
            weight_col,
            F.col("interval_weight") * F.col("hetero_weight")
        )
        
        # Normalize weights to have mean 1 (safe division)
        mean_weight = df.agg(F.mean(weight_col)).collect()[0][0]
        if mean_weight > 0:
            df = df.withColumn(
                weight_col,
                F.col(weight_col) / mean_weight
            )
        # If mean_weight is 0, keep original weights
        
        df = df.withColumn(
            "weight_type", F.lit("case_shiller")
        )
        
        return df.drop("interval_weight", "hetero_weight")
        
    def _bmn_weights(
        self,
        repeat_sales_df: DataFrame,
        weight_col: str,
        age_correction: bool = False
    ) -> DataFrame:
        """
        Apply BMN temporal weights.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales
            weight_col: Name for weight column
            age_correction: Include property age in weighting
            
        Returns:
            DataFrame with BMN weights
        """
        if age_correction and "property_age" in repeat_sales_df.columns:
            df = repeat_sales_df.withColumn(
                weight_col,
                self.bmn_weight_udf(
                    F.col("holding_period_days"),
                    F.col("property_age")
                )
            )
        else:
            df = repeat_sales_df.withColumn(
                weight_col,
                self.bmn_weight_udf(
                    F.col("holding_period_days"),
                    F.lit(None)
                )
            )
            
        # Normalize weights (safe division)
        mean_weight = df.agg(F.mean(weight_col)).collect()[0][0]
        if mean_weight > 0:
            df = df.withColumn(
                weight_col,
                F.col(weight_col) / mean_weight
            )
        
        df = df.withColumn(
            "weight_type", F.lit("bmn")
        )
        
        return df
        
    def _custom_weights(
        self,
        repeat_sales_df: DataFrame,
        weight_col: str,
        weight_func: Optional[Callable] = None,
        **kwargs
    ) -> DataFrame:
        """
        Apply custom weighting function.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales
            weight_col: Name for weight column
            weight_func: Custom weight function (UDF)
            **kwargs: Additional arguments for weight function
            
        Returns:
            DataFrame with custom weights
        """
        if weight_func is None:
            raise ValueError("No custom weight function provided")
            
        # Apply custom function
        df = repeat_sales_df.withColumn(
            weight_col,
            weight_func(*[F.col(col) for col in kwargs.get("weight_cols", [])])
        ).withColumn(
            "weight_type", F.lit("custom")
        )
        
        return df
        
    def geographic_weights(
        self,
        repeat_sales_df: DataFrame,
        target_lat: float,
        target_lon: float,
        decay_distance_km: float = 10.0,
        min_weight: float = 0.1
    ) -> DataFrame:
        """
        Apply geographic distance-based weights.
        
        Properties closer to target location get higher weights.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales
            target_lat: Target latitude
            target_lon: Target longitude
            decay_distance_km: Distance decay parameter
            min_weight: Minimum weight
            
        Returns:
            DataFrame with geographic weights
        """
        # Import distance calculator
        from rsai.src.geography.distance import DistanceCalculator
        
        dist_calc = DistanceCalculator(self.spark)
        
        # Calculate distance to target
        df = dist_calc.add_distance_to_point(
            repeat_sales_df,
            target_lat,
            target_lon,
            lat_col="latitude",
            lon_col="longitude",
            distance_col="distance_km"
        )
        
        # Exponential decay weight
        df = df.withColumn(
            "weight",
            F.greatest(
                F.lit(min_weight),
                F.exp(-F.col("distance_km") / decay_distance_km)
            )
        ).withColumn(
            "weight_type", F.lit("geographic")
        )
        
        return df
        
    def temporal_weights(
        self,
        repeat_sales_df: DataFrame,
        reference_date: date,
        decay_years: float = 2.0,
        forward_weight: float = 0.5
    ) -> DataFrame:
        """
        Apply temporal distance weights.
        
        Sales closer to reference date get higher weights.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales
            reference_date: Reference date
            decay_years: Temporal decay parameter
            forward_weight: Weight for future sales vs past
            
        Returns:
            DataFrame with temporal weights
        """
        df = repeat_sales_df.withColumn(
            "days_from_ref",
            F.datediff(F.col("sale2_date"), F.lit(reference_date))
        )
        
        # Different decay for past and future
        df = df.withColumn(
            "weight",
            F.when(
                F.col("days_from_ref") >= 0,
                forward_weight * F.exp(-F.abs(F.col("days_from_ref")) / (decay_years * 365))
            ).otherwise(
                F.exp(-F.abs(F.col("days_from_ref")) / (decay_years * 365))
            )
        ).withColumn(
            "weight_type", F.lit("temporal")
        )
        
        return df.drop("days_from_ref")
        
    def quality_adjusted_weights(
        self,
        repeat_sales_df: DataFrame,
        quality_scores: Optional[DataFrame] = None,
        min_quality: float = 0.1
    ) -> DataFrame:
        """
        Apply quality-based weights.
        
        Higher quality data gets higher weights.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales
            quality_scores: Optional DataFrame with quality scores
            min_quality: Minimum quality weight
            
        Returns:
            DataFrame with quality weights
        """
        if quality_scores is not None:
            # Join with provided quality scores
            df = repeat_sales_df.join(
                quality_scores.select("pair_id", "quality_score"),
                on="pair_id",
                how="left"
            ).fillna({"quality_score": 0.5})
        else:
            # Calculate quality score based on data completeness
            df = repeat_sales_df.withColumn(
                "quality_score",
                F.when(
                    F.col("tract").isNotNull() &
                    F.col("property_type").isNotNull() &
                    (F.size(F.col("validation_flags")) == 0),
                    1.0
                ).when(
                    F.col("tract").isNotNull(),
                    0.8
                ).otherwise(0.5)
            )
            
        # Apply quality weight with minimum
        df = df.withColumn(
            "weight",
            F.greatest(F.lit(min_quality), F.col("quality_score"))
        ).withColumn(
            "weight_type", F.lit("quality")
        ).drop("quality_score")
        
        return df
        
    def combine_weights(
        self,
        repeat_sales_df: DataFrame,
        weight_schemes: Dict[WeightingScheme, float]
    ) -> DataFrame:
        """
        Combine multiple weighting schemes.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales
            weight_schemes: Dictionary of scheme to weight
            
        Returns:
            DataFrame with combined weights
        """
        # Normalize scheme weights
        total_weight = sum(weight_schemes.values())
        normalized_weights = {
            k: v/total_weight for k, v in weight_schemes.items()
        }
        
        # Calculate each weight type
        combined_df = repeat_sales_df
        weight_cols = []
        
        for scheme, scheme_weight in normalized_weights.items():
            # Create calculator for this scheme
            calc = WeightCalculator(self.spark, scheme)
            
            # Calculate weights
            temp_df = calc.calculate_weights(
                repeat_sales_df,
                weight_col=f"weight_{scheme.value}"
            )
            
            # Extract weight column
            weight_col = f"weight_{scheme.value}"
            combined_df = combined_df.join(
                temp_df.select("pair_id", weight_col),
                on="pair_id",
                how="left"
            )
            
            weight_cols.append((weight_col, scheme_weight))
            
        # Combine weights
        weight_expr = sum(
            F.col(col) * weight for col, weight in weight_cols
        )
        
        combined_df = combined_df.withColumn(
            "weight", weight_expr
        ).withColumn(
            "weight_type", F.lit("combined")
        )
        
        # Drop temporary columns
        for col, _ in weight_cols:
            combined_df = combined_df.drop(col)
            
        return combined_df