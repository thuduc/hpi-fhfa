"""
Weighting schemes for repeat sales index calculation using Polars.

This module implements various weighting schemes including equal weights,
value weights, Case-Shiller weights, and custom weighting functions.
"""

import logging
from typing import Optional, Dict, Any, Callable
from datetime import date
import numpy as np

import polars as pl

from rsai.src.data.models import WeightingScheme

logger = logging.getLogger(__name__)


class WeightCalculator:
    """Calculate weights for repeat sales pairs using various schemes."""
    
    def __init__(self, scheme: WeightingScheme = WeightingScheme.EQUAL):
        """
        Initialize weight calculator.
        
        Args:
            scheme: Weighting scheme to use
        """
        # Handle both enum and string inputs
        if isinstance(scheme, str):
            self.scheme = WeightingScheme(scheme)
        else:
            self.scheme = scheme
            
        self.weight_functions = {
            WeightingScheme.EQUAL: self.equal_weights,
            WeightingScheme.VALUE: self.value_weights,
            WeightingScheme.CASE_SHILLER: self.case_shiller_weights,
            WeightingScheme.BMN: self.bmn_weights,
            WeightingScheme.CUSTOM: self.custom_weights
        }
        self.custom_weight_func: Optional[Callable] = None
        
    def calculate_weights(
        self,
        repeat_sales_df: pl.DataFrame,
        **kwargs
    ) -> pl.DataFrame:
        """
        Calculate weights for repeat sales pairs.
        
        Args:
            repeat_sales_df: Polars DataFrame with repeat sales
            **kwargs: Additional arguments for specific weight schemes
            
        Returns:
            Polars DataFrame with weights added
        """
        logger.info(f"Calculating {self.scheme.value if hasattr(self.scheme, 'value') else self.scheme} weights")
        
        weight_func = self.weight_functions.get(self.scheme)
        if weight_func is None:
            raise ValueError(f"Unknown weighting scheme: {self.scheme}")
            
        return weight_func(repeat_sales_df, **kwargs)
        
    def equal_weights(
        self,
        repeat_sales_df: pl.DataFrame,
        **kwargs
    ) -> pl.DataFrame:
        """
        Apply equal weights to all observations.
        
        Args:
            repeat_sales_df: Repeat sales DataFrame
            
        Returns:
            DataFrame with equal weights
        """
        return repeat_sales_df.with_columns([
            pl.lit(1.0).alias("weight"),
            pl.lit("equal").alias("weight_type")
        ])
        
    def value_weights(
        self,
        repeat_sales_df: pl.DataFrame,
        value_col: str = "sale_value",
        **kwargs
    ) -> pl.DataFrame:
        """
        Apply weights based on transaction values.
        
        Higher value transactions get more weight.
        
        Args:
            repeat_sales_df: Repeat sales DataFrame
            value_col: Column to use for value weighting
            
        Returns:
            DataFrame with value-based weights
        """
        # Calculate average sale value for each pair
        df = repeat_sales_df.with_columns([
            ((pl.col("sale1_price") + pl.col("sale2_price")) / 2).alias("avg_value")
        ])
        
        # Normalize by median to avoid extreme weights
        median_value = df["avg_value"].median()
        
        df = df.with_columns([
            (pl.col("avg_value") / median_value).alias("weight"),
            pl.lit("value").alias("weight_type")
        ])
        
        # Cap weights to avoid outliers dominating
        max_weight = df["weight"].quantile(0.95)
        df = df.with_columns([
            pl.when(pl.col("weight") > max_weight)
            .then(max_weight)
            .otherwise(pl.col("weight"))
            .alias("weight")
        ])
        
        return df
        
    def case_shiller_weights(
        self,
        repeat_sales_df: pl.DataFrame,
        interval_correction: bool = True,
        heteroscedasticity_correction: bool = True,
        **kwargs
    ) -> pl.DataFrame:
        """
        Apply Case-Shiller three-stage weighting.
        
        Includes corrections for:
        1. Time interval between sales
        2. Heteroscedasticity related to price levels
        
        Args:
            repeat_sales_df: Repeat sales DataFrame
            interval_correction: Apply interval weighting
            heteroscedasticity_correction: Apply heteroscedasticity correction
            
        Returns:
            DataFrame with Case-Shiller weights
        """
        df = repeat_sales_df
        
        # Stage 1: Initial regression (equal weights)
        # This would typically use the BMN regression results
        # For now, we'll calculate interval-based weights
        
        if interval_correction:
            # Weight inversely proportional to time interval
            # Longer intervals have more noise
            df = df.with_columns([
                (1.0 / pl.col("holding_period_days").cast(pl.Float64).sqrt()).alias("interval_weight")
            ])
        else:
            df = df.with_columns([
                pl.lit(1.0).alias("interval_weight")
            ])
            
        # Stage 2: Heteroscedasticity correction
        if heteroscedasticity_correction:
            # Calculate price-level based adjustment
            # Higher priced properties tend to have more volatile returns
            
            # Use log of average price
            df = df.with_columns([
                ((pl.col("sale1_price") + pl.col("sale2_price")) / 2).log().alias("log_avg_price")
            ])
            
            # Estimate heteroscedasticity pattern
            # Simple approach: weight inversely with price level
            price_10 = df["log_avg_price"].quantile(0.1)
            price_90 = df["log_avg_price"].quantile(0.9)
            price_range = price_90 - price_10
            
            df = df.with_columns([
                (1.0 - 0.5 * (pl.col("log_avg_price") - price_10) / price_range)
                .clip(0.5, 1.5)
                .alias("hetero_weight")
            ])
        else:
            df = df.with_columns([
                pl.lit(1.0).alias("hetero_weight")
            ])
            
        # Combine weights
        df = df.with_columns([
            (pl.col("interval_weight") * pl.col("hetero_weight")).alias("weight"),
            pl.lit("case_shiller").alias("weight_type")
        ])
        
        # Normalize weights to have mean 1
        mean_weight = df["weight"].mean()
        df = df.with_columns([
            (pl.col("weight") / mean_weight).alias("weight")
        ])
        
        # Drop intermediate columns
        cols_to_drop = ["interval_weight", "hetero_weight"]
        if heteroscedasticity_correction:
            cols_to_drop.append("log_avg_price")
        df = df.drop(cols_to_drop)
        
        return df
        
    def bmn_weights(
        self,
        repeat_sales_df: pl.DataFrame,
        age_correction: bool = True,
        **kwargs
    ) -> pl.DataFrame:
        """
        Apply BMN-style weights with age correction.
        
        Args:
            repeat_sales_df: Repeat sales DataFrame
            age_correction: Apply age-based correction
            
        Returns:
            DataFrame with BMN weights
        """
        df = repeat_sales_df
        
        # Basic interval weighting
        df = df.with_columns([
            (1.0 / (1.0 + pl.col("holding_period_days") / 365.0)).alias("base_weight")
        ])
        
        # Age correction if property age available
        if age_correction and "property_age" in df.columns:
            # Older properties may have more measurement error
            df = df.with_columns([
                (1.0 / (1.0 + pl.col("property_age") / 50.0)).alias("age_weight")
            ])
            
            df = df.with_columns([
                (pl.col("base_weight") * pl.col("age_weight")).alias("weight")
            ])
            
            df = df.drop(["age_weight"])
        else:
            df = df.with_columns([
                pl.col("base_weight").alias("weight")
            ])
            
        df = df.with_columns([
            pl.lit("bmn").alias("weight_type")
        ]).drop(["base_weight"])
        
        # Normalize
        mean_weight = df["weight"].mean()
        df = df.with_columns([
            (pl.col("weight") / mean_weight).alias("weight")
        ])
        
        return df
        
    def custom_weights(
        self,
        repeat_sales_df: pl.DataFrame,
        weight_func: Optional[Callable] = None,
        **kwargs
    ) -> pl.DataFrame:
        """
        Apply custom weighting function.
        
        Args:
            repeat_sales_df: Repeat sales DataFrame
            weight_func: Custom weight function
            
        Returns:
            DataFrame with custom weights
        """
        if weight_func is None:
            weight_func = self.custom_weight_func
            
        if weight_func is None:
            raise ValueError("No custom weight function provided")
            
        # Apply custom function
        weights = weight_func(repeat_sales_df, **kwargs)
        
        # Add weights to DataFrame
        if isinstance(weights, pl.Series):
            df = repeat_sales_df.with_columns([
                weights.alias("weight"),
                pl.lit("custom").alias("weight_type")
            ])
        elif isinstance(weights, np.ndarray):
            df = repeat_sales_df.with_columns([
                pl.Series("weight", weights),
                pl.lit("custom").alias("weight_type")
            ])
        else:
            raise ValueError("Custom weight function must return Series or array")
            
        return df
        
    def geographic_weights(
        self,
        repeat_sales_df: pl.DataFrame,
        target_lat: float,
        target_lon: float,
        decay_distance_km: float = 10.0,
        min_weight: float = 0.1,
        **kwargs
    ) -> pl.DataFrame:
        """
        Apply geographic distance-based weights.
        
        Properties closer to target location get higher weights.
        
        Args:
            repeat_sales_df: Repeat sales DataFrame
            target_lat: Target latitude
            target_lon: Target longitude
            decay_distance_km: Distance at which weight decays to 0.5
            min_weight: Minimum weight
            
        Returns:
            DataFrame with geographic weights
        """
        from rsai.src.geography.distance import haversine_distance
        
        # Calculate distance from target for each property
        if not all(col in repeat_sales_df.columns for col in ["latitude", "longitude"]):
            raise ValueError("Geographic coordinates required for geographic weights")
            
        # Calculate distances
        distances = []
        for row in repeat_sales_df.iter_rows(named=True):
            if row["latitude"] and row["longitude"]:
                dist = haversine_distance(
                    target_lat, target_lon,
                    row["latitude"], row["longitude"],
                    "km"
                )
            else:
                dist = decay_distance_km * 10  # Penalize missing coordinates
                
            distances.append(dist)
            
        # Convert distances to weights using exponential decay
        distances = np.array(distances)
        weights = np.exp(-distances / decay_distance_km)
        weights = np.maximum(weights, min_weight)
        
        # Add to DataFrame
        df = repeat_sales_df.with_columns([
            pl.Series("weight", weights),
            pl.Series("distance_km", distances),
            pl.lit("geographic").alias("weight_type")
        ])
        
        return df
        
    def temporal_weights(
        self,
        repeat_sales_df: pl.DataFrame,
        reference_date: date,
        decay_years: float = 2.0,
        forward_weight: float = 0.5,
        **kwargs
    ) -> pl.DataFrame:
        """
        Apply temporal distance-based weights.
        
        Sales closer to reference date get higher weights.
        
        Args:
            repeat_sales_df: Repeat sales DataFrame
            reference_date: Reference date for weighting
            decay_years: Years at which weight decays to 0.5
            forward_weight: Weight for future sales relative to past
            
        Returns:
            DataFrame with temporal weights
        """
        # Calculate temporal distance for each sale
        df = repeat_sales_df.with_columns([
            ((pl.col("sale2_date") - reference_date).dt.total_milliseconds() / (365.25 * 24 * 60 * 60 * 1000)).alias("years_diff")
        ])
        
        # Apply asymmetric exponential decay
        df = df.with_columns([
            pl.when(pl.col("years_diff") >= 0)
            .then(pl.col("years_diff").abs() / decay_years * -forward_weight)
            .otherwise(pl.col("years_diff").abs() / decay_years * -1)
            .exp()
            .alias("weight")
        ])
        
        df = df.with_columns([
            pl.lit("temporal").alias("weight_type")
        ]).drop(["years_diff"])
        
        return df
        
    def quality_adjusted_weights(
        self,
        repeat_sales_df: pl.DataFrame,
        quality_scores: Optional[pl.DataFrame] = None,
        min_quality: float = 0.1,
        **kwargs
    ) -> pl.DataFrame:
        """
        Apply weights based on data quality scores.
        
        Args:
            repeat_sales_df: Repeat sales DataFrame
            quality_scores: DataFrame with quality scores by pair_id
            min_quality: Minimum quality score
            
        Returns:
            DataFrame with quality-adjusted weights
        """
        if quality_scores is None:
            # Calculate simple quality scores based on data completeness
            quality_cols = ["sale1_price", "sale2_price", "sale1_date", "sale2_date"]
            
            df = repeat_sales_df.with_columns([
                pl.sum_horizontal([pl.col(c).is_not_null() for c in quality_cols])
                .truediv(len(quality_cols))
                .alias("quality_score")
            ])
        else:
            # Join with provided quality scores
            df = repeat_sales_df.join(
                quality_scores.select(["pair_id", "quality_score"]),
                on="pair_id",
                how="left"
            )
            
        # Apply quality-based weights
        df = df.with_columns([
            pl.col("quality_score").fill_null(min_quality).clip(min_quality, 1.0).alias("weight"),
            pl.lit("quality").alias("weight_type")
        ])
        
        return df
        
    def combine_weights(
        self,
        repeat_sales_df: pl.DataFrame,
        weight_schemes: Dict[WeightingScheme, float],
        **kwargs
    ) -> pl.DataFrame:
        """
        Combine multiple weighting schemes.
        
        Args:
            repeat_sales_df: Repeat sales DataFrame
            weight_schemes: Dict mapping schemes to their relative weights
            
        Returns:
            DataFrame with combined weights
        """
        combined_weights = None
        total_scheme_weight = sum(weight_schemes.values())
        
        for scheme, scheme_weight in weight_schemes.items():
            # Calculate weights for this scheme
            calculator = WeightCalculator(scheme)
            df_weighted = calculator.calculate_weights(repeat_sales_df, **kwargs)
            
            # Extract weights
            weights = df_weighted["weight"] * (scheme_weight / total_scheme_weight)
            
            # Combine
            if combined_weights is None:
                combined_weights = weights
            else:
                combined_weights = combined_weights + weights
                
        # Add combined weights to DataFrame
        df = repeat_sales_df.with_columns([
            combined_weights.alias("weight"),
            pl.lit("combined").alias("weight_type")
        ])
        
        return df
        
    def diagnose_weights(
        self,
        weighted_df: pl.DataFrame
    ) -> Dict[str, Any]:
        """
        Diagnose weight distribution and potential issues.
        
        Args:
            weighted_df: DataFrame with weights
            
        Returns:
            Dictionary with diagnostic statistics
        """
        weights = weighted_df["weight"]
        
        diagnostics = {
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "min": float(weights.min()),
            "max": float(weights.max()),
            "cv": float(weights.std() / weights.mean()) if weights.mean() > 0 else np.inf,
            "percentiles": {
                "p1": float(weights.quantile(0.01)),
                "p5": float(weights.quantile(0.05)),
                "p25": float(weights.quantile(0.25)),
                "p50": float(weights.quantile(0.50)),
                "p75": float(weights.quantile(0.75)),
                "p95": float(weights.quantile(0.95)),
                "p99": float(weights.quantile(0.99))
            },
            "zero_weights": int((weights == 0).sum()),
            "extreme_weights": int((weights > weights.quantile(0.99)).sum()),
            "effective_sample_size": float(weights.sum()**2 / (weights**2).sum())
        }
        
        # Add warnings
        warnings = []
        if diagnostics["cv"] > 2:
            warnings.append("High weight variability (CV > 2)")
        if diagnostics["zero_weights"] > 0:
            warnings.append(f"{diagnostics['zero_weights']} observations have zero weight")
        if diagnostics["max"] / diagnostics["mean"] > 10:
            warnings.append("Extreme weight outliers detected")
            
        diagnostics["warnings"] = warnings
        
        return diagnostics