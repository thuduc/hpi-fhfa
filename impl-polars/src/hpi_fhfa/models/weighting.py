"""Weighting scheme implementations for index aggregation."""

import polars as pl
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import structlog

logger = structlog.get_logger()


class WeightingScheme(ABC):
    """Base class for different weighting schemes."""
    
    @abstractmethod
    def calculate_weights(
        self, 
        supertract_df: pl.DataFrame, 
        period: int,
        geographic_df: Optional[pl.DataFrame] = None,
        transaction_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Calculate weights for supertracts.
        
        Args:
            supertract_df: DataFrame with supertract information
            period: Time period for weight calculation
            geographic_df: Geographic data (for static weights)
            transaction_df: Transaction data (for dynamic weights)
            
        Returns:
            DataFrame with supertract_id and weight columns
        """
        pass
    
    def normalize_weights(self, weights_df: pl.DataFrame) -> pl.DataFrame:
        """Ensure weights sum to 1.0."""
        total_weight = weights_df["weight"].sum()
        if total_weight > 0:
            return weights_df.with_columns(
                (pl.col("weight") / total_weight).alias("weight")
            )
        else:
            # Equal weights if all are zero
            n = len(weights_df)
            return weights_df.with_columns(
                pl.lit(1.0 / n).alias("weight")
            )


class SampleWeights(WeightingScheme):
    """w_sample: Share of half-pairs in the sample."""
    
    def calculate_weights(
        self, 
        supertract_df: pl.DataFrame, 
        period: int,
        geographic_df: Optional[pl.DataFrame] = None,
        transaction_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Calculate weights based on share of half-pairs."""
        logger.debug(f"Calculating sample weights for period {period}")
        
        # Filter to current period
        period_df = supertract_df.filter(pl.col("period") == period)
        
        # Aggregate half-pairs by supertract (in case there are multiple rows per supertract)
        weights = (
            period_df
            .group_by("supertract_id")
            .agg(pl.col("total_half_pairs").first().alias("weight"))
        )
        
        return self.normalize_weights(weights)


class ValueWeights(WeightingScheme):
    """w_value: Share of aggregate housing value (Laspeyres index)."""
    
    def calculate_weights(
        self, 
        supertract_df: pl.DataFrame, 
        period: int,
        geographic_df: Optional[pl.DataFrame] = None,
        transaction_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Calculate weights based on housing value."""
        logger.debug(f"Calculating value weights for period {period}")
        
        if geographic_df is None:
            raise ValueError("Geographic data required for value weights")
        
        # Get supertract mapping for period
        period_df = supertract_df.filter(pl.col("period") == period)
        
        # Aggregate housing values by supertract
        tract_values = geographic_df.select(["tract_id", "housing_value"])
        
        # Join and aggregate
        weights = (
            period_df
            .join(tract_values, on="tract_id", how="left")
            .group_by("supertract_id")
            .agg(pl.col("housing_value").sum().alias("weight"))
        )
        
        return self.normalize_weights(weights)


class UnitWeights(WeightingScheme):
    """w_unit: Share of housing units."""
    
    def calculate_weights(
        self, 
        supertract_df: pl.DataFrame, 
        period: int,
        geographic_df: Optional[pl.DataFrame] = None,
        transaction_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Calculate weights based on housing units."""
        logger.debug(f"Calculating unit weights for period {period}")
        
        if geographic_df is None:
            raise ValueError("Geographic data required for unit weights")
        
        # Get supertract mapping for period
        period_df = supertract_df.filter(pl.col("period") == period)
        
        # Aggregate housing units by supertract
        tract_units = geographic_df.select(["tract_id", "housing_units"])
        
        # Join and aggregate
        weights = (
            period_df
            .join(tract_units, on="tract_id", how="left")
            .group_by("supertract_id")
            .agg(pl.col("housing_units").sum().alias("weight"))
        )
        
        return self.normalize_weights(weights)


class UPBWeights(WeightingScheme):
    """w_upb: Share of unpaid principal balance."""
    
    def calculate_weights(
        self, 
        supertract_df: pl.DataFrame, 
        period: int,
        geographic_df: Optional[pl.DataFrame] = None,
        transaction_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Calculate weights based on unpaid principal balance.
        
        Note: This is a placeholder implementation. In practice, UPB data
        would come from mortgage servicing records.
        """
        logger.warning("UPB weights not fully implemented - using transaction values as proxy")
        
        if transaction_df is None:
            raise ValueError("Transaction data required for UPB weights")
        
        # Use recent transaction values as proxy for UPB
        # Filter to transactions in current period
        period_transactions = transaction_df.filter(
            pl.col("transaction_date").dt.year() == period
        )
        
        # Get supertract mapping
        period_df = supertract_df.filter(pl.col("period") == period)
        
        # Join and aggregate transaction values
        weights = (
            period_transactions
            .join(period_df.select(["tract_id", "supertract_id"]), 
                  left_on="census_tract", right_on="tract_id", how="inner")
            .group_by("supertract_id")
            .agg(pl.col("transaction_price").sum().alias("weight"))
        )
        
        return self.normalize_weights(weights)


class CollegeWeights(WeightingScheme):
    """w_college: Share of college-educated population (static, 2010 Census)."""
    
    def calculate_weights(
        self, 
        supertract_df: pl.DataFrame, 
        period: int,
        geographic_df: Optional[pl.DataFrame] = None,
        transaction_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Calculate weights based on college-educated share."""
        logger.debug(f"Calculating college weights for period {period}")
        
        if geographic_df is None:
            raise ValueError("Geographic data required for college weights")
        
        # Get supertract mapping for period
        period_df = supertract_df.filter(pl.col("period") == period)
        
        # Calculate college-educated population by supertract
        tract_college = geographic_df.with_columns(
            (pl.col("college_share") * pl.col("housing_units")).alias("college_pop")
        ).select(["tract_id", "college_pop"])
        
        # Join and aggregate
        weights = (
            period_df
            .join(tract_college, on="tract_id", how="left")
            .group_by("supertract_id")
            .agg(pl.col("college_pop").sum().alias("weight"))
        )
        
        return self.normalize_weights(weights)


class NonWhiteWeights(WeightingScheme):
    """w_nonwhite: Share of non-white population (static, 2010 Census)."""
    
    def calculate_weights(
        self, 
        supertract_df: pl.DataFrame, 
        period: int,
        geographic_df: Optional[pl.DataFrame] = None,
        transaction_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Calculate weights based on non-white population share."""
        logger.debug(f"Calculating non-white weights for period {period}")
        
        if geographic_df is None:
            raise ValueError("Geographic data required for non-white weights")
        
        # Get supertract mapping for period
        period_df = supertract_df.filter(pl.col("period") == period)
        
        # Calculate non-white population by supertract
        tract_nonwhite = geographic_df.with_columns(
            (pl.col("nonwhite_share") * pl.col("housing_units")).alias("nonwhite_pop")
        ).select(["tract_id", "nonwhite_pop"])
        
        # Join and aggregate
        weights = (
            period_df
            .join(tract_nonwhite, on="tract_id", how="left")
            .group_by("supertract_id")
            .agg(pl.col("nonwhite_pop").sum().alias("weight"))
        )
        
        return self.normalize_weights(weights)


class WeightingFactory:
    """Factory for creating weighting scheme instances."""
    
    WEIGHT_SCHEMES = {
        "sample": SampleWeights,
        "value": ValueWeights,
        "unit": UnitWeights,
        "upb": UPBWeights,
        "college": CollegeWeights,
        "nonwhite": NonWhiteWeights
    }
    
    @classmethod
    def create(cls, scheme_name: str) -> WeightingScheme:
        """Create a weighting scheme instance.
        
        Args:
            scheme_name: Name of weighting scheme
            
        Returns:
            WeightingScheme instance
            
        Raises:
            ValueError: If scheme name is not recognized
        """
        if scheme_name not in cls.WEIGHT_SCHEMES:
            raise ValueError(
                f"Unknown weighting scheme: {scheme_name}. "
                f"Valid options: {list(cls.WEIGHT_SCHEMES.keys())}"
            )
        
        return cls.WEIGHT_SCHEMES[scheme_name]()
    
    @classmethod
    def calculate_all_weights(
        cls,
        scheme_names: List[str],
        supertract_df: pl.DataFrame,
        period: int,
        geographic_df: Optional[pl.DataFrame] = None,
        transaction_df: Optional[pl.DataFrame] = None
    ) -> Dict[str, pl.DataFrame]:
        """Calculate weights for multiple schemes.
        
        Args:
            scheme_names: List of scheme names to calculate
            supertract_df: Supertract data
            period: Time period
            geographic_df: Geographic data
            transaction_df: Transaction data
            
        Returns:
            Dictionary mapping scheme names to weight DataFrames
        """
        weights = {}
        
        for scheme_name in scheme_names:
            try:
                scheme = cls.create(scheme_name)
                weights[scheme_name] = scheme.calculate_weights(
                    supertract_df, period, geographic_df, transaction_df
                )
                logger.info(f"Calculated {scheme_name} weights for period {period}")
            except Exception as e:
                logger.error(f"Failed to calculate {scheme_name} weights: {e}")
                raise
        
        return weights