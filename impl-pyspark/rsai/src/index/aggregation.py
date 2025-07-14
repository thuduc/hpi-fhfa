"""
Index aggregation across geographic levels using PySpark.

This module handles aggregating submarket indices to higher geographic
levels using various weighting schemes.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import date

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from rsai.src.data.models import (
    GeographyLevel,
    WeightingScheme,
    IndexValue
)

logger = logging.getLogger(__name__)


class IndexAggregator:
    """Aggregates indices across geographic hierarchies using PySpark."""
    
    # Geographic hierarchy mapping
    HIERARCHY = {
        GeographyLevel.TRACT: GeographyLevel.COUNTY,
        GeographyLevel.SUPERTRACT: GeographyLevel.COUNTY,
        GeographyLevel.COUNTY: GeographyLevel.CBSA,
        GeographyLevel.CBSA: GeographyLevel.STATE,
        GeographyLevel.STATE: GeographyLevel.NATIONAL
    }
    
    def __init__(self, spark: SparkSession):
        """
        Initialize aggregator.
        
        Args:
            spark: SparkSession instance
        """
        self.spark = spark
        
    def aggregate_indices(
        self,
        index_df: DataFrame,
        from_level: GeographyLevel,
        to_level: GeographyLevel,
        geography_mapping: DataFrame,
        weighting_scheme: WeightingScheme = WeightingScheme.EQUAL,
        weight_data: Optional[DataFrame] = None
    ) -> DataFrame:
        """
        Aggregate indices from one geographic level to another.
        
        Args:
            index_df: DataFrame with index values
            from_level: Source geographic level
            to_level: Target geographic level
            geography_mapping: DataFrame mapping lower to higher geography
            weighting_scheme: Weighting scheme to use
            weight_data: Optional DataFrame with weights
            
        Returns:
            DataFrame with aggregated indices
        """
        from_level_str = from_level.value if hasattr(from_level, 'value') else from_level
        to_level_str = to_level.value if hasattr(to_level, 'value') else to_level
        logger.info(f"Aggregating from {from_level_str} to {to_level_str}")
        
        # Validate aggregation path
        if not self._validate_aggregation(from_level, to_level):
            raise ValueError(f"Cannot aggregate from {from_level} to {to_level}")
            
        # Add geography mapping
        mapped_df = self._add_geography_mapping(
            index_df, geography_mapping, from_level, to_level
        )
        
        # Calculate weights
        weighted_df = self._apply_weights(
            mapped_df, weighting_scheme, weight_data
        )
        
        # Aggregate to target level
        aggregated_df = self._perform_aggregation(
            weighted_df, to_level
        )
        
        return aggregated_df
        
    def create_hierarchical_indices(
        self,
        base_indices: Dict[str, DataFrame],
        geography_mappings: Dict[str, DataFrame],
        target_levels: List[GeographyLevel],
        weighting_scheme: WeightingScheme = WeightingScheme.EQUAL
    ) -> Dict[GeographyLevel, DataFrame]:
        """
        Create indices for multiple geographic levels.
        
        Args:
            base_indices: Dict of base level DataFrames by geography
            geography_mappings: Dict of mapping DataFrames
            target_levels: List of target levels to create
            weighting_scheme: Weighting scheme to use
            
        Returns:
            Dictionary mapping level to index DataFrame
        """
        results = {}
        
        # Start with base indices
        for geo_id, df in base_indices.items():
            level = self._detect_level(geo_id)
            if level not in results:
                results[level] = df
            else:
                results[level] = results[level].union(df)
                
        # Aggregate to each target level
        for target_level in target_levels:
            if target_level in results:
                continue
                
            # Find source level that has both data and mapping
            source_level = self._find_source_level(target_level, results.keys(), geography_mappings)
            
            if source_level is None:
                logger.warning(f"Cannot create indices for {target_level}")
                continue
                
            # Get mapping
            mapping_key = f"{source_level.value if hasattr(source_level, 'value') else source_level}_to_{target_level.value if hasattr(target_level, 'value') else target_level}"
                
            # Aggregate
            aggregated_df = self.aggregate_indices(
                results[source_level],
                source_level,
                target_level,
                geography_mappings[mapping_key],
                weighting_scheme
            )
            
            results[target_level] = aggregated_df
            
        return results
        
    def _validate_aggregation(
        self,
        from_level: GeographyLevel,
        to_level: GeographyLevel
    ) -> bool:
        """Validate if aggregation path is valid."""
        # Check direct hierarchy
        if self.HIERARCHY.get(from_level) == to_level:
            return True
            
        # Check multi-level hierarchy
        current = from_level
        while current in self.HIERARCHY:
            current = self.HIERARCHY[current]
            if current == to_level:
                return True
                
        return False
        
    def _add_geography_mapping(
        self,
        index_df: DataFrame,
        mapping_df: DataFrame,
        from_level: GeographyLevel,
        to_level: GeographyLevel
    ) -> DataFrame:
        """Add higher geography mapping to index data."""
        # Get the actual column names from the mapping DataFrame
        mapping_columns = mapping_df.columns
        
        # Expected column names (with and without _id suffix)
        from_level_str = from_level.value if hasattr(from_level, 'value') else from_level
        to_level_str = to_level.value if hasattr(to_level, 'value') else to_level
        
        # Try different column naming patterns
        from_col = None
        to_col = None
        
        # Try with _id suffix first
        if f"{from_level_str}_id" in mapping_columns:
            from_col = f"{from_level_str}_id"
        elif from_level_str in mapping_columns:
            from_col = from_level_str
        
        if f"{to_level_str}_id" in mapping_columns:
            to_col = f"{to_level_str}_id"
        elif to_level_str in mapping_columns:
            to_col = to_level_str
        
        # Handle generic from_id/to_id columns
        if from_col is None and "from_id" in mapping_columns:
            from_col = "from_id"
        if to_col is None and "to_id" in mapping_columns:
            to_col = "to_id"
            
        # If we still don't have column names, raise an error
        if from_col is None or to_col is None:
            raise ValueError(f"Could not find appropriate columns in mapping DataFrame. Available columns: {mapping_columns}")
            
        # Join with index data
        mapped_df = index_df.join(
            mapping_df.select(
                F.col(from_col).alias("geography_id"),
                F.col(to_col).alias("target_geography_id")
            ),
            on="geography_id",
            how="left"
        )
        
        # Log unmapped geographies
        unmapped_count = mapped_df.filter(
            F.col("target_geography_id").isNull()
        ).count()
        
        if unmapped_count > 0:
            logger.warning(f"{unmapped_count} geographies could not be mapped")
            
        return mapped_df.filter(F.col("target_geography_id").isNotNull())
        
    def _apply_weights(
        self,
        mapped_df: DataFrame,
        weighting_scheme: WeightingScheme,
        weight_data: Optional[DataFrame] = None
    ) -> DataFrame:
        """Apply weights based on scheme."""
        if weighting_scheme == WeightingScheme.EQUAL:
            return mapped_df.withColumn("weight", F.lit(1.0))
            
        elif weighting_scheme == WeightingScheme.VALUE:
            # Weight by median price * number of pairs
            return mapped_df.withColumn(
                "weight",
                F.coalesce(F.col("median_price"), F.lit(1.0)) * 
                F.col("num_pairs")
            )
            
        elif weight_data is not None:
            # Use provided weights
            return mapped_df.join(
                weight_data.select("geography_id", "period", "weight"),
                on=["geography_id", "period"],
                how="left"
            ).fillna({"weight": 1.0})
            
        else:
            # Default to transaction count
            return mapped_df.withColumn(
                "weight",
                F.col("num_pairs")
            )
            
    def _perform_aggregation(
        self,
        weighted_df: DataFrame,
        to_level: GeographyLevel
    ) -> DataFrame:
        """Perform the actual aggregation."""
        # Group by target geography and period
        aggregated_df = weighted_df.groupBy(
            "target_geography_id", "period"
        ).agg(
            # Weighted average of index values (safe division)
            F.when(F.sum("weight") > 0,
                F.sum(F.col("index_value") * F.col("weight")) / F.sum("weight")
            ).otherwise(F.lit(None)).alias("index_value"),
            
            # Sum of pairs and properties
            F.sum("num_pairs").alias("num_pairs"),
            F.sum("num_properties").alias("num_properties"),
            
            # Weighted median price
            F.expr("percentile_approx(median_price, 0.5)").alias("median_price"),
            
            # Count of submarkets
            F.countDistinct("geography_id").alias("num_submarkets"),
            
            # Total weight
            F.sum("weight").alias("total_weight")
        )
        
        # Rename and add geography level (match base indices column order)
        aggregated_df = aggregated_df.select(
            F.col("target_geography_id").alias("geography_id"),
            F.lit(to_level.value if hasattr(to_level, 'value') else to_level).alias("geography_level"),
            "index_value",
            "median_price",
            "num_pairs",
            "num_properties",
            "num_submarkets",
            "period"
        )
        
        return aggregated_df
        
    def chain_indices(
        self,
        index_df: DataFrame,
        reference_period: Optional[date] = None,
        reference_value: float = 100.0
    ) -> DataFrame:
        """
        Chain index values to a reference period.
        
        Args:
            index_df: DataFrame with index values
            reference_period: Reference period (default: first period)
            reference_value: Value at reference period
            
        Returns:
            DataFrame with chained indices
        """
        # Get reference period if not specified
        if reference_period is None:
            reference_period = index_df.agg(
                F.min("period")
            ).collect()[0][0]
            
        # Get reference values for each geography
        reference_df = index_df.filter(
            F.col("period") == reference_period
        ).select(
            "geography_level",
            "geography_id",
            F.col("index_value").alias("reference_index")
        )
        
        # Join and calculate chained values
        chained_df = index_df.join(
            reference_df,
            on=["geography_level", "geography_id"],
            how="left"
        )
        
        # Calculate chained index (safe division)
        chained_df = chained_df.withColumn(
            "chained_index",
            F.when(F.col("reference_index") > 0,
                (F.col("index_value") / F.col("reference_index")) * reference_value
            ).otherwise(F.lit(None))
        ).drop("reference_index")
        
        return chained_df
        
    def calculate_growth_rates(
        self,
        index_df: DataFrame,
        periods: List[int] = [1, 12]
    ) -> DataFrame:
        """
        Calculate growth rates over various periods.
        
        Args:
            index_df: DataFrame with index values
            periods: List of periods to calculate (e.g., [1, 12] for MoM and YoY)
            
        Returns:
            DataFrame with growth rates added
        """
        # Window for lag calculations
        window = Window.partitionBy(
            "geography_level", "geography_id"
        ).orderBy("period")
        
        result_df = index_df
        
        for period in periods:
            col_name = f"growth_{period}m"
            
            result_df = result_df.withColumn(
                col_name,
                F.when(F.lag("index_value", period).over(window) > 0,
                    (F.col("index_value") / 
                     F.lag("index_value", period).over(window) - 1) * 100
                ).otherwise(F.lit(None))
            )
            
        return result_df
        
    def _detect_level(self, geography_id: str) -> GeographyLevel:
        """Detect geography level from ID format."""
        # Simple heuristic based on ID length/format
        if len(geography_id) == 11:  # Census tract
            return GeographyLevel.TRACT
        elif len(geography_id) == 5:  # County FIPS
            return GeographyLevel.COUNTY
        elif len(geography_id) == 2:  # State FIPS
            return GeographyLevel.STATE
        else:
            return GeographyLevel.TRACT  # Default
            
    def _find_source_level(
        self,
        target_level: GeographyLevel,
        available_levels: List[GeographyLevel],
        geography_mappings: Dict[str, DataFrame]
    ) -> Optional[GeographyLevel]:
        """Find the best source level for aggregation."""
        # First try direct hierarchy
        for level, next_level in self.HIERARCHY.items():
            if next_level == target_level and level in available_levels:
                mapping_key = f"{level.value}_to_{target_level.value if hasattr(target_level, 'value') else target_level}"
                if mapping_key in geography_mappings:
                    return level
        
        # If no direct hierarchy, try to find any suitable level with mapping
        # Priority order: COUNTY -> TRACT -> SUPERTRACT -> CBSA -> STATE
        priority_order = [
            GeographyLevel.COUNTY,
            GeographyLevel.TRACT,
            GeographyLevel.SUPERTRACT, 
            GeographyLevel.CBSA,
            GeographyLevel.STATE
        ]
        
        for level in priority_order:
            if level in available_levels and level != target_level:
                mapping_key = f"{level.value}_to_{target_level.value if hasattr(target_level, 'value') else target_level}"
                if mapping_key in geography_mappings:
                    return level
                
        return None
        
    def export_indices(
        self,
        index_dfs: Dict[GeographyLevel, DataFrame],
        output_path: str,
        format: str = "parquet"
    ) -> Dict[str, str]:
        """
        Export indices to files.
        
        Args:
            index_dfs: Dictionary of DataFrames by geography level
            output_path: Base output path
            format: Output format ('parquet' or 'csv')
            
        Returns:
            Dictionary mapping level to file path
        """
        output_paths = {}
        
        for level, df in index_dfs.items():
            file_path = f"{output_path}/indices_{level.value}.{format}"
            
            if format == "parquet":
                df.write.mode("overwrite").parquet(file_path)
            elif format == "csv":
                df.write.mode("overwrite").csv(file_path, header=True)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            output_paths[level.value] = file_path
            logger.info(f"Exported {level.value} indices to {file_path}")
            
        return output_paths