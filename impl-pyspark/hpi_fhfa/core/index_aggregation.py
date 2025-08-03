"""Index aggregation module for HPI-FHFA project"""

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from typing import Dict, List, Optional
import numpy as np
import logging

from ..utils.logging_config import setup_logging


class IndexAggregator:
    """Aggregates supertract-level indices to city-level indices with various weighting schemes"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = setup_logging(self.__class__.__name__)
        self.weight_types = [
            "sample", "value", "unit", "upb", "college", "nonwhite"
        ]
        
    def calculate_weights(
        self,
        supertract_data: DataFrame,
        weight_data: DataFrame,
        weight_type: str,
        year: int
    ) -> DataFrame:
        """
        Calculate normalized weights for aggregation
        
        Args:
            supertract_data: DataFrame with supertract information
            weight_data: DataFrame with weight measures
            weight_type: Type of weight to use
            year: Year for time-varying weights
            
        Returns:
            DataFrame with supertract_id and normalized weight
        """
        self.logger.debug(f"Calculating {weight_type} weights for year {year}")
        
        if weight_type == "sample":
            # Sample weights based on half-pairs
            weights = supertract_data.groupBy("supertract_id").agg(
                F.sum("min_half_pairs").alias("weight_raw")
            )
        
        elif weight_type in ["value", "unit", "upb"]:
            # Time-varying weights - aggregate by supertract
            weights = weight_data.filter(
                F.col("year") == year
            ).join(
                supertract_data.select("supertract_id", F.explode("tract_list").alias("census_tract")),
                on="census_tract",
                how="inner"
            ).groupBy("supertract_id").agg(
                F.sum(F.col(f"{weight_type}_measure")).alias("weight_raw")
            )
            
        else:  # college, nonwhite
            # Static weights from 2010 - aggregate by supertract
            weights = weight_data.filter(
                F.col("year") == 2010
            ).join(
                supertract_data.select("supertract_id", F.explode("tract_list").alias("census_tract")),
                on="census_tract",
                how="inner"
            ).groupBy("supertract_id").agg(
                F.avg(F.col(f"{weight_type}_share")).alias("weight_raw")
            )
        
        # Normalize weights within each CBSA
        window_spec = Window.partitionBy("cbsa_code")
        
        weights = weights.join(
            supertract_data.select("supertract_id", "cbsa_code"),
            on="supertract_id",
            how="inner"
        ).withColumn(
            "total_weight", F.sum("weight_raw").over(window_spec)
        ).withColumn(
            "weight", F.col("weight_raw") / F.col("total_weight")
        ).select("supertract_id", "cbsa_code", "weight")
        
        return weights
    
    def aggregate_city_index(
        self,
        supertract_indices: DataFrame,
        weights: DataFrame,
        cbsa: str,
        year: int
    ) -> Dict[str, float]:
        """
        Aggregate supertract indices to city level
        
        Args:
            supertract_indices: DataFrame with supertract appreciation rates
            weights: DataFrame with normalized weights
            cbsa: CBSA code
            year: Year of aggregation
            
        Returns:
            Dictionary with aggregated appreciation rate and index metadata
        """
        self.logger.debug(f"Aggregating indices for CBSA {cbsa}, year {year}")
        
        # Filter for specific CBSA and join with weights
        weighted_indices = supertract_indices.filter(
            F.col("cbsa_code") == cbsa
        ).join(
            weights,
            on=["supertract_id", "cbsa_code"],
            how="inner"
        ).withColumn(
            "weighted_appreciation",
            F.col("appreciation_rate") * F.col("weight")
        )
        
        # Calculate weighted average appreciation
        result_df = weighted_indices.agg(
            F.sum("weighted_appreciation").alias("city_appreciation"),
            F.count("*").alias("num_supertracts"),
            F.sum("weight").alias("total_weight")
        )
        
        if result_df.count() == 0:
            return {
                "appreciation_rate": 0.0,
                "num_supertracts": 0,
                "coverage": 0.0
            }
        
        result = result_df.collect()[0]
        
        return {
            "appreciation_rate": result["city_appreciation"],
            "num_supertracts": result["num_supertracts"],
            "coverage": result["total_weight"]  # Should be ~1.0
        }
    
    def construct_index_series(
        self,
        appreciations: Dict[int, float],
        cbsa_code: str,
        weight_type: str,
        base_year: int = 1989
    ) -> DataFrame:
        """
        Construct cumulative index series from appreciation rates
        
        Args:
            appreciations: Dictionary of year -> appreciation rate
            cbsa_code: CBSA code
            weight_type: Weight type used
            base_year: Base year for index (default 1989)
            
        Returns:
            DataFrame with complete index series
        """
        self.logger.info(f"Constructing index series for CBSA {cbsa_code}, weight {weight_type}")
        
        index_data = []
        cumulative_index = 100.0  # Base = 100
        prev_index = 100.0
        
        for year in sorted(appreciations.keys()):
            if year > base_year:
                # Apply appreciation
                cumulative_index *= np.exp(appreciations[year])
                yoy_change = (cumulative_index / prev_index - 1) * 100
            else:
                yoy_change = 0.0
                
            index_data.append({
                "cbsa_code": cbsa_code,
                "year": year,
                "weight_type": weight_type,
                "appreciation_rate": float(appreciations[year]),
                "index_value": float(cumulative_index),
                "yoy_change": float(yoy_change)
            })
            
            prev_index = cumulative_index
        
        return self.spark.createDataFrame(index_data)
    
    def process_all_weights(
        self,
        supertract_indices: DataFrame,
        supertract_data: DataFrame,
        weight_data: DataFrame,
        cbsa_code: str,
        year: int
    ) -> DataFrame:
        """
        Process all weight types for a given CBSA and year
        
        Args:
            supertract_indices: DataFrame with supertract appreciation rates
            supertract_data: DataFrame with supertract information
            weight_data: DataFrame with weight measures
            cbsa_code: CBSA code
            year: Year of aggregation
            
        Returns:
            DataFrame with results for all weight types
        """
        results = []
        
        for weight_type in self.weight_types:
            try:
                # Calculate weights
                weights = self.calculate_weights(
                    supertract_data, weight_data, weight_type, year
                )
                
                # Aggregate to city level
                city_result = self.aggregate_city_index(
                    supertract_indices, weights, cbsa_code, year
                )
                
                results.append({
                    "cbsa_code": cbsa_code,
                    "year": year,
                    "weight_type": weight_type,
                    "appreciation_rate": city_result["appreciation_rate"],
                    "num_supertracts": city_result["num_supertracts"],
                    "index_value": 100.0,  # Will be calculated later
                    "yoy_change": 0.0  # Will be calculated later
                })
                
            except Exception as e:
                self.logger.warning(
                    f"Failed to calculate {weight_type} weights for CBSA {cbsa_code}, "
                    f"year {year}: {str(e)}"
                )
        
        if results:
            return self.spark.createDataFrame(results)
        else:
            # Return empty DataFrame with schema
            from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
            schema = StructType([
                StructField("cbsa_code", StringType(), False),
                StructField("year", IntegerType(), False),
                StructField("weight_type", StringType(), False),
                StructField("appreciation_rate", DoubleType(), True),
                StructField("num_supertracts", IntegerType(), True)
            ])
            return self.spark.createDataFrame([], schema)