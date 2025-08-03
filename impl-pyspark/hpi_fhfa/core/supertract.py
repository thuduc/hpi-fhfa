"""Supertract algorithm implementation for dynamic census tract aggregation"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, DoubleType
from pyspark.broadcast import Broadcast
from typing import Dict, Set, List, Optional, Tuple
import logging

from ..utils.logging_config import setup_logging
from ..utils.geo_utils import haversine_distance


class SupertractAlgorithm:
    """
    Implements dynamic aggregation of census tracts into supertracts
    to ensure minimum observation thresholds for robust estimation
    """
    
    def __init__(self, spark: SparkSession, min_half_pairs: int = 40):
        self.spark = spark
        self.min_half_pairs = min_half_pairs
        self.logger = setup_logging(self.__class__.__name__)
        
    def create_supertracts(
        self, 
        half_pairs: DataFrame,
        geographic_data: DataFrame,
        year: int
    ) -> DataFrame:
        """
        Dynamic aggregation of census tracts into supertracts
        
        Args:
            half_pairs: DataFrame with half-pair counts by tract/year
            geographic_data: DataFrame with tract geographic information
            year: Year for which to create supertracts
            
        Returns:
            DataFrame with supertract mappings
        """
        self.logger.info(f"Creating supertracts for year {year}")
        
        # Filter for specific year and adjacent year
        year_data = half_pairs.filter(
            F.col("year").isin([year, year - 1])
        ).groupBy("census_tract", "cbsa_code").agg(
            F.min(F.col("total_half_pairs")).alias("min_half_pairs")
        )
        
        # Join with geographic data
        tract_data = year_data.join(
            geographic_data.select("census_tract", "centroid_lat", "centroid_lon"),
            on="census_tract",
            how="left"
        )
        
        # Convert to list for processing
        tract_list = tract_data.collect()
        
        # Create mapping of tracts to supertracts
        supertract_mapping = {}
        supertract_data = {}
        
        for tract_row in tract_list:
            tract_id = tract_row["census_tract"]
            cbsa = tract_row["cbsa_code"]
            
            # Initialize each tract as its own supertract
            supertract_mapping[tract_id] = tract_id
            supertract_data[tract_id] = {
                "tracts": [tract_id],
                "min_half_pairs": tract_row["min_half_pairs"],
                "cbsa": cbsa,
                "lat": tract_row["centroid_lat"],
                "lon": tract_row["centroid_lon"]
            }
        
        # Iteratively merge tracts below threshold
        iteration = 0
        max_iterations = 100
        
        while iteration < max_iterations:
            merged_any = False
            
            # Find tracts below threshold
            below_threshold = [
                tract_id for tract_id, data in supertract_data.items()
                if data["min_half_pairs"] < self.min_half_pairs
            ]
            
            if not below_threshold:
                break
                
            self.logger.info(
                f"Iteration {iteration}: {len(below_threshold)} supertracts below threshold"
            )
            
            # Process each tract below threshold
            for tract_id in below_threshold:
                if tract_id not in supertract_data:  # Already merged
                    continue
                    
                tract_info = supertract_data[tract_id]
                
                # Find nearest neighbor
                nearest_id = self._find_nearest_supertract(
                    tract_id, 
                    tract_info,
                    supertract_data,
                    geographic_data
                )
                
                if nearest_id and nearest_id != tract_id:
                    # Merge tracts
                    self._merge_supertracts(
                        tract_id, 
                        nearest_id, 
                        supertract_data,
                        supertract_mapping
                    )
                    merged_any = True
            
            if not merged_any:
                break
                
            iteration += 1
        
        # Convert back to DataFrame
        result_data = []
        for supertract_id, data in supertract_data.items():
            result_data.append({
                "supertract_id": supertract_id,
                "cbsa_code": data["cbsa"],
                "tract_list": data["tracts"],
                "min_half_pairs": data["min_half_pairs"],
                "num_tracts": len(data["tracts"])
            })
        
        result_df = self.spark.createDataFrame(result_data)
        
        self.logger.info(
            f"Created {result_df.count()} supertracts from "
            f"{len(tract_list)} census tracts"
        )
        
        return result_df
    
    def _find_nearest_supertract(
        self,
        tract_id: str,
        tract_info: Dict,
        supertract_data: Dict,
        geographic_data: DataFrame
    ) -> Optional[str]:
        """Find the nearest supertract to merge with"""
        
        min_distance = float('inf')
        nearest_id = None
        
        # Only consider supertracts in the same CBSA
        for candidate_id, candidate_info in supertract_data.items():
            if (candidate_id != tract_id and 
                candidate_info["cbsa"] == tract_info["cbsa"]):
                
                # Skip if the candidate already has sufficient data
                # unless merging would benefit both
                if candidate_info["min_half_pairs"] >= self.min_half_pairs:
                    # Don't merge with tracts that already have sufficient data
                    continue
                
                # Calculate distance between centroids
                distance = haversine_distance(
                    tract_info["lat"], tract_info["lon"],
                    candidate_info["lat"], candidate_info["lon"]
                )
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_id = candidate_id
        
        return nearest_id
    
    def _merge_supertracts(
        self,
        source_id: str,
        target_id: str,
        supertract_data: Dict,
        supertract_mapping: Dict
    ):
        """Merge source supertract into target supertract"""
        
        source_info = supertract_data[source_id]
        target_info = supertract_data[target_id]
        
        # Combine tract lists
        target_info["tracts"].extend(source_info["tracts"])
        
        # Update minimum half-pairs (sum of both)
        target_info["min_half_pairs"] += source_info["min_half_pairs"]
        
        # Update centroid (weighted average)
        total_tracts = len(target_info["tracts"])
        source_weight = len(source_info["tracts"]) / total_tracts
        target_weight = 1 - source_weight
        
        target_info["lat"] = (
            target_info["lat"] * target_weight + 
            source_info["lat"] * source_weight
        )
        target_info["lon"] = (
            target_info["lon"] * target_weight + 
            source_info["lon"] * source_weight
        )
        
        # Update mapping for all source tracts
        for tract in source_info["tracts"]:
            supertract_mapping[tract] = target_id
        
        # Remove source supertract
        del supertract_data[source_id]
    
    def create_tract_to_supertract_mapping(
        self,
        supertracts: DataFrame
    ) -> DataFrame:
        """
        Create a mapping from individual tracts to supertracts
        
        Args:
            supertracts: DataFrame with supertract definitions
            
        Returns:
            DataFrame with tract to supertract mapping
        """
        # Explode tract_list to create one row per tract
        mapping = supertracts.select(
            F.col("supertract_id"),
            F.col("cbsa_code"),
            F.explode(F.col("tract_list")).alias("census_tract")
        )
        
        return mapping
    
    def calculate_supertract_statistics(
        self,
        supertracts: DataFrame,
        half_pairs: DataFrame,
        year: int
    ) -> DataFrame:
        """
        Calculate statistics for each supertract
        
        Args:
            supertracts: DataFrame with supertract definitions
            half_pairs: DataFrame with half-pair counts
            year: Year for statistics
            
        Returns:
            DataFrame with supertract statistics
        """
        # Create tract mapping
        mapping = self.create_tract_to_supertract_mapping(supertracts)
        
        # Join half-pairs with mapping
        supertract_half_pairs = half_pairs.filter(
            F.col("year") == year
        ).join(
            mapping,
            on="census_tract",
            how="inner"
        ).groupBy("supertract_id").agg(
            F.sum("total_half_pairs").alias("total_half_pairs"),
            F.count("census_tract").alias("tract_count"),
            F.avg("total_half_pairs").alias("avg_half_pairs_per_tract")
        )
        
        # Join with supertract info
        stats = supertracts.join(
            supertract_half_pairs,
            on="supertract_id",
            how="left"
        )
        
        return stats