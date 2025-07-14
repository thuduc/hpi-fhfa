"""
Geographic distance calculations using PySpark UDFs.

This module provides utilities for calculating distances between
geographic coordinates using the haversine formula.
"""

import logging
import math
from typing import Tuple, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

logger = logging.getLogger(__name__)


class DistanceCalculator:
    """Calculate geographic distances using PySpark."""
    
    EARTH_RADIUS_KM = 6371.0  # Earth's radius in kilometers
    
    def __init__(self, spark: SparkSession):
        """
        Initialize distance calculator.
        
        Args:
            spark: SparkSession instance
        """
        self.spark = spark
        
        # Register UDFs
        self._register_udfs()
        
    def _register_udfs(self):
        """Register User Defined Functions for distance calculations."""
        # Haversine distance UDF
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate haversine distance between two points."""
            if any(v is None for v in [lat1, lon1, lat2, lon2]):
                return None
                
            # Earth's radius in kilometers (local constant to avoid serialization issues)
            earth_radius_km = 6371.0
                
            # Convert to radians
            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)
            
            # Haversine formula
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = (math.sin(dlat / 2) ** 2 + 
                 math.cos(lat1_rad) * math.cos(lat2_rad) * 
                 math.sin(dlon / 2) ** 2)
            c = 2 * math.asin(math.sqrt(a))
            
            return earth_radius_km * c
            
        # Register as Spark UDF
        self.haversine_udf = F.udf(haversine_distance, DoubleType())
        
    def calculate_pairwise_distances(
        self,
        df: DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        id_col: str = "tract"
    ) -> DataFrame:
        """
        Calculate pairwise distances between all points in DataFrame.
        
        Args:
            df: DataFrame with coordinates
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            id_col: Name of ID column
            
        Returns:
            DataFrame with pairwise distances
        """
        # Self-join to create all pairs
        df1 = df.select(
            F.col(id_col).alias("id1"),
            F.col(lat_col).alias("lat1"),
            F.col(lon_col).alias("lon1")
        )
        
        df2 = df.select(
            F.col(id_col).alias("id2"),
            F.col(lat_col).alias("lat2"),
            F.col(lon_col).alias("lon2")
        )
        
        # Cross join and calculate distances
        distances_df = df1.crossJoin(df2).withColumn(
            "distance_km",
            self.haversine_udf(
                F.col("lat1"), F.col("lon1"),
                F.col("lat2"), F.col("lon2")
            )
        )
        
        return distances_df
        
    def add_distance_to_point(
        self,
        df: DataFrame,
        target_lat: float,
        target_lon: float,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        distance_col: str = "distance_to_target"
    ) -> DataFrame:
        """
        Add distance to a specific point for all rows.
        
        Args:
            df: DataFrame with coordinates
            target_lat: Target latitude
            target_lon: Target longitude
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            distance_col: Name for new distance column
            
        Returns:
            DataFrame with distance column added
        """
        return df.withColumn(
            distance_col,
            self.haversine_udf(
                F.col(lat_col),
                F.col(lon_col),
                F.lit(target_lat),
                F.lit(target_lon)
            )
        )
        
    def find_nearest_neighbors(
        self,
        df: DataFrame,
        k: int = 5,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        id_col: str = "tract"
    ) -> DataFrame:
        """
        Find k nearest neighbors for each point.
        
        Args:
            df: DataFrame with coordinates
            k: Number of nearest neighbors
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            id_col: Name of ID column
            
        Returns:
            DataFrame with nearest neighbors
        """
        # Calculate all pairwise distances
        distances_df = self.calculate_pairwise_distances(
            df, lat_col, lon_col, id_col
        )
        
        # Filter out self-distances
        distances_df = distances_df.filter(
            F.col("id1") != F.col("id2")
        )
        
        # Window function to rank by distance
        from pyspark.sql.window import Window
        
        window = Window.partitionBy("id1").orderBy("distance_km")
        
        # Add rank and filter top k
        neighbors_df = distances_df.withColumn(
            "rank", F.row_number().over(window)
        ).filter(
            F.col("rank") <= k
        )
        
        return neighbors_df
        
    def create_distance_matrix(
        self,
        df: DataFrame,
        lat_col: str = "centroid_lat",
        lon_col: str = "centroid_lon",
        id_col: str = "tract"
    ) -> DataFrame:
        """
        Create a distance matrix between all geographic units.
        
        Args:
            df: DataFrame with geographic units
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            id_col: Name of ID column
            
        Returns:
            DataFrame representing distance matrix
        """
        # Filter to rows with valid coordinates
        valid_df = df.filter(
            F.col(lat_col).isNotNull() &
            F.col(lon_col).isNotNull()
        )
        
        if valid_df.count() == 0:
            logger.warning("No valid coordinates found for distance matrix")
            return self.spark.createDataFrame([], schema="id1 string, id2 string, distance double")
            
        # Calculate pairwise distances
        distances_df = self.calculate_pairwise_distances(
            valid_df, lat_col, lon_col, id_col
        )
        
        # Pivot to matrix format if needed
        # For large datasets, keeping in long format is more efficient
        return distances_df
        
    def cluster_by_distance(
        self,
        df: DataFrame,
        max_distance_km: float,
        lat_col: str = "centroid_lat",
        lon_col: str = "centroid_lon",
        id_col: str = "tract"
    ) -> DataFrame:
        """
        Cluster geographic units based on distance threshold.
        
        Args:
            df: DataFrame with geographic units
            max_distance_km: Maximum distance for clustering
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            id_col: Name of ID column
            
        Returns:
            DataFrame with cluster assignments
        """
        # Calculate distances
        distances_df = self.calculate_pairwise_distances(
            df, lat_col, lon_col, id_col
        )
        
        # Create adjacency based on distance threshold
        adjacency_df = distances_df.filter(
            (F.col("distance_km") <= max_distance_km) &
            (F.col("id1") != F.col("id2"))
        ).select("id1", "id2")
        
        # Use GraphFrames if available for connected components
        try:
            from graphframes import GraphFrame
            
            # Create vertices DataFrame
            vertices = df.select(F.col(id_col).alias("id")).distinct()
            
            # Create edges DataFrame
            edges = adjacency_df.select(
                F.col("id1").alias("src"),
                F.col("id2").alias("dst")
            )
            
            # Create graph
            g = GraphFrame(vertices, edges)
            
            # Find connected components
            components = g.connectedComponents()
            
            # Join back with original data
            result_df = df.join(
                components,
                df[id_col] == components["id"],
                how="left"
            ).drop("id")
            
            # Rename component column to cluster
            result_df = result_df.withColumnRenamed("component", "distance_cluster")
            
        except ImportError:
            logger.warning("GraphFrames not available, using simple clustering")
            
            # Fallback: assign each point to its own cluster
            from pyspark.sql.functions import monotonically_increasing_id
            
            result_df = df.withColumn(
                "distance_cluster",
                monotonically_increasing_id()
            )
            
        return result_df
        
    def calculate_centroid(
        self,
        df: DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        weight_col: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Calculate the centroid of a set of coordinates.
        
        Args:
            df: DataFrame with coordinates
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            weight_col: Optional column for weighted centroid
            
        Returns:
            Tuple of (centroid_latitude, centroid_longitude)
        """
        # Filter valid coordinates
        valid_df = df.filter(
            F.col(lat_col).isNotNull() &
            F.col(lon_col).isNotNull()
        )
        
        if valid_df.count() == 0:
            return (None, None)
            
        if weight_col:
            # Weighted centroid
            result = valid_df.agg(
                (F.sum(F.col(lat_col) * F.col(weight_col)) / 
                 F.sum(weight_col)).alias("centroid_lat"),
                (F.sum(F.col(lon_col) * F.col(weight_col)) / 
                 F.sum(weight_col)).alias("centroid_lon")
            ).collect()[0]
        else:
            # Simple average
            result = valid_df.agg(
                F.mean(lat_col).alias("centroid_lat"),
                F.mean(lon_col).alias("centroid_lon")
            ).collect()[0]
            
        return (result["centroid_lat"], result["centroid_lon"])