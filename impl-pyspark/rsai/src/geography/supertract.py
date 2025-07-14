"""
Supertract generation using PySpark distributed computing.

This module implements dynamic supertract generation through
hierarchical clustering of census tracts.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans, BisectingKMeans
import numpy as np

from rsai.src.data.models import (
    TractInfo,
    SupertractDefinition,
    RSAIConfig
)
from rsai.src.geography.distance import DistanceCalculator

logger = logging.getLogger(__name__)


class SupertractGenerator:
    """
    Generates supertracts by clustering census tracts using PySpark.
    
    Supertracts are clusters of geographically contiguous census tracts
    that meet minimum transaction thresholds.
    """
    
    def __init__(
        self,
        spark: SparkSession,
        min_transactions: int = 40,
        max_distance_km: float = 10.0,
        min_tracts: int = 1,
        max_tracts: int = 20,
        method: str = "kmeans"
    ):
        """
        Initialize supertract generator.
        
        Args:
            spark: SparkSession instance
            min_transactions: Minimum transactions per supertract
            max_distance_km: Maximum distance between tract centroids
            min_tracts: Minimum tracts per supertract
            max_tracts: Maximum tracts per supertract
            method: Clustering method ('kmeans' or 'bisecting')
        """
        self.spark = spark
        self.min_transactions = min_transactions
        self.max_distance_km = max_distance_km
        self.min_tracts = min_tracts
        self.max_tracts = max_tracts
        self.method = method
        
    def prepare_tract_data(
        self,
        repeat_sales_df: DataFrame,
        properties_df: Optional[DataFrame] = None
    ) -> DataFrame:
        """
        Prepare tract-level statistics for clustering.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales
            properties_df: Optional properties DataFrame for centroids
            
        Returns:
            DataFrame with tract statistics
        """
        logger.info("Preparing tract data for supertract generation")
        
        # Calculate tract statistics from repeat sales
        tract_stats = repeat_sales_df.filter(
            F.col("tract").isNotNull()
        ).groupBy("tract", "county").agg(
            F.count("*").alias("num_transactions"),
            F.countDistinct("property_id").alias("num_properties"),
            F.median("sale2_price").alias("median_price"),
            F.mean("log_price_ratio").alias("avg_log_return"),
            F.stddev("log_price_ratio").alias("std_log_return")
        )
        
        # Add centroids if properties data available
        if properties_df is not None:
            tract_centroids = properties_df.filter(
                F.col("tract").isNotNull() &
                F.col("latitude").isNotNull() &
                F.col("longitude").isNotNull()
            ).groupBy("tract").agg(
                F.mean("latitude").alias("centroid_lat"),
                F.mean("longitude").alias("centroid_lon")
            )
            
            tract_stats = tract_stats.join(
                tract_centroids,
                on="tract",
                how="left"
            )
        else:
            # Add null centroids
            tract_stats = tract_stats.withColumn(
                "centroid_lat", F.lit(None)
            ).withColumn(
                "centroid_lon", F.lit(None)
            )
            
        logger.info(f"Prepared data for {tract_stats.count()} tracts")
        return tract_stats
        
    def generate_supertracts(
        self,
        tract_stats_df: DataFrame
    ) -> List[SupertractDefinition]:
        """
        Generate supertracts through clustering.
        
        Args:
            tract_stats_df: DataFrame with tract statistics
            
        Returns:
            List of SupertractDefinition objects
        """
        supertracts = []
        
        # Process by county
        counties = tract_stats_df.select("county").distinct().collect()
        
        for row in counties:
            county = row["county"]
            logger.info(f"Generating supertracts for county {county}")
            
            county_tracts = tract_stats_df.filter(
                F.col("county") == county
            )
            
            if self.method == "kmeans":
                county_supertracts = self._cluster_kmeans(county_tracts, county)
            elif self.method == "bisecting":
                county_supertracts = self._cluster_bisecting(county_tracts, county)
            else:
                raise ValueError(f"Unknown clustering method: {self.method}")
                
            supertracts.extend(county_supertracts)
            
        return supertracts
        
    def _cluster_kmeans(
        self,
        tract_df: DataFrame,
        county: str
    ) -> List[SupertractDefinition]:
        """
        Cluster tracts using K-means.
        
        Args:
            tract_df: DataFrame with tract data for one county
            county: County FIPS code
            
        Returns:
            List of supertracts for the county
        """
        # Check if we have enough data
        num_tracts = tract_df.count()
        if num_tracts < self.min_tracts:
            return []
            
        # Prepare features for clustering
        feature_cols = ["centroid_lat", "centroid_lon", "median_price", 
                       "num_transactions", "avg_log_return"]
        
        # Filter tracts with complete data
        complete_df = tract_df.filter(
            F.col("centroid_lat").isNotNull() &
            F.col("centroid_lon").isNotNull()
        )
        
        if complete_df.count() < self.min_tracts:
            # Fall back to non-spatial clustering
            feature_cols = ["median_price", "num_transactions", "avg_log_return"]
            complete_df = tract_df.filter(
                F.col("median_price").isNotNull()
            )
            
        # Vectorize features
        assembler = VectorAssembler(
            inputCols=[col for col in feature_cols if col in complete_df.columns],
            outputCol="features"
        )
        
        feature_df = assembler.transform(complete_df)
        
        # Scale features
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        
        scaler_model = scaler.fit(feature_df)
        scaled_df = scaler_model.transform(feature_df)
        
        # Determine optimal number of clusters
        total_transactions = tract_df.agg(
            F.sum("num_transactions").alias("total")
        ).collect()[0]["total"]
        
        max_k = min(
            int(num_tracts / self.min_tracts),
            int(total_transactions / self.min_transactions)
        )
        
        if max_k < 1:
            max_k = 1
            
        # Run K-means
        best_k = self._find_optimal_k(scaled_df, max_k)
        
        kmeans = KMeans(
            k=best_k,
            featuresCol="scaled_features",
            predictionCol="cluster",
            maxIter=20,
            seed=42
        )
        
        model = kmeans.fit(scaled_df)
        clustered_df = model.transform(scaled_df)
        
        # Convert clusters to supertracts
        return self._clusters_to_supertracts(clustered_df, county)
        
    def _cluster_bisecting(
        self,
        tract_df: DataFrame,
        county: str
    ) -> List[SupertractDefinition]:
        """
        Cluster tracts using Bisecting K-means.
        
        Args:
            tract_df: DataFrame with tract data for one county
            county: County FIPS code
            
        Returns:
            List of supertracts for the county
        """
        # Similar to kmeans but using BisectingKMeans
        # which can be better for hierarchical clustering
        
        # Check if we have enough data
        num_tracts = tract_df.count()
        if num_tracts < self.min_tracts:
            return []
            
        # Prepare features
        feature_cols = ["centroid_lat", "centroid_lon", "median_price", 
                       "num_transactions"]
        
        complete_df = tract_df.filter(
            F.col("centroid_lat").isNotNull() &
            F.col("centroid_lon").isNotNull()
        )
        
        if complete_df.count() < self.min_tracts:
            feature_cols = ["median_price", "num_transactions"]
            complete_df = tract_df
            
        # Vectorize and scale
        assembler = VectorAssembler(
            inputCols=[col for col in feature_cols if col in complete_df.columns],
            outputCol="features"
        )
        
        feature_df = assembler.transform(complete_df)
        
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        
        scaler_model = scaler.fit(feature_df)
        scaled_df = scaler_model.transform(feature_df)
        
        # Determine number of clusters
        total_transactions = tract_df.agg(
            F.sum("num_transactions").alias("total")
        ).collect()[0]["total"]
        
        k = min(
            int(num_tracts / self.min_tracts),
            int(total_transactions / self.min_transactions)
        )
        
        if k < 1:
            k = 1
            
        # Run Bisecting K-means
        bkm = BisectingKMeans(
            k=k,
            featuresCol="scaled_features",
            predictionCol="cluster",
            maxIter=20,
            seed=42
        )
        
        model = bkm.fit(scaled_df)
        clustered_df = model.transform(scaled_df)
        
        return self._clusters_to_supertracts(clustered_df, county)
        
    def _find_optimal_k(
        self,
        scaled_df: DataFrame,
        max_k: int
    ) -> int:
        """
        Find optimal number of clusters using elbow method.
        
        Args:
            scaled_df: DataFrame with scaled features
            max_k: Maximum number of clusters to test
            
        Returns:
            Optimal k value
        """
        if max_k <= 1:
            return 1
            
        # For large datasets, sample to speed up
        sample_fraction = min(1.0, 1000.0 / scaled_df.count())
        sample_df = scaled_df.sample(fraction=sample_fraction, seed=42)
        
        costs = []
        k_values = range(1, min(max_k + 1, 10))
        
        for k in k_values:
            kmeans = KMeans(
                k=k,
                featuresCol="scaled_features",
                maxIter=10,
                seed=42
            )
            model = kmeans.fit(sample_df)
            costs.append(model.summary.trainingCost)
            
        # Simple elbow detection
        if len(costs) > 2:
            # Calculate rate of change
            deltas = [costs[i] - costs[i+1] for i in range(len(costs)-1)]
            # Find where rate of change decreases most
            if deltas:
                elbow = deltas.index(max(deltas)) + 1
                return k_values[elbow]
                
        return min(3, max_k)  # Default
        
    def _clusters_to_supertracts(
        self,
        clustered_df: DataFrame,
        county: str
    ) -> List[SupertractDefinition]:
        """
        Convert cluster assignments to supertract definitions.
        
        Args:
            clustered_df: DataFrame with cluster assignments
            county: County FIPS code
            
        Returns:
            List of SupertractDefinition objects
        """
        supertracts = []
        
        # Aggregate by cluster
        cluster_stats = clustered_df.groupBy("cluster").agg(
            F.collect_list("tract").alias("tract_ids"),
            F.sum("num_transactions").alias("total_transactions"),
            F.sum("num_properties").alias("total_properties"),
            F.mean("median_price").alias("avg_median_price"),
            F.min("centroid_lat").alias("min_lat"),
            F.max("centroid_lat").alias("max_lat"),
            F.min("centroid_lon").alias("min_lon"),
            F.max("centroid_lon").alias("max_lon")
        ).collect()
        
        for i, row in enumerate(cluster_stats):
            # Check if cluster meets thresholds
            if (row["total_transactions"] >= self.min_transactions and
                len(row["tract_ids"]) >= self.min_tracts and
                len(row["tract_ids"]) <= self.max_tracts):
                
                supertract = SupertractDefinition(
                    supertract_id=f"{county}_ST_{i+1:03d}",
                    name=f"Supertract {i+1}",
                    county=county,
                    tract_ids=row["tract_ids"],
                    num_properties=row["total_properties"],
                    num_transactions=row["total_transactions"],
                    median_price=row["avg_median_price"],
                    min_lat=row["min_lat"] or 0.0,
                    max_lat=row["max_lat"] or 0.0,
                    min_lon=row["min_lon"] or 0.0,
                    max_lon=row["max_lon"] or 0.0
                )
                
                supertracts.append(supertract)
            else:
                # Try to merge small clusters or split large ones
                if row["total_transactions"] < self.min_transactions:
                    logger.warning(
                        f"Cluster {i} has only {row['total_transactions']} transactions"
                    )
                elif len(row["tract_ids"]) > self.max_tracts:
                    logger.warning(
                        f"Cluster {i} has {len(row['tract_ids'])} tracts (max: {self.max_tracts})"
                    )
                    
        return supertracts
        
    def validate_supertracts(
        self,
        supertracts: List[SupertractDefinition],
        tract_stats_df: DataFrame
    ) -> Tuple[List[SupertractDefinition], Dict[str, Any]]:
        """
        Validate generated supertracts.
        
        Args:
            supertracts: List of generated supertracts
            tract_stats_df: Original tract statistics
            
        Returns:
            Tuple of (valid supertracts, validation metrics)
        """
        valid_supertracts = []
        
        # Collect all tract IDs in supertracts
        all_supertract_tracts = set()
        for st in supertracts:
            all_supertract_tracts.update(st.tract_ids)
            
        # Find unclustered tracts
        all_tracts = set(
            row["tract"] for row in 
            tract_stats_df.select("tract").distinct().collect()
        )
        unclustered_tracts = all_tracts - all_supertract_tracts
        
        # Validate each supertract
        for st in supertracts:
            if (st.num_transactions >= self.min_transactions and
                len(st.tract_ids) >= self.min_tracts and
                len(st.tract_ids) <= self.max_tracts):
                valid_supertracts.append(st)
                
        metrics = {
            "total_tracts": len(all_tracts),
            "clustered_tracts": len(all_supertract_tracts),
            "unclustered_tracts": len(unclustered_tracts),
            "total_supertracts": len(supertracts),
            "valid_supertracts": len(valid_supertracts),
            "coverage_rate": len(all_supertract_tracts) / len(all_tracts)
        }
        
        logger.info(
            f"Validated {len(valid_supertracts)}/{len(supertracts)} supertracts"
        )
        
        return valid_supertracts, metrics