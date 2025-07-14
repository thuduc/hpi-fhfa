"""
Supertract generation algorithm using Polars.

This module implements algorithms for aggregating census tracts into supertracts
based on transaction density, geographic proximity, and other criteria.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
from dataclasses import dataclass

import polars as pl
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

from rsai.src.data.models import SupertractDefinition, GeographyLevel
from rsai.src.geography.distance import (
    calculate_distance_matrix_polars,
    calculate_centroid,
    haversine_distance
)

logger = logging.getLogger(__name__)


@dataclass
class TractInfo:
    """Information about a census tract."""
    tract_id: str
    county_fips: str
    num_properties: int
    num_transactions: int
    median_price: Optional[float]
    centroid_lat: Optional[float]
    centroid_lon: Optional[float]
    boundary: Optional[Polygon] = None


class SupertractGenerator:
    """Generates supertracts by aggregating census tracts."""
    
    def __init__(
        self,
        min_transactions: int = 100,
        max_distance_km: float = 10.0,
        min_tracts: int = 1,
        max_tracts: int = 20,
        method: str = "hierarchical"
    ):
        """
        Initialize supertract generator.
        
        Args:
            min_transactions: Minimum transactions per supertract
            max_distance_km: Maximum distance between tract centroids
            min_tracts: Minimum number of tracts per supertract
            max_tracts: Maximum number of tracts per supertract
            method: Clustering method ('hierarchical', 'dbscan', 'greedy')
        """
        self.min_transactions = min_transactions
        self.max_distance_km = max_distance_km
        self.min_tracts = min_tracts
        self.max_tracts = max_tracts
        self.method = method
        self.tract_info: Dict[str, TractInfo] = {}
        self.supertracts: List[SupertractDefinition] = []
        
    def prepare_tract_data(
        self,
        transactions_df: pl.DataFrame,
        properties_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Prepare tract-level statistics from transaction data.
        
        Args:
            transactions_df: Polars DataFrame with transactions
            properties_df: Optional DataFrame with property locations
            
        Returns:
            Polars DataFrame with tract statistics
        """
        logger.info("Preparing tract data for supertract generation")
        
        # Ensure we have tract information
        if "tract" not in transactions_df.columns:
            raise ValueError("Transaction data must include 'tract' column")
            
        # Calculate tract statistics
        tract_stats = transactions_df.group_by(["tract", "county_fips"]).agg([
            pl.count().alias("num_transactions"),
            pl.col("property_id").n_unique().alias("num_properties"),
            pl.col("sale_price").median().alias("median_price")
        ]).filter(
            pl.col("tract").is_not_null() & 
            pl.col("county_fips").is_not_null()
        )
        
        # Add geographic centroids if property data available
        if properties_df is not None and all(col in properties_df.columns for col in ["latitude", "longitude", "tract"]):
            # Calculate tract centroids
            centroids = properties_df.filter(
                pl.col("tract").is_not_null() &
                pl.col("latitude").is_not_null() &
                pl.col("longitude").is_not_null()
            ).group_by("tract").agg([
                pl.col("latitude").mean().alias("centroid_lat"),
                pl.col("longitude").mean().alias("centroid_lon"),
                pl.col("latitude").min().alias("min_lat"),
                pl.col("latitude").max().alias("max_lat"),
                pl.col("longitude").min().alias("min_lon"),
                pl.col("longitude").max().alias("max_lon")
            ])
            
            # Merge with tract stats
            tract_stats = tract_stats.join(
                centroids,
                on="tract",
                how="left"
            )
        else:
            # Add null columns for consistency
            tract_stats = tract_stats.with_columns([
                pl.lit(None).alias("centroid_lat"),
                pl.lit(None).alias("centroid_lon"),
                pl.lit(None).alias("min_lat"),
                pl.lit(None).alias("max_lat"),
                pl.lit(None).alias("min_lon"),
                pl.lit(None).alias("max_lon")
            ])
            
        # Store tract information
        for row in tract_stats.iter_rows(named=True):
            self.tract_info[row["tract"]] = TractInfo(
                tract_id=row["tract"],
                county_fips=row["county_fips"],
                num_properties=row["num_properties"],
                num_transactions=row["num_transactions"],
                median_price=row["median_price"],
                centroid_lat=row.get("centroid_lat"),
                centroid_lon=row.get("centroid_lon")
            )
            
        logger.info(f"Prepared data for {len(self.tract_info)} tracts")
        return tract_stats
        
    def generate_supertracts(
        self,
        tract_stats_df: pl.DataFrame
    ) -> List[SupertractDefinition]:
        """
        Generate supertracts using the specified method.
        
        Args:
            tract_stats_df: Polars DataFrame with tract statistics
            
        Returns:
            List of SupertractDefinition objects
        """
        logger.info(f"Generating supertracts using {self.method} method")
        
        if self.method == "hierarchical":
            return self._hierarchical_clustering(tract_stats_df)
        elif self.method == "dbscan":
            return self._dbscan_clustering(tract_stats_df)
        elif self.method == "greedy":
            return self._greedy_aggregation(tract_stats_df)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
    def _hierarchical_clustering(
        self,
        tract_stats_df: pl.DataFrame
    ) -> List[SupertractDefinition]:
        """
        Use hierarchical clustering to create supertracts.
        
        Args:
            tract_stats_df: Polars DataFrame with tract statistics
            
        Returns:
            List of SupertractDefinition objects
        """
        # Filter to tracts with coordinates
        geo_tracts = tract_stats_df.filter(
            pl.col("centroid_lat").is_not_null() &
            pl.col("centroid_lon").is_not_null()
        )
        
        if len(geo_tracts) == 0:
            logger.warning("No tracts with geographic coordinates")
            return self._fallback_grouping(tract_stats_df)
            
        # Group by county for separate clustering
        supertracts = []
        
        for county in geo_tracts["county_fips"].unique():
            county_tracts = geo_tracts.filter(pl.col("county_fips") == county)
            
            if len(county_tracts) <= self.max_tracts:
                # Small county - create single supertract
                supertract = self._create_supertract_from_tracts(
                    county_tracts["tract"].to_list(),
                    county
                )
                if supertract:
                    supertracts.append(supertract)
                continue
                
            # Calculate distance matrix
            distances = calculate_distance_matrix_polars(
                county_tracts,
                "centroid_lat",
                "centroid_lon",
                "haversine",
                "km"
            )
            
            # Perform hierarchical clustering
            # Calculate appropriate number of clusters to respect max_tracts
            min_clusters = int(np.ceil(len(county_tracts) / self.max_tracts))
            
            clustering = AgglomerativeClustering(
                n_clusters=min_clusters,
                metric="precomputed",
                linkage="average"
            )
            
            labels = clustering.fit_predict(distances)
            
            # Create supertracts from clusters
            county_tracts = county_tracts.with_columns(
                pl.Series("cluster", labels)
            )
            
            for cluster_id in range(labels.max() + 1):
                cluster_tracts = county_tracts.filter(
                    pl.col("cluster") == cluster_id
                )
                
                # Check size constraints
                total_transactions = cluster_tracts["num_transactions"].sum()
                num_tracts = len(cluster_tracts)
                
                if (total_transactions >= self.min_transactions and
                    num_tracts >= self.min_tracts and
                    num_tracts <= self.max_tracts):
                    
                    supertract = self._create_supertract_from_tracts(
                        cluster_tracts["tract"].to_list(),
                        county
                    )
                    if supertract:
                        supertracts.append(supertract)
                else:
                    # Split or merge as needed
                    adjusted = self._adjust_cluster_size(
                        cluster_tracts,
                        county_tracts,
                        distances
                    )
                    supertracts.extend(adjusted)
                    
        self.supertracts = supertracts
        return supertracts
        
    def _dbscan_clustering(
        self,
        tract_stats_df: pl.DataFrame
    ) -> List[SupertractDefinition]:
        """
        Use DBSCAN clustering to create supertracts.
        
        Args:
            tract_stats_df: Polars DataFrame with tract statistics
            
        Returns:
            List of SupertractDefinition objects
        """
        # Filter to tracts with coordinates
        geo_tracts = tract_stats_df.filter(
            pl.col("centroid_lat").is_not_null() &
            pl.col("centroid_lon").is_not_null()
        )
        
        if len(geo_tracts) == 0:
            return self._fallback_grouping(tract_stats_df)
            
        supertracts = []
        
        for county in geo_tracts["county_fips"].unique():
            county_tracts = geo_tracts.filter(pl.col("county_fips") == county)
            
            # Extract coordinates
            coords = county_tracts.select(["centroid_lat", "centroid_lon"]).to_numpy()
            coords_rad = np.radians(coords)
            
            # Calculate min samples based on transaction density
            avg_transactions = county_tracts["num_transactions"].mean()
            min_samples = max(
                self.min_tracts,
                int(self.min_transactions / avg_transactions)
            )
            
            # DBSCAN clustering
            eps_rad = self.max_distance_km / 6371  # Convert km to radians
            clustering = DBSCAN(
                eps=eps_rad,
                min_samples=min_samples,
                metric='haversine'
            )
            
            labels = clustering.fit_predict(coords_rad)
            
            # Create supertracts from clusters
            county_tracts = county_tracts.with_columns(
                pl.Series("cluster", labels)
            )
            
            for cluster_id in set(labels):
                if cluster_id == -1:  # Noise points
                    continue
                    
                cluster_tracts = county_tracts.filter(
                    pl.col("cluster") == cluster_id
                )
                
                supertract = self._create_supertract_from_tracts(
                    cluster_tracts["tract"].to_list(),
                    county
                )
                if supertract:
                    supertracts.append(supertract)
                    
            # Handle noise points
            noise_tracts = county_tracts.filter(pl.col("cluster") == -1)
            if len(noise_tracts) > 0:
                # Try to assign to nearest cluster or create singleton
                for row in noise_tracts.iter_rows(named=True):
                    if row["num_transactions"] >= self.min_transactions:
                        # Create singleton supertract
                        supertract = self._create_supertract_from_tracts(
                            [row["tract"]],
                            county
                        )
                        if supertract:
                            supertracts.append(supertract)
                            
        self.supertracts = supertracts
        return supertracts
        
    def _greedy_aggregation(
        self,
        tract_stats_df: pl.DataFrame
    ) -> List[SupertractDefinition]:
        """
        Use greedy algorithm to aggregate adjacent tracts.
        
        Args:
            tract_stats_df: Polars DataFrame with tract statistics
            
        Returns:
            List of SupertractDefinition objects
        """
        supertracts = []
        
        for county in tract_stats_df["county_fips"].unique():
            county_tracts = tract_stats_df.filter(pl.col("county_fips") == county)
            
            # Sort by number of transactions (ascending)
            sorted_tracts = county_tracts.sort("num_transactions")
            
            # Track assigned tracts
            assigned = set()
            
            # Process tracts with few transactions first
            for row in sorted_tracts.iter_rows(named=True):
                tract_id = row["tract"]
                
                if tract_id in assigned:
                    continue
                    
                # Start new supertract
                current_tracts = [tract_id]
                current_transactions = row["num_transactions"]
                assigned.add(tract_id)
                
                # Find nearby tracts to add
                if row["centroid_lat"] and row["centroid_lon"]:
                    candidates = self._find_nearby_unassigned_tracts(
                        row["centroid_lat"],
                        row["centroid_lon"],
                        county_tracts,
                        assigned
                    )
                    
                    for candidate in candidates:
                        if len(current_tracts) >= self.max_tracts:
                            break
                            
                        if current_transactions >= self.min_transactions:
                            break
                            
                        current_tracts.append(candidate["tract"])
                        current_transactions += candidate["num_transactions"]
                        assigned.add(candidate["tract"])
                        
                # Create supertract if it meets criteria
                if (current_transactions >= self.min_transactions or
                    len(current_tracts) >= self.min_tracts):
                    
                    supertract = self._create_supertract_from_tracts(
                        current_tracts,
                        county
                    )
                    if supertract:
                        supertracts.append(supertract)
                        
        self.supertracts = supertracts
        return supertracts
        
    def _create_supertract_from_tracts(
        self,
        tract_ids: List[str],
        county_fips: str
    ) -> Optional[SupertractDefinition]:
        """
        Create a supertract definition from a list of tract IDs.
        
        Args:
            tract_ids: List of tract IDs
            county_fips: County FIPS code
            
        Returns:
            SupertractDefinition or None if invalid
        """
        if not tract_ids:
            return None
            
        # Aggregate statistics
        num_properties = 0
        num_transactions = 0
        prices = []
        lats = []
        lons = []
        
        for tract_id in tract_ids:
            if tract_id in self.tract_info:
                info = self.tract_info[tract_id]
                num_properties += info.num_properties
                num_transactions += info.num_transactions
                
                if info.median_price:
                    prices.append(info.median_price)
                    
                if info.centroid_lat and info.centroid_lon:
                    lats.append(info.centroid_lat)
                    lons.append(info.centroid_lon)
                    
        # Calculate median price
        median_price = np.median(prices) if prices else None
        
        # Calculate bounds
        min_lat = min(lats) if lats else None
        max_lat = max(lats) if lats else None
        min_lon = min(lons) if lons else None
        max_lon = max(lons) if lons else None
        
        # Generate ID
        supertract_id = f"{county_fips}_ST_{len(self.supertracts) + 1:03d}"
        
        return SupertractDefinition(
            supertract_id=supertract_id,
            name=f"Supertract {len(self.supertracts) + 1}",
            county_fips=county_fips,
            tract_ids=sorted(tract_ids),
            num_properties=num_properties,
            num_transactions=num_transactions,
            median_price=median_price,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon
        )
        
    def _find_nearby_unassigned_tracts(
        self,
        lat: float,
        lon: float,
        county_tracts: pl.DataFrame,
        assigned: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Find nearby unassigned tracts sorted by distance.
        
        Args:
            lat: Reference latitude
            lon: Reference longitude
            county_tracts: DataFrame with county tract data
            assigned: Set of already assigned tract IDs
            
        Returns:
            List of tract info dictionaries sorted by distance
        """
        candidates = []
        
        for row in county_tracts.iter_rows(named=True):
            if row["tract"] in assigned:
                continue
                
            if row["centroid_lat"] and row["centroid_lon"]:
                dist = haversine_distance(
                    lat, lon,
                    row["centroid_lat"], row["centroid_lon"],
                    "km"
                )
                
                if dist <= self.max_distance_km:
                    row["distance"] = dist
                    candidates.append(row)
                    
        # Sort by distance
        return sorted(candidates, key=lambda x: x["distance"])
        
    def _adjust_cluster_size(
        self,
        cluster_tracts: pl.DataFrame,
        all_tracts: pl.DataFrame,
        distance_matrix: np.ndarray
    ) -> List[SupertractDefinition]:
        """
        Adjust cluster size to meet constraints.
        
        Args:
            cluster_tracts: Tracts in current cluster
            all_tracts: All tracts in county
            distance_matrix: Pairwise distance matrix
            
        Returns:
            List of adjusted supertracts
        """
        supertracts = []
        total_transactions = cluster_tracts["num_transactions"].sum()
        num_tracts = len(cluster_tracts)
        
        if total_transactions < self.min_transactions:
            # Cluster too small - try to merge with nearby clusters
            # For now, skip small clusters
            logger.warning(f"Skipping small cluster with {num_tracts} tracts")
            
        elif num_tracts > self.max_tracts:
            # Cluster too large - split into smaller supertracts
            # Use k-means to split
            n_splits = int(np.ceil(num_tracts / self.max_tracts))
            
            # Get indices for this cluster
            cluster_indices = all_tracts.with_row_index(name="temp_row_idx").filter(
                pl.col("tract").is_in(cluster_tracts["tract"])
            )["temp_row_idx"].to_list()
            
            # Extract sub-distance matrix
            sub_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
            
            # Hierarchical clustering with fixed number of clusters
            sub_clustering = AgglomerativeClustering(
                n_clusters=n_splits,
                metric="precomputed",
                linkage="average"
            )
            
            sub_labels = sub_clustering.fit_predict(sub_distances)
            
            # Create supertracts from sub-clusters
            cluster_tracts = cluster_tracts.with_columns(
                pl.Series("subcluster", sub_labels)
            )
            
            for sub_id in range(n_splits):
                sub_tracts = cluster_tracts.filter(
                    pl.col("subcluster") == sub_id
                )
                
                supertract = self._create_supertract_from_tracts(
                    sub_tracts["tract"].to_list(),
                    cluster_tracts["county_fips"][0]
                )
                if supertract:
                    supertracts.append(supertract)
                    
        else:
            # Cluster size is fine
            supertract = self._create_supertract_from_tracts(
                cluster_tracts["tract"].to_list(),
                cluster_tracts["county_fips"][0]
            )
            if supertract:
                supertracts.append(supertract)
                
        return supertracts
        
    def _fallback_grouping(
        self,
        tract_stats_df: pl.DataFrame
    ) -> List[SupertractDefinition]:
        """
        Fallback grouping when geographic data is not available.
        
        Args:
            tract_stats_df: Polars DataFrame with tract statistics
            
        Returns:
            List of SupertractDefinition objects
        """
        logger.warning("Using fallback grouping without geographic data")
        supertracts = []
        
        # Group by county and create supertracts based on tract IDs
        for county in tract_stats_df["county_fips"].unique():
            county_tracts = tract_stats_df.filter(
                pl.col("county_fips") == county
            ).sort("tract")
            
            current_tracts = []
            current_transactions = 0
            
            for row in county_tracts.iter_rows(named=True):
                current_tracts.append(row["tract"])
                current_transactions += row["num_transactions"]
                
                # Check if we should create a supertract
                if (len(current_tracts) >= self.max_tracts or
                    current_transactions >= self.min_transactions * 2):
                    
                    supertract = self._create_supertract_from_tracts(
                        current_tracts,
                        county
                    )
                    if supertract:
                        supertracts.append(supertract)
                        
                    current_tracts = []
                    current_transactions = 0
                    
            # Handle remaining tracts
            if current_tracts:
                supertract = self._create_supertract_from_tracts(
                    current_tracts,
                    county
                )
                if supertract:
                    supertracts.append(supertract)
                    
        self.supertracts = supertracts
        return supertracts
        
    def export_mapping(self) -> pl.DataFrame:
        """
        Export tract to supertract mapping as a Polars DataFrame.
        
        Returns:
            Polars DataFrame with mapping
        """
        mappings = []
        
        for supertract in self.supertracts:
            for tract_id in supertract.tract_ids:
                mappings.append({
                    "tract_id": tract_id,
                    "supertract_id": supertract.supertract_id,
                    "county_fips": supertract.county_fips,
                    "num_tracts_in_supertract": len(supertract.tract_ids)
                })
                
        return pl.DataFrame(mappings)