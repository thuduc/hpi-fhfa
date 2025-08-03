"""Main processing pipeline for HPI calculation."""

import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import structlog
from datetime import datetime
import pickle

from ..config.settings import HPIConfig
from ..data.loader import DataLoader
from ..data.validators import DataValidator
from ..data.filters import TransactionFilter
from ..models.bmn_regression import BMNRegression
from ..models.supertract import SupertractBuilder
from ..models.weighting import WeightingFactory
from ..processing.repeat_sales import RepeatSalesIdentifier
from ..processing.half_pairs import HalfPairsCalculator
from ..utils.exceptions import ProcessingError

logger = structlog.get_logger()


@dataclass
class HPIResults:
    """Container for HPI calculation results."""
    tract_indices: pl.DataFrame
    city_indices: Dict[str, pl.DataFrame]
    metadata: Dict[str, any]


class HPIPipeline:
    """Main processing pipeline for HPI calculation."""
    
    def __init__(self, config: HPIConfig):
        """Initialize pipeline with configuration.
        
        Args:
            config: HPI configuration object
        """
        self.config = config
        self.data_loader = DataLoader(config)
        self.validator = DataValidator(config)
        self.checkpoint_dir = config.output_path / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def run(self, start_year: Optional[int] = None, end_year: Optional[int] = None) -> HPIResults:
        """Execute full pipeline from data loading to index generation.
        
        Args:
            start_year: Override config start year
            end_year: Override config end year
            
        Returns:
            HPIResults with tract and city indices
        """
        start_year = start_year or self.config.start_year
        end_year = end_year or self.config.end_year
        
        logger.info(
            "Starting HPI pipeline",
            start_year=start_year,
            end_year=end_year
        )
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load and validate data
            transactions, geographic_df = self._load_and_validate_data()
            
            # Step 2: Identify repeat sales
            repeat_sales = self._identify_repeat_sales(transactions)
            
            # Step 3: Apply filters
            filtered_sales = self._apply_filters(repeat_sales)
            
            # Step 4: Calculate half-pairs
            half_pairs = self._calculate_half_pairs(filtered_sales, start_year, end_year)
            
            # Step 5: Build supertracts for each period
            all_supertracts = self._build_all_supertracts(
                half_pairs, geographic_df, start_year, end_year
            )
            
            # Step 6: Run BMN regressions
            regression_results = self._run_bmn_regressions(
                all_supertracts, filtered_sales, start_year, end_year
            )
            
            # Step 7: Calculate indices
            tract_indices = self._calculate_tract_indices(
                regression_results, all_supertracts, start_year, end_year
            )
            
            city_indices = self._calculate_city_indices(
                regression_results, all_supertracts, geographic_df,
                filtered_sales, start_year, end_year
            )
            
            # Compile results
            end_time = datetime.now()
            metadata = {
                "start_year": start_year,
                "end_year": end_year,
                "n_transactions": len(transactions),
                "n_repeat_sales": len(repeat_sales),
                "n_filtered_sales": len(filtered_sales),
                "processing_time": (end_time - start_time).total_seconds(),
                "timestamp": end_time.isoformat()
            }
            
            results = HPIResults(
                tract_indices=tract_indices,
                city_indices=city_indices,
                metadata=metadata
            )
            
            # Save results
            self._save_results(results)
            
            logger.info(
                "Pipeline completed successfully",
                processing_time=metadata["processing_time"],
                n_tract_indices=len(tract_indices),
                n_city_indices=len(city_indices)
            )
            
            return results
            
        except Exception as e:
            logger.error("Pipeline failed", error=str(e))
            raise ProcessingError(f"Pipeline failed: {e}")
    
    def _load_and_validate_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load and validate transaction and geographic data."""
        checkpoint = self._load_checkpoint("validated_data")
        if checkpoint is not None:
            logger.info("Loading validated data from checkpoint")
            return checkpoint["transactions"], checkpoint["geographic"]
        
        logger.info("Loading transaction data")
        transactions = self.data_loader.load_transactions()
        
        if self.config.validate_data:
            logger.info("Validating transaction data")
            transactions = self.validator.validate_transactions(transactions)
        
        logger.info("Loading geographic data")
        geographic_df = self.data_loader.load_geographic_data()
        
        if self.config.validate_data:
            logger.info("Validating geographic data")
            geographic_df = self.validator.validate_geographic_data(geographic_df)
        
        self._save_checkpoint("validated_data", {
            "transactions": transactions,
            "geographic": geographic_df
        })
        
        return transactions, geographic_df
    
    def _identify_repeat_sales(self, transactions: pl.DataFrame) -> pl.DataFrame:
        """Identify repeat sales from transactions."""
        checkpoint = self._load_checkpoint("repeat_sales")
        if checkpoint is not None:
            logger.info("Loading repeat sales from checkpoint")
            return checkpoint
        
        identifier = RepeatSalesIdentifier()
        repeat_sales = identifier.identify_repeat_sales(transactions)
        
        stats = identifier.get_statistics()
        logger.info(
            "Repeat sales identified",
            n_repeat_sales=stats["n_repeat_sales"],
            repeat_sales_pct=f"{stats['repeat_sales_pct']:.1f}%"
        )
        
        self._save_checkpoint("repeat_sales", repeat_sales)
        return repeat_sales
    
    def _apply_filters(self, repeat_sales: pl.DataFrame) -> pl.DataFrame:
        """Apply transaction filters."""
        checkpoint = self._load_checkpoint("filtered_sales")
        if checkpoint is not None:
            logger.info("Loading filtered sales from checkpoint")
            return checkpoint
        
        filter = TransactionFilter()
        filtered_sales = filter.apply_filters(repeat_sales)
        
        filter_summary = filter.get_filter_summary()
        logger.info(
            "Filters applied",
            n_filtered=len(filtered_sales),
            filters=filter_summary.to_dicts() if len(filter_summary) > 0 else []
        )
        
        self._save_checkpoint("filtered_sales", filtered_sales)
        return filtered_sales
    
    def _calculate_half_pairs(
        self,
        filtered_sales: pl.DataFrame,
        start_year: int,
        end_year: int
    ) -> pl.DataFrame:
        """Calculate half-pairs by tract and period."""
        checkpoint = self._load_checkpoint("half_pairs")
        if checkpoint is not None:
            logger.info("Loading half-pairs from checkpoint")
            return checkpoint
        
        calculator = HalfPairsCalculator()
        periods = list(range(start_year, end_year + 1))
        half_pairs = calculator.calculate_half_pairs(filtered_sales, periods)
        
        summary = calculator.get_tract_summary(half_pairs)
        logger.info(
            "Half-pairs calculated",
            n_tract_periods=len(half_pairs),
            n_sparse_tracts=len(summary.filter(pl.col("min_half_pairs") < 40))
        )
        
        self._save_checkpoint("half_pairs", half_pairs)
        return half_pairs
    
    def _build_all_supertracts(
        self,
        half_pairs: pl.DataFrame,
        geographic_df: pl.DataFrame,
        start_year: int,
        end_year: int
    ) -> Dict[int, pl.DataFrame]:
        """Build supertracts for all periods."""
        checkpoint = self._load_checkpoint("supertracts")
        if checkpoint is not None:
            logger.info("Loading supertracts from checkpoint")
            return checkpoint
        
        builder = SupertractBuilder(geographic_df)
        all_supertracts = {}
        
        for period in range(start_year, end_year + 1):
            logger.info(f"Building supertracts for period {period}")
            supertracts = builder.build_supertracts(half_pairs, period)
            supertract_mapping = builder.create_supertract_mapping(supertracts)
            all_supertracts[period] = supertract_mapping
            
            # Checkpoint every N periods
            if (self.config.checkpoint_frequency > 0 and 
                period % self.config.checkpoint_frequency == 0):
                self._save_checkpoint("supertracts", all_supertracts)
        
        self._save_checkpoint("supertracts", all_supertracts)
        return all_supertracts
    
    def _run_bmn_regressions(
        self,
        all_supertracts: Dict[int, pl.DataFrame],
        filtered_sales: pl.DataFrame,
        start_year: int,
        end_year: int
    ) -> Dict[str, Dict[int, float]]:
        """Run BMN regressions for each supertract."""
        checkpoint = self._load_checkpoint("regression_results")
        if checkpoint is not None:
            logger.info("Loading regression results from checkpoint")
            return checkpoint
        
        periods = list(range(start_year, end_year + 1))
        
        # Build mapping of supertracts to component tracts
        supertract_components = {}
        for period_mapping in all_supertracts.values():
            for row in period_mapping.iter_rows(named=True):
                supertract_id = row["supertract_id"]
                tract_id = row["tract_id"]
                
                if supertract_id not in supertract_components:
                    supertract_components[supertract_id] = set()
                supertract_components[supertract_id].add(tract_id)
        
        logger.info(f"Running BMN regressions for {len(supertract_components)} supertracts")
        
        # Use parallel processing if enabled
        if self.config.n_jobs != 1:
            from ..utils.performance import OptimizedOperations
            regression_results = OptimizedOperations.optimize_bmn_regression(
                supertract_components,
                filtered_sales,
                periods,
                n_jobs=self.config.n_jobs
            )
        else:
            # Sequential processing
            regression_results = {}
            for supertract_id, component_tracts in supertract_components.items():
                # Filter sales for these tracts
                supertract_sales = filtered_sales.filter(
                    pl.col("census_tract").is_in(list(component_tracts))
                )
                
                if len(supertract_sales) >= len(periods):
                    try:
                        bmn = BMNRegression(periods)
                        bmn.fit(supertract_sales)
                        regression_results[supertract_id] = bmn.get_index_values()
                    except Exception as e:
                        logger.warning(
                            f"BMN regression failed for supertract {supertract_id}",
                            error=str(e)
                        )
        
        logger.info(f"Completed {len(regression_results)} BMN regressions")
        self._save_checkpoint("regression_results", regression_results)
        return regression_results
    
    def _calculate_tract_indices(
        self,
        regression_results: Dict[str, Dict[int, float]],
        all_supertracts: Dict[int, pl.DataFrame],
        start_year: int,
        end_year: int
    ) -> pl.DataFrame:
        """Calculate tract-level indices."""
        from ..indices.tract_level import TractLevelIndex
        tract_index = TractLevelIndex()
        return tract_index.calculate_indices(
            regression_results,
            all_supertracts,
            start_year,
            end_year
        )
    
    def _calculate_city_indices(
        self,
        regression_results: Dict[str, Dict[int, float]],
        all_supertracts: Dict[int, pl.DataFrame],
        geographic_df: pl.DataFrame,
        transaction_df: pl.DataFrame,
        start_year: int,
        end_year: int
    ) -> Dict[str, pl.DataFrame]:
        """Calculate city-level indices for all weight schemes."""
        from ..indices.city_level import CityLevelIndex
        city_index = CityLevelIndex(self.config.weight_schemes)
        return city_index.calculate_all_indices(
            regression_results,
            all_supertracts,
            geographic_df,
            transaction_df,
            start_year,
            end_year
        )
    
    def _save_results(self, results: HPIResults) -> None:
        """Save final results."""
        # Save tract indices
        tract_path = self.data_loader.save_results(
            results.tract_indices,
            "tract_level_indices"
        )
        logger.info(f"Saved tract indices to {tract_path}")
        
        # Save city indices
        for weight_scheme, indices in results.city_indices.items():
            city_path = self.data_loader.save_results(
                indices,
                f"city_level_indices_{weight_scheme}"
            )
            logger.info(f"Saved {weight_scheme} city indices to {city_path}")
        
        # Save metadata
        metadata_path = self.config.output_path / "metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(results.metadata, f, indent=2)
    
    def _save_checkpoint(self, name: str, data: any) -> None:
        """Save checkpoint data."""
        if self.config.checkpoint_frequency <= 0:
            return
            
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        logger.debug(f"Saved checkpoint: {name}")
    
    def _load_checkpoint(self, name: str) -> Optional[any]:
        """Load checkpoint data if exists."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        return None