"""Data loading functionality for HPI-FHFA."""

import polars as pl
from pathlib import Path
from typing import Optional, Union, List
import structlog

from .schemas import (
    TRANSACTION_SCHEMA, 
    GEOGRAPHIC_SCHEMA,
    validate_schema,
    cast_to_schema
)
from ..config.settings import HPIConfig
from ..utils.exceptions import DataValidationError

logger = structlog.get_logger()


class DataLoader:
    """Handle data loading operations for HPI pipeline."""
    
    def __init__(self, config: HPIConfig):
        """Initialize data loader with configuration.
        
        Args:
            config: HPI configuration object
        """
        self.config = config
        self.use_lazy = config.use_lazy_evaluation
        
    def load_transactions(
        self, 
        path: Optional[Path] = None,
        columns: Optional[List[str]] = None
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """Load transaction data from file.
        
        Args:
            path: Path to transaction data file (uses config path if not provided)
            columns: Specific columns to load (loads all if not provided)
            
        Returns:
            Transaction data as DataFrame or LazyFrame
            
        Raises:
            DataValidationError: If data loading fails
        """
        data_path = path or self.config.transaction_data_path
        logger.info("Loading transaction data", path=str(data_path))
        
        try:
            # Determine file format and load accordingly
            if data_path.suffix == ".parquet":
                if self.use_lazy:
                    df = pl.scan_parquet(data_path)
                    if columns:
                        df = df.select(columns)
                else:
                    df = pl.read_parquet(data_path, columns=columns)
            elif data_path.suffix == ".csv":
                if self.use_lazy:
                    df = pl.scan_csv(data_path)
                    if columns:
                        df = df.select(columns)
                else:
                    df = pl.read_csv(data_path, columns=columns)
            elif data_path.suffix in [".arrow", ".feather"]:
                if self.use_lazy:
                    df = pl.scan_ipc(data_path)
                    if columns:
                        df = df.select(columns)
                else:
                    df = pl.read_ipc(data_path, columns=columns)
            else:
                raise DataValidationError(
                    f"Unsupported file format: {data_path.suffix}"
                )
            
            # Cast to expected schema types
            if not self.use_lazy:
                df = cast_to_schema(df, TRANSACTION_SCHEMA)
                
            logger.info("Transaction data loaded successfully")
            return df
            
        except Exception as e:
            logger.error("Failed to load transaction data", error=str(e))
            raise DataValidationError(f"Failed to load transaction data: {e}")
    
    def load_geographic_data(
        self, 
        path: Optional[Path] = None
    ) -> pl.DataFrame:
        """Load geographic/census tract data.
        
        Args:
            path: Path to geographic data file
            
        Returns:
            Geographic data as DataFrame
            
        Raises:
            DataValidationError: If data loading fails
        """
        data_path = path or self.config.geographic_data_path
        logger.info("Loading geographic data", path=str(data_path))
        
        try:
            # Geographic data is typically smaller, so we don't use lazy loading
            if data_path.suffix == ".parquet":
                df = pl.read_parquet(data_path)
            elif data_path.suffix == ".csv":
                df = pl.read_csv(data_path)
            elif data_path.suffix in [".arrow", ".feather"]:
                df = pl.read_ipc(data_path)
            else:
                raise DataValidationError(
                    f"Unsupported file format: {data_path.suffix}"
                )
            
            # Cast to expected schema
            df = cast_to_schema(df, GEOGRAPHIC_SCHEMA)
            
            logger.info(
                "Geographic data loaded successfully",
                n_tracts=len(df),
                n_cbsas=df["cbsa_code"].n_unique()
            )
            return df
            
        except Exception as e:
            logger.error("Failed to load geographic data", error=str(e))
            raise DataValidationError(f"Failed to load geographic data: {e}")
    
    def save_results(
        self,
        df: pl.DataFrame,
        name: str,
        format: str = "parquet"
    ) -> Path:
        """Save results to output directory.
        
        Args:
            df: DataFrame to save
            name: Base name for output file
            format: Output format (parquet, csv, arrow)
            
        Returns:
            Path to saved file
        """
        output_file = self.config.output_path / f"{name}.{format}"
        
        logger.info("Saving results", path=str(output_file), format=format)
        
        try:
            if format == "parquet":
                df.write_parquet(output_file)
            elif format == "csv":
                df.write_csv(output_file)
            elif format in ["arrow", "feather"]:
                df.write_ipc(output_file)
            else:
                raise ValueError(f"Unsupported output format: {format}")
                
            logger.info("Results saved successfully", size_mb=output_file.stat().st_size / 1024 / 1024)
            return output_file
            
        except Exception as e:
            logger.error("Failed to save results", error=str(e))
            raise
    
    def load_checkpoint(self, checkpoint_name: str) -> Optional[pl.DataFrame]:
        """Load a checkpoint file if it exists.
        
        Args:
            checkpoint_name: Name of checkpoint file
            
        Returns:
            Checkpoint data if exists, None otherwise
        """
        checkpoint_path = self.config.output_path / "checkpoints" / f"{checkpoint_name}.parquet"
        
        if checkpoint_path.exists():
            logger.info("Loading checkpoint", path=str(checkpoint_path))
            return pl.read_parquet(checkpoint_path)
        
        return None
    
    def save_checkpoint(self, df: pl.DataFrame, checkpoint_name: str) -> None:
        """Save a checkpoint file.
        
        Args:
            df: DataFrame to checkpoint
            checkpoint_name: Name for checkpoint file
        """
        checkpoint_dir = self.config.output_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}.parquet"
        logger.info("Saving checkpoint", path=str(checkpoint_path))
        
        df.write_parquet(checkpoint_path)