"""Settings configuration for HPI-FHFA implementation."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class Settings:
    """Configuration settings for HPI-FHFA processing."""
    
    # Data paths
    transaction_data_path: Optional[str] = None
    census_data_path: Optional[str] = None
    output_path: Optional[str] = None
    
    # Processing parameters
    min_half_pairs: int = 40
    max_cagr: float = 0.30
    chunk_size: int = 100000
    n_jobs: int = -1  # Number of parallel jobs (-1 = all CPUs)
    
    # Index calculation parameters
    base_year: int = 1989
    start_year: int = 1989
    end_year: int = 2021
    
    # Weight configuration
    default_weight_type: str = "sample"
    
    # Performance settings
    use_sparse_matrices: bool = True
    use_numba: bool = True
    memory_limit_gb: Optional[float] = None
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def from_json(cls, json_path: str) -> "Settings":
        """Load settings from JSON file."""
        with open(json_path, 'r') as f:
            config = json.load(f)
        return cls(**config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Settings":
        """Create settings from dictionary."""
        return cls(**config_dict)
    
    def to_json(self, json_path: str) -> None:
        """Save settings to JSON file."""
        config_dict = {
            k: v for k, v in self.__dict__.items() 
            if v is not None
        }
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate(self) -> None:
        """Validate settings consistency."""
        if self.start_year < 1975:
            raise ValueError("Start year cannot be before 1975")
        
        if self.end_year > 2021:
            raise ValueError("End year cannot be after 2021")
        
        if self.start_year > self.end_year:
            raise ValueError("Start year must be before end year")
        
        if self.base_year < self.start_year or self.base_year > self.end_year:
            raise ValueError("Base year must be between start and end years")
        
        if self.min_half_pairs < 1:
            raise ValueError("Minimum half-pairs must be at least 1")
        
        if self.max_cagr <= 0 or self.max_cagr >= 1:
            raise ValueError("Maximum CAGR must be between 0 and 1")
        
        if self.chunk_size < 1000:
            raise ValueError("Chunk size must be at least 1000")


def get_default_settings() -> Settings:
    """Get default settings instance."""
    return Settings()