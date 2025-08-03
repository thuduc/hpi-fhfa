"""Configuration management for HPI-FHFA."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..utils.exceptions import ConfigurationError


@dataclass
class HPIConfig:
    """Main configuration class for HPI pipeline."""
    
    # Data paths
    transaction_data_path: Path
    geographic_data_path: Path
    output_path: Path
    
    # Processing settings
    start_year: int = 1989
    end_year: int = 2021
    chunk_size: int = 100_000
    n_jobs: int = -1
    
    # Weight schemes to calculate
    weight_schemes: List[str] = field(default_factory=lambda: [
        "sample", "value", "unit", "upb", "college", "nonwhite"
    ])
    
    # Validation settings
    validate_data: bool = True
    strict_validation: bool = False
    
    # Performance settings
    use_lazy_evaluation: bool = True
    checkpoint_frequency: int = 10  # Checkpoint every N periods
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_paths()
        self._validate_years()
        self._validate_weight_schemes()
    
    def _validate_paths(self) -> None:
        """Validate that required paths exist."""
        if not self.transaction_data_path.exists():
            raise ConfigurationError(
                f"Transaction data path does not exist: {self.transaction_data_path}"
            )
        if not self.geographic_data_path.exists():
            raise ConfigurationError(
                f"Geographic data path does not exist: {self.geographic_data_path}"
            )
        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def _validate_years(self) -> None:
        """Validate year range."""
        if self.start_year < 1975:
            raise ConfigurationError("Start year cannot be before 1975")
        if self.end_year > 2021:
            raise ConfigurationError("End year cannot be after 2021")
        if self.start_year >= self.end_year:
            raise ConfigurationError("Start year must be before end year")
    
    def _validate_weight_schemes(self) -> None:
        """Validate weight scheme names."""
        valid_schemes = {"sample", "value", "unit", "upb", "college", "nonwhite"}
        invalid_schemes = set(self.weight_schemes) - valid_schemes
        if invalid_schemes:
            raise ConfigurationError(
                f"Invalid weight schemes: {invalid_schemes}. "
                f"Valid options are: {valid_schemes}"
            )


def load_config(config_path: Optional[Path] = None) -> HPIConfig:
    """Load configuration from file or environment."""
    # This is a placeholder for more complex config loading
    # In production, this would read from YAML/JSON or environment variables
    raise NotImplementedError("Config loading from file not yet implemented")