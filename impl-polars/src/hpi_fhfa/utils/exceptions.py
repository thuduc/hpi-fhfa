"""Custom exceptions for HPI-FHFA."""


class HPIError(Exception):
    """Base exception for HPI-FHFA package."""
    pass


class ConfigurationError(HPIError):
    """Raised when configuration is invalid."""
    pass


class DataValidationError(HPIError):
    """Raised when data validation fails."""
    pass


class ProcessingError(HPIError):
    """Raised when data processing encounters an error."""
    pass


class InsufficientDataError(HPIError):
    """Raised when there is insufficient data for calculation."""
    pass


class ValidationError(HPIError):
    """Raised when validation fails."""
    pass