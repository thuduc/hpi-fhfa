"""
HPI-FHFA: Federal Housing Finance Agency Repeat-Sales Aggregation Index Implementation

This package implements the FHFA RSAI method for constructing house price indices
based on Contat & Larson (2022).
"""

__version__ = "0.1.0"
__author__ = "HPI-FHFA Implementation Team"

from .config import constants
from .data import schemas
from .models import bmn_regression, price_relatives, repeat_sales

__all__ = [
    "constants",
    "schemas", 
    "bmn_regression",
    "price_relatives",
    "repeat_sales"
]