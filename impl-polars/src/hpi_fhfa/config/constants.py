"""System constants for HPI-FHFA implementation."""

from datetime import date

# Minimum half-pairs required for supertract formation
MIN_HALF_PAIRS = 40

# Filter thresholds
MAX_CAGR_THRESHOLD = 0.30  # 30% compound annual growth rate
MAX_CUMULATIVE_APPRECIATION = 10.0
MIN_CUMULATIVE_APPRECIATION = 0.25

# Base year for index normalization
BASE_YEAR = 1989
BASE_INDEX_VALUE = 100.0

# Date ranges
MIN_TRANSACTION_DATE = date(1975, 1, 1)
MAX_TRANSACTION_DATE = date(2021, 12, 31)
INDEX_START_YEAR = 1989
INDEX_END_YEAR = 2021

# Geographic constants
CENSUS_TRACT_YEAR = 2010

# Performance settings
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_N_JOBS = -1  # Use all available cores