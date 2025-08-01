"""Constants for HPI-FHFA implementation based on PRD specifications."""

# Supertract algorithm parameters
MIN_HALF_PAIRS = 40  # Minimum half-pairs per year for tract/supertract

# Data filtering thresholds
MAX_CAGR = 0.30  # Maximum compound annual growth rate (30%)
MAX_CUMULATIVE_APPRECIATION = 10.0  # Maximum cumulative appreciation (10x)
MIN_CUMULATIVE_APPRECIATION = 0.25  # Minimum cumulative appreciation (0.25x)

# Time period constants
BASE_YEAR = 1989  # Base year for index normalization
START_YEAR = 1975  # Earliest transaction year in dataset
END_YEAR = 2021  # Latest transaction year in dataset
INDEX_START_YEAR = 1989  # First year of index calculation
INDEX_END_YEAR = 2021  # Last year of index calculation

# Geographic constants
CENSUS_TRACT_YEAR = 2010  # Census tract boundary definitions year
TRACT_ID_LENGTH = 11  # Standard Census tract ID length

# Weight types
WEIGHT_TYPES = {
    "sample": "Share of half-pairs",
    "value": "Share of aggregate housing value (Laspeyres)",
    "unit": "Share of housing units", 
    "upb": "Share of unpaid principal balance",
    "college": "Share of college-educated population",
    "nonwhite": "Share of non-white population"
}

# Time-varying vs static weights
TIME_VARYING_WEIGHTS = ["sample", "value", "unit", "upb"]
STATIC_WEIGHTS = ["college", "nonwhite"]

# Regression parameters
BMN_CONVERGENCE_TOLERANCE = 1e-6
BMN_MAX_ITERATIONS = 100

# Data processing
CHUNK_SIZE = 100000  # Default chunk size for large dataset processing
DEFAULT_DISTANCE_UNIT = "miles"  # Default unit for distance calculations

# Index parameters
INDEX_BASE_VALUE = 100.0  # Base value for normalized indices (100 = base year)

# File formats
SUPPORTED_INPUT_FORMATS = [".parquet", ".csv", ".feather"]
DEFAULT_OUTPUT_FORMAT = ".parquet"