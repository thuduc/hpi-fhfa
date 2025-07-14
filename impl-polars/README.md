# RSAI Model Implementation with Polars

This is an implementation of the Refined Stratified-Adjusted Index (RSAI) model using Polars for high-performance data processing.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Project Structure

```
rsai/
├── calibration/      # Model calibration modules
├── data/            # Data loading and handling
├── evaluation/      # Model evaluation metrics
├── models/          # Core RSAI model implementations
├── preprocessing/   # Data preprocessing utilities
└── utils/          # General utilities

tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
└── fixtures/       # Test data and fixtures
```

## Usage

```python
import polars as pl
from rsai.models import RSAIModel
from rsai.data import DataLoader

# Load data
loader = DataLoader()
data = loader.load_transactions("path/to/data.parquet")

# Train model
model = RSAIModel()
model.fit(data)

# Make predictions
predictions = model.predict(new_data)
```

## Testing

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit

# Run with coverage
pytest --cov=rsai
```

## Development

```bash
# Format code
black rsai tests

# Sort imports
isort rsai tests

# Type checking
mypy rsai

# Linting
flake8 rsai tests
```