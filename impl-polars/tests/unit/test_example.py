"""Example unit test file to demonstrate testing structure."""

import pytest
import polars as pl
from datetime import datetime


class TestExample:
    """Example test class."""
    
    def test_polars_dataframe_creation(self):
        """Test that we can create a Polars DataFrame."""
        df = pl.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6]
        })
        
        assert len(df) == 3
        assert df.shape == (3, 2)
        assert df.columns == ["a", "b"]
    
    def test_polars_operations(self):
        """Test basic Polars operations."""
        df = pl.DataFrame({
            "values": [10, 20, 30, 40, 50]
        })
        
        result = df.select([
            pl.col("values").mean().alias("mean"),
            pl.col("values").sum().alias("sum")
        ])
        
        assert result["mean"][0] == 30.0
        assert result["sum"][0] == 150
    
    @pytest.mark.parametrize("input_val,expected", [
        ([1, 2, 3], 6),
        ([10, 20, 30], 60),
        ([5], 5),
    ])
    def test_sum_calculation(self, input_val, expected):
        """Test sum calculation with different inputs."""
        df = pl.DataFrame({"values": input_val})
        result = df.select(pl.col("values").sum())[0, 0]
        assert result == expected
    
    def test_date_handling(self):
        """Test date handling in Polars."""
        df = pl.DataFrame({
            "dates": ["2023-01-01", "2023-06-15", "2023-12-31"]
        })
        
        df = df.with_columns(
            pl.col("dates").str.to_date()
        )
        
        assert df.dtypes[0] == pl.Date
        assert df["dates"][0] == datetime(2023, 1, 1).date()


@pytest.fixture
def sample_transaction_data():
    """Fixture providing sample transaction data."""
    return pl.DataFrame({
        "property_id": list(range(1, 101)),
        "sale_price": [200000 + i * 1000 for i in range(100)],
        "sale_date": ["2023-01-01"] * 50 + ["2023-02-01"] * 50,
        "living_area": [1500 + i * 10 for i in range(100)],
        "bedrooms": [3] * 60 + [4] * 40,
        "bathrooms": [2] * 80 + [3] * 20,
    })


def test_with_fixture(sample_transaction_data):
    """Test using the fixture."""
    assert len(sample_transaction_data) == 100
    assert sample_transaction_data.columns == [
        "property_id", "sale_price", "sale_date", 
        "living_area", "bedrooms", "bathrooms"
    ]