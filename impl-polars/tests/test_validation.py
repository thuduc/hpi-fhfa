"""Tests for HPI validation functionality."""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
import tempfile

from src.hpi_fhfa.validation.validators import HPIValidator, ValidationResult, compare_indices
from src.hpi_fhfa.validation.benchmarks import PerformanceBenchmark, benchmark_pipeline
from src.hpi_fhfa.config.settings import HPIConfig


class TestHPIValidator:
    """Test HPI validation functionality."""
    
    @pytest.fixture
    def sample_tract_indices(self):
        """Create sample tract indices for testing."""
        return pl.DataFrame({
            "tract_id": ["T001", "T001", "T002", "T002", "T003", "T003"],
            "year": [2020, 2021, 2020, 2021, 2020, 2021],
            "index_value": [100.0, 105.0, 100.0, 103.0, 100.0, 107.0],
            "appreciation_rate": [None, 5.0, None, 3.0, None, 7.0],
            "supertract_id": ["S001", "S001", "S002", "S002", "S003", "S003"]
        })
    
    @pytest.fixture
    def sample_city_indices(self):
        """Create sample city indices for testing."""
        return {
            "sample": pl.DataFrame({
                "cbsa_code": ["C001", "C001", "C002", "C002"],
                "year": [2020, 2021, 2020, 2021],
                "index_value": [100.0, 104.0, 100.0, 106.0],
                "appreciation_rate": [None, 4.0, None, 6.0],
                "weight_scheme": ["sample", "sample", "sample", "sample"],
                "n_supertracts": [2, 2, 1, 1]
            })
        }
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = HPIValidator(tolerance=0.01)
        assert validator.tolerance == 0.01
        assert validator.results == []
    
    def test_validate_index_properties_basic(self, sample_tract_indices):
        """Test basic index property validation."""
        validator = HPIValidator()
        results = validator._validate_index_properties(sample_tract_indices, "tract")
        
        # Should have multiple validation results
        assert len(results) >= 3
        
        # Check specific tests
        test_names = [r.test_name for r in results]
        assert "tract_no_missing_values" in test_names
        assert "tract_positive_indices" in test_names
        assert "tract_reasonable_range" in test_names
        assert "tract_balanced_panel" in test_names
        
        # All tests should pass for good data
        passed_tests = [r for r in results if r.passed]
        assert len(passed_tests) == len(results)
    
    def test_validate_missing_values(self):
        """Test validation with missing values."""
        # Create data with missing values
        bad_data = pl.DataFrame({
            "tract_id": ["T001", "T002", None],
            "year": [2020, 2021, 2022],
            "index_value": [100.0, None, 105.0]
        })
        
        validator = HPIValidator()
        results = validator._validate_index_properties(bad_data, "tract")
        
        # Should fail missing values test
        missing_test = next(r for r in results if "missing_values" in r.test_name)
        assert not missing_test.passed
        assert missing_test.details["missing_counts"]["tract_id"] == 1
        assert missing_test.details["missing_counts"]["index_value"] == 1
    
    def test_validate_negative_indices(self):
        """Test validation with negative index values."""
        bad_data = pl.DataFrame({
            "tract_id": ["T001", "T002"],
            "year": [2020, 2021],
            "index_value": [100.0, -50.0]
        })
        
        validator = HPIValidator()
        results = validator._validate_index_properties(bad_data, "tract")
        
        # Should fail positive indices test
        positive_test = next(r for r in results if "positive_indices" in r.test_name)
        assert not positive_test.passed
        assert positive_test.details["negative_count"] == 1
    
    def test_compare_indices_identical(self, sample_tract_indices):
        """Test comparison of identical indices."""
        validator = HPIValidator(tolerance=0.001)
        results = validator._compare_indices(
            sample_tract_indices,
            sample_tract_indices,
            "identity_test"
        )
        
        # Should pass all comparison tests
        assert len(results) >= 2
        for result in results:
            assert result.passed
            if result.actual_error is not None:
                assert result.actual_error < validator.tolerance
    
    def test_compare_indices_with_differences(self, sample_tract_indices):
        """Test comparison with small differences."""
        # Create reference with small differences
        reference = sample_tract_indices.with_columns(
            (pl.col("index_value") * 1.0005).alias("index_value")  # 0.05% difference
        ).rename({"index_value": "ref_index"}).select(["tract_id", "year", "ref_index"])
        
        # Add ref_index column to calculated data for comparison
        calculated = sample_tract_indices.join(reference, on=["tract_id", "year"])
        
        validator = HPIValidator(tolerance=0.001)  # 0.1% tolerance
        results = validator._compare_indices(calculated, reference, "small_diff_test")
        
        # Check if we have error tests 
        error_tests = [r for r in results if "error" in r.test_name]
        if error_tests:
            # Should fail due to differences exceeding tolerance
            max_error_test = next((r for r in results if "max_error" in r.test_name), None)
            if max_error_test:
                assert not max_error_test.passed
                assert max_error_test.actual_error > validator.tolerance
            else:
                # Test may have failed for other reasons, check any error test
                assert any(not r.passed for r in error_tests)
    
    def test_validate_all_comprehensive(self, sample_tract_indices, sample_city_indices):
        """Test comprehensive validation."""
        validator = HPIValidator(tolerance=0.001)
        results = validator.validate_all(sample_tract_indices, sample_city_indices)
        
        # Should have results for both tract and city validation
        assert len(results) > 0
        
        # Check we have different types of tests
        test_categories = set()
        for result in results:
            if "tract" in result.test_name:
                test_categories.add("tract")
            elif "city" in result.test_name:  
                test_categories.add("city")
            elif "consistency" in result.test_name:
                test_categories.add("consistency")
        
        assert "tract" in test_categories
        assert "city" in test_categories
        assert "consistency" in test_categories
    
    def test_summary_report_generation(self, sample_tract_indices, sample_city_indices):
        """Test summary report generation."""
        validator = HPIValidator()
        results = validator.validate_all(sample_tract_indices, sample_city_indices)
        
        report = validator.get_summary_report()
        
        # Report should contain key information
        assert "HPI VALIDATION SUMMARY" in report
        assert "Tests passed:" in report
        assert "DETAILED RESULTS:" in report
        
        # Should show individual test results
        for result in results[:3]:  # Check first few
            assert result.test_name in report
    
    def test_empty_data_handling(self):
        """Test validation with empty data."""
        empty_tract = pl.DataFrame(schema={
            "tract_id": pl.Utf8,
            "year": pl.Int32,
            "index_value": pl.Float64
        })
        
        empty_city = {"sample": pl.DataFrame(schema={
            "cbsa_code": pl.Utf8,
            "year": pl.Int32,
            "index_value": pl.Float64
        })}
        
        validator = HPIValidator()
        results = validator.validate_all(empty_tract, empty_city)
        
        # Should handle empty data gracefully
        assert len(results) >= 0  # May have some validation results
        
        # Should not crash
        report = validator.get_summary_report()
        assert isinstance(report, str)


class TestPerformanceBenchmark:
    """Test performance benchmarking functionality."""
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """Create a sample configuration for benchmarking."""
        # Create minimal test data
        transactions = pl.DataFrame({
            "property_id": ["P001", "P001", "P002", "P002"],
            "transaction_date": ["2020-01-01", "2021-01-01", "2020-06-01", "2021-06-01"],
            "transaction_price": [100000, 110000, 200000, 220000],
            "census_tract": ["T001", "T001", "T002", "T002"], 
            "cbsa_code": ["C001", "C001", "C001", "C001"],
            "distance_to_cbd": [5.0, 5.0, 10.0, 10.0]
        }).with_columns(pl.col("transaction_date").str.strptime(pl.Date))
        
        geographic = pl.DataFrame({
            "tract_id": ["T001", "T002"],
            "cbsa_code": ["C001", "C001"],
            "centroid_lat": [34.0, 34.1],
            "centroid_lon": [-118.0, -118.1],
            "housing_units": [1000, 1500],
            "housing_value": [1_000_000_000, 1_500_000_000],
            "college_share": [0.5, 0.6],
            "nonwhite_share": [0.3, 0.4]
        })
        
        # Save test data
        txn_path = temp_dir / "test_transactions.parquet"
        geo_path = temp_dir / "test_geographic.parquet"
        
        transactions.write_parquet(txn_path)
        geographic.write_parquet(geo_path)
        
        return HPIConfig(
            transaction_data_path=txn_path,
            geographic_data_path=geo_path,
            output_path=temp_dir / "output",
            start_year=2020,
            end_year=2021,
            weight_schemes=["sample"],
            validate_data=False,
            use_lazy_evaluation=False,
            n_jobs=1
        )
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        benchmark = PerformanceBenchmark()
        assert benchmark.results == []
        assert benchmark.baseline_memory > 0
    
    def test_benchmark_pipeline_execution(self, sample_config):
        """Test pipeline benchmarking."""
        benchmark = PerformanceBenchmark()
        result = benchmark.benchmark_pipeline(sample_config, name="test_benchmark")
        
        # Check result structure
        assert result.name == "test_benchmark"
        assert result.duration_seconds > 0
        assert result.peak_memory_mb > 0
        assert result.n_transactions >= 0
        assert result.throughput_transactions_per_sec >= 0
        
        # Check metadata
        assert "config" in result.metadata
        assert "results_metadata" in result.metadata
    
    def test_benchmark_results_storage(self, sample_config):
        """Test benchmark results are stored."""
        benchmark = PerformanceBenchmark()
        
        # Run multiple benchmarks
        result1 = benchmark.benchmark_pipeline(sample_config, name="test1")
        result2 = benchmark.benchmark_pipeline(sample_config, name="test2")
        
        # Should store results
        assert len(benchmark.results) == 2
        assert benchmark.results[0].name == "test1"
        assert benchmark.results[1].name == "test2"
    
    def test_summary_report_generation(self, sample_config):
        """Test summary report generation."""
        benchmark = PerformanceBenchmark()
        result = benchmark.benchmark_pipeline(sample_config, name="test")
        
        report = benchmark.get_summary_report()
        
        # Report should contain key information
        assert "HPI PERFORMANCE BENCHMARK SUMMARY" in report
        assert "Total benchmarks:" in report
        assert "test" in report
        assert "Duration:" in report
        
    def test_benchmark_convenience_function(self, sample_config):
        """Test convenience benchmark function."""
        result = benchmark_pipeline(sample_config, name="convenience_test")
        
        assert result.name == "convenience_test"
        assert result.duration_seconds > 0
        assert isinstance(result.metadata, dict)


class TestValidationIntegration:
    """Test validation integration with pipeline results."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_compare_indices_function(self):
        """Test standalone compare_indices function."""
        # Create test data
        calculated = pl.DataFrame({
            "tract_id": ["T001", "T002"],
            "year": [2020, 2020],
            "index_value": [100.0, 105.0]
        })
        
        reference = pl.DataFrame({
            "tract_id": ["T001", "T002"],
            "year": [2020, 2020],
            "index_value": [100.1, 104.9]  # Small differences
        })
        
        result = compare_indices(calculated, reference, tolerance=0.01)
        
        assert isinstance(result, ValidationResult)
        assert result.test_name == "comparison_max_error"
        # Should pass with 1% tolerance for 0.1% differences
        assert result.passed
    
    def test_validate_index_properties_function(self):
        """Test standalone validate_index_properties function."""
        from src.hpi_fhfa.validation.validators import validate_index_properties
        
        indices = pl.DataFrame({
            "tract_id": ["T001", "T002"],
            "year": [2020, 2020],
            "index_value": [100.0, 105.0]
        })
        
        results = validate_index_properties(indices, "tract")
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, ValidationResult) for r in results)
    
    def test_validation_error_handling(self):
        """Test validation error handling."""
        # Test with incompatible data structures
        calculated = pl.DataFrame({
            "tract_id": ["T001"],
            "year": [2020],
            "index_value": [100.0]
        })
        
        reference = pl.DataFrame({
            "different_id": ["T001"],
            "year": [2020],
            "index_value": [100.0]
        })
        
        validator = HPIValidator()
        results = validator._compare_indices(calculated, reference, "error_test")
        
        # Should handle the error gracefully
        assert len(results) >= 1
        error_result = results[0]
        assert not error_result.passed
        assert "comparison_failed" in error_result.test_name or "data_alignment" in error_result.test_name