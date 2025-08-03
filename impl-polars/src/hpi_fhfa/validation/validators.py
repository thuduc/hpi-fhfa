"""Validation utilities for HPI calculation results."""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import structlog

from ..utils.exceptions import ValidationError

logger = structlog.get_logger()


@dataclass
class ValidationResult:
    """Result of HPI validation."""
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any]
    tolerance: Optional[float] = None
    actual_error: Optional[float] = None


class HPIValidator:
    """Validator for HPI calculation results."""
    
    def __init__(self, tolerance: float = 0.001):
        """Initialize validator with numerical tolerance.
        
        Args:
            tolerance: Maximum relative error tolerance (default 0.1%)
        """
        self.tolerance = tolerance
        self.results: List[ValidationResult] = []
        
    def validate_all(
        self,
        tract_indices: pl.DataFrame,
        city_indices: Dict[str, pl.DataFrame],
        reference_tract: Optional[pl.DataFrame] = None,
        reference_city: Optional[Dict[str, pl.DataFrame]] = None
    ) -> List[ValidationResult]:
        """Run all validation tests.
        
        Args:
            tract_indices: Calculated tract-level indices
            city_indices: Calculated city-level indices by weight scheme
            reference_tract: Reference tract indices for comparison
            reference_city: Reference city indices for comparison
            
        Returns:
            List of validation results
        """
        self.results = []
        
        # Validate index properties
        self.results.extend(self._validate_index_properties(tract_indices, "tract"))
        
        for scheme, indices in city_indices.items():
            self.results.extend(self._validate_index_properties(indices, f"city_{scheme}"))
        
        # Validate against reference if provided
        if reference_tract is not None:
            self.results.extend(self._compare_indices(
                tract_indices, reference_tract, "tract_vs_reference"
            ))
            
        if reference_city is not None:
            for scheme in city_indices:
                if scheme in reference_city:
                    self.results.extend(self._compare_indices(
                        city_indices[scheme], reference_city[scheme], 
                        f"city_{scheme}_vs_reference"
                    ))
        
        # Validate cross-consistency
        self.results.extend(self._validate_consistency(tract_indices, city_indices))
        
        return self.results
    
    def _validate_index_properties(
        self, 
        indices: pl.DataFrame, 
        index_type: str
    ) -> List[ValidationResult]:
        """Validate basic properties of indices."""
        results = []
        
        # Test 1: No missing values in key columns
        if index_type.startswith("tract"):
            key_cols = ["tract_id", "year", "index_value"]
        else:
            key_cols = ["cbsa_code", "year", "index_value"]
            
        missing_counts = {col: indices[col].null_count() for col in key_cols}
        total_missing = sum(missing_counts.values())
        
        results.append(ValidationResult(
            test_name=f"{index_type}_no_missing_values",
            passed=total_missing == 0,
            message=f"No missing values in key columns" if total_missing == 0 
                   else f"Found {total_missing} missing values",
            details={"missing_counts": missing_counts}
        ))
        
        # Test 2: Index values are positive
        if "index_value" in indices.columns:
            negative_count = indices.filter(pl.col("index_value") <= 0).height
            results.append(ValidationResult(
                test_name=f"{index_type}_positive_indices",
                passed=negative_count == 0,
                message=f"All indices positive" if negative_count == 0
                       else f"Found {negative_count} non-positive indices",
                details={"negative_count": negative_count}
            ))
        
        # Test 3: Reasonable index range (0.1 to 10x base value)
        if "index_value" in indices.columns and not indices["index_value"].is_empty():
            index_stats = indices["index_value"].describe()
            min_val = indices["index_value"].min()
            max_val = indices["index_value"].max()
            
            reasonable_range = 10 <= min_val <= 1000 and 10 <= max_val <= 1000
            results.append(ValidationResult(
                test_name=f"{index_type}_reasonable_range",
                passed=reasonable_range,
                message=f"Indices in reasonable range" if reasonable_range
                       else f"Indices outside expected range: [{min_val:.2f}, {max_val:.2f}]",
                details={"min_value": min_val, "max_value": max_val, "stats": index_stats}
            ))
        
        # Test 4: Balanced panel (if applicable)
        if index_type.startswith("tract") and "tract_id" in indices.columns:
            tract_year_counts = indices.group_by("tract_id").len()
            unique_year_count = indices["year"].n_unique()
            unbalanced_tracts = tract_year_counts.filter(
                pl.col("len") != unique_year_count
            ).height
            
            results.append(ValidationResult(
                test_name=f"{index_type}_balanced_panel",
                passed=unbalanced_tracts == 0,
                message=f"Balanced panel achieved" if unbalanced_tracts == 0
                       else f"Found {unbalanced_tracts} tracts with incomplete years",
                details={"unbalanced_tracts": unbalanced_tracts, "expected_years": unique_year_count}
            ))
        
        return results
    
    def _compare_indices(
        self,
        calculated: pl.DataFrame,
        reference: pl.DataFrame,
        comparison_name: str
    ) -> List[ValidationResult]:
        """Compare calculated indices with reference implementation."""
        results = []
        
        try:
            # Align datasets for comparison
            if "tract_id" in calculated.columns:
                join_keys = ["tract_id", "year"]
            else:
                join_keys = ["cbsa_code", "year"]
                
            # Join datasets
            comparison = calculated.join(
                reference.select(join_keys + ["index_value"]).rename({"index_value": "ref_index"}),
                on=join_keys,
                how="inner"
            )
            
            if comparison.height == 0:
                results.append(ValidationResult(
                    test_name=f"{comparison_name}_data_alignment",
                    passed=False,
                    message="No matching records between calculated and reference data",
                    details={"calculated_rows": calculated.height, "reference_rows": reference.height}
                ))
                return results
            
            # Calculate relative errors
            comparison = comparison.with_columns([
                ((pl.col("index_value") - pl.col("ref_index")) / pl.col("ref_index")).abs().alias("rel_error")
            ])
            
            # Statistics
            max_error = comparison["rel_error"].max()
            mean_error = comparison["rel_error"].mean()
            median_error = comparison["rel_error"].median()
            error_95th = comparison["rel_error"].quantile(0.95)
            
            # Test: Maximum error within tolerance
            max_error_ok = max_error <= self.tolerance
            results.append(ValidationResult(
                test_name=f"{comparison_name}_max_error",
                passed=max_error_ok,
                message=f"Max relative error: {max_error:.4f}" + 
                       (f" (within {self.tolerance:.3f})" if max_error_ok else f" (exceeds {self.tolerance:.3f})"),
                details={
                    "max_error": max_error,
                    "mean_error": mean_error,
                    "median_error": median_error,
                    "error_95th": error_95th,
                    "n_comparisons": comparison.height
                },
                tolerance=self.tolerance,
                actual_error=max_error
            ))
            
            # Test: Mean error within tolerance/2
            mean_error_ok = mean_error <= self.tolerance / 2
            results.append(ValidationResult(
                test_name=f"{comparison_name}_mean_error",
                passed=mean_error_ok,
                message=f"Mean relative error: {mean_error:.4f}" + 
                       (f" (within {self.tolerance/2:.3f})" if mean_error_ok else f" (exceeds {self.tolerance/2:.3f})"),
                details={"mean_error": mean_error, "n_comparisons": comparison.height},
                tolerance=self.tolerance / 2,
                actual_error=mean_error
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                test_name=f"{comparison_name}_comparison_failed",
                passed=False,
                message=f"Comparison failed: {str(e)}",
                details={"error": str(e)}
            ))
            
        return results
    
    def _validate_consistency(
        self,
        tract_indices: pl.DataFrame,
        city_indices: Dict[str, pl.DataFrame]
    ) -> List[ValidationResult]:
        """Validate consistency between tract and city indices."""
        results = []
        
        # Test 1: City indices should have reasonable relationship to tract indices
        # This is a simplified test - in practice, the relationship is complex
        if not tract_indices.is_empty() and city_indices:
            # Get year range from tract indices
            if "year" in tract_indices.columns:
                tract_years = set(tract_indices["year"].unique().to_list())
                
                for scheme, city_df in city_indices.items():
                    if not city_df.is_empty() and "year" in city_df.columns:
                        city_years = set(city_df["year"].unique().to_list())
                        year_overlap = len(tract_years & city_years)
                        
                        results.append(ValidationResult(
                            test_name=f"consistency_{scheme}_year_coverage",
                            passed=year_overlap > 0,
                            message=f"Year overlap: {year_overlap} years" if year_overlap > 0
                                   else "No year overlap between tract and city indices",
                            details={
                                "tract_years": len(tract_years),
                                "city_years": len(city_years),
                                "overlap": year_overlap
                            }
                        ))
        
        return results
    
    def get_summary_report(self) -> str:
        """Generate a summary report of validation results."""
        if not self.results:
            return "No validation results available."
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        report = [
            "HPI VALIDATION SUMMARY",
            "=" * 50,
            f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)",
            "",
            "DETAILED RESULTS:",
            "-" * 30
        ]
        
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            report.append(f"{status}: {result.test_name}")
            report.append(f"  {result.message}")
            if result.actual_error is not None and result.tolerance is not None:
                report.append(f"  Error: {result.actual_error:.6f} (tolerance: {result.tolerance:.6f})")
            report.append("")
        
        return "\n".join(report)


def compare_indices(
    calculated: pl.DataFrame,
    reference: pl.DataFrame,
    tolerance: float = 0.001
) -> ValidationResult:
    """Compare two index DataFrames."""
    validator = HPIValidator(tolerance=tolerance)
    return validator._compare_indices(calculated, reference, "comparison")[0]


def validate_index_properties(
    indices: pl.DataFrame,
    index_type: str = "index",
    tolerance: float = 0.001
) -> List[ValidationResult]:
    """Validate basic properties of index DataFrame."""
    validator = HPIValidator(tolerance=tolerance)
    return validator._validate_index_properties(indices, index_type)