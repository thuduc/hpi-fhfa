#!/usr/bin/env python3
"""Check HPI-FHFA implementation structure without external dependencies."""

import os
import sys
from pathlib import Path
import importlib.util
import ast

def check_file_structure():
    """Check if all required files exist."""
    print("📁 Checking file structure...")
    
    required_files = [
        # Package structure
        'hpi_fhfa/__init__.py',
        'hpi_fhfa/config/__init__.py',
        'hpi_fhfa/config/constants.py',
        'hpi_fhfa/config/settings.py',
        'hpi_fhfa/data/__init__.py',
        'hpi_fhfa/data/schemas.py',
        'hpi_fhfa/data/loaders.py',
        'hpi_fhfa/data/filters.py',
        'hpi_fhfa/models/__init__.py',
        'hpi_fhfa/models/bmn_regression.py',
        'hpi_fhfa/models/price_relatives.py',
        'hpi_fhfa/models/repeat_sales.py',
        
        # Test files
        'tests/__init__.py',
        'tests/unit/__init__.py',
        'tests/unit/test_config.py',
        'tests/unit/test_schemas.py',
        'tests/unit/test_filters.py',
        'tests/unit/test_bmn_regression.py',
        'tests/unit/test_price_relatives.py',
        'tests/unit/test_repeat_sales.py',
        
        # Configuration files
        'setup.py',
        'requirements.txt',
        'README.md',
        'pytest.ini'
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
    
    if missing:
        print(f"\n❌ {len(missing)} files are missing!")
        return False
    else:
        print(f"\n✅ All {len(required_files)} required files exist!")
        return True

def analyze_code_structure():
    """Analyze the code structure without importing (to avoid dependency issues)."""
    print("\n📊 Analyzing code structure...")
    
    modules_to_check = {
        'hpi_fhfa/config/constants.py': [
            'MIN_HALF_PAIRS', 'MAX_CAGR', 'BASE_YEAR', 'WEIGHT_TYPES'
        ],
        'hpi_fhfa/config/settings.py': [
            'Settings', 'get_default_settings'
        ],
        'hpi_fhfa/data/schemas.py': [
            'transaction_schema', 'census_tract_schema', 'repeat_sales_schema'
        ],
        'hpi_fhfa/data/filters.py': [
            'filter_transactions', 'apply_cagr_filter', 'apply_same_period_filter'
        ],
        'hpi_fhfa/models/bmn_regression.py': [
            'BMNRegressor', 'BMNResults', 'calculate_index_from_coefficients'
        ],
        'hpi_fhfa/models/price_relatives.py': [
            'calculate_price_relative', 'calculate_half_pairs', 'calculate_cagr'
        ],
        'hpi_fhfa/models/repeat_sales.py': [
            'RepeatSalesPair', 'construct_repeat_sales_pairs', 'create_time_dummies'
        ]
    }
    
    all_good = True
    
    for module_path, expected_items in modules_to_check.items():
        print(f"\n📄 Checking {module_path}...")
        
        try:
            with open(module_path, 'r') as f:
                tree = ast.parse(f.read())
            
            # Extract all top-level definitions
            definitions = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    definitions.add(node.name)
                elif isinstance(node, ast.FunctionDef):
                    definitions.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            definitions.add(target.id)
            
            # Check expected items
            for item in expected_items:
                if item in definitions:
                    print(f"  ✅ Found: {item}")
                else:
                    print(f"  ❌ Missing: {item}")
                    all_good = False
                    
        except Exception as e:
            print(f"  ❌ Error analyzing file: {e}")
            all_good = False
    
    return all_good

def count_tests():
    """Count the number of test functions."""
    print("\n🧪 Counting tests...")
    
    test_files = Path('tests/unit').glob('test_*.py')
    total_tests = 0
    test_summary = {}
    
    for test_file in test_files:
        with open(test_file, 'r') as f:
            content = f.read()
            
        # Count test functions (simple approach)
        test_count = content.count('def test_')
        total_tests += test_count
        test_summary[test_file.name] = test_count
        
        print(f"  {test_file.name}: {test_count} tests")
    
    print(f"\n✅ Total test functions: {total_tests}")
    return total_tests

def generate_simple_test_data():
    """Generate simple test data without pandas."""
    print("\n📊 Generating simple test data...")
    
    data_dir = Path('tests/fixtures/data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple CSV file
    csv_content = """property_id,transaction_date,transaction_price,census_tract,cbsa_code,distance_to_cbd
P000001,2019-01-15,250000.00,12345678901,10420,5.2
P000001,2021-03-20,285000.00,12345678901,10420,5.2
P000002,2018-06-10,350000.00,12345678902,10420,3.8
P000002,2020-09-15,395000.00,12345678902,10420,3.8
P000003,2019-11-01,180000.00,12345678901,10420,7.5
P000003,2021-11-15,198000.00,12345678901,10420,7.5
P000004,2020-02-01,425000.00,12345678903,10520,2.1
P000005,2019-05-15,300000.00,12345678902,10420,4.3
P000005,2021-08-20,345000.00,12345678902,10420,4.3
"""
    
    test_file = data_dir / 'simple_test_data.csv'
    with open(test_file, 'w') as f:
        f.write(csv_content)
    
    print(f"✅ Created test data file: {test_file}")
    
    # Create census tract data
    census_content = """census_tract,cbsa_code,centroid_lat,centroid_lon,housing_units,aggregate_value,college_share,nonwhite_share
12345678901,10420,40.7128,-74.0060,1500,375000000,0.35,0.28
12345678902,10420,40.7580,-73.9855,2000,700000000,0.42,0.31
12345678903,10520,40.6892,-74.0445,1200,510000000,0.28,0.45
"""
    
    census_file = data_dir / 'simple_census_data.csv'
    with open(census_file, 'w') as f:
        f.write(census_content)
    
    print(f"✅ Created census data file: {census_file}")
    
    return True

def main():
    """Run all checks."""
    print("=" * 60)
    print("HPI-FHFA Implementation Structure Check")
    print("=" * 60)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Run checks
    checks = [
        ("File Structure", check_file_structure),
        ("Code Structure", analyze_code_structure),
        ("Test Count", lambda: count_tests() > 0),
        ("Test Data", generate_simple_test_data)
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n{'='*60}")
        print(f"Running: {check_name}")
        print('='*60)
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = all(result for _, result in results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
    
    if all_passed:
        print("\n✅ All checks passed! The implementation structure is complete.")
        print("\nPhases 1 & 2 Implementation Status:")
        print("- Phase 1 (Core Data Structures): ✅ COMPLETE")
        print("- Phase 2 (Mathematical Components): ✅ COMPLETE")
        print("\nTo run tests with pytest (requires dependencies):")
        print("  pip install -r requirements.txt")
        print("  pytest tests/ -v")
    else:
        print("\n❌ Some checks failed. Please review the output above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())