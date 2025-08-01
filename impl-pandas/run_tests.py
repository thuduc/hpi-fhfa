#!/usr/bin/env python3
"""Test runner script for HPI-FHFA implementation."""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required = ['pandas', 'numpy', 'pytest', 'pandera', 'scipy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nTo install dependencies, run:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def create_test_environment():
    """Create necessary directories for tests."""
    dirs = [
        'tests/fixtures/data',
        'coverage_html_report'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Test directories created")

def run_tests():
    """Run the test suite."""
    print("\n" + "="*60)
    print("Running HPI-FHFA Test Suite")
    print("="*60 + "\n")
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Create test environment
    create_test_environment()
    
    # Run tests with coverage
    cmd = [
        sys.executable, '-m', 'pytest',
        '-v',  # Verbose output
        '--tb=short',  # Short traceback
        '--cov=hpi_fhfa',  # Coverage for our package
        '--cov-report=term-missing',  # Show missing lines
        '--cov-report=html',  # HTML report
        'tests/'
    ]
    
    print("\nRunning tests with command:")
    print(" ".join(cmd))
    print("\n" + "-"*60 + "\n")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("📊 Coverage report saved to: coverage_html_report/index.html")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Some tests failed. Please check the output above.")
        print("="*60)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())