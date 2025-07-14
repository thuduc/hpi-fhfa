#!/usr/bin/env python
"""
Test runner script for RSAI model tests.

This script provides convenient commands for running different test suites
and generating test data.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: list) -> int:
    """Run a command and return the exit code."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="RSAI test runner")
    parser.add_argument("command", choices=["all", "unit", "integration", "coverage", "data"],
                       help="Test command to run")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("-k", "--keyword", type=str,
                       help="Run tests matching keyword")
    parser.add_argument("-x", "--exitfirst", action="store_true",
                       help="Exit on first failure")
    parser.add_argument("--no-cov", action="store_true",
                       help="Disable coverage reporting")
    
    args = parser.parse_args()
    
    # Base pytest command
    pytest_cmd = ["pytest"]
    
    if args.verbose:
        pytest_cmd.append("-vv")
    
    if args.exitfirst:
        pytest_cmd.append("-x")
    
    if args.keyword:
        pytest_cmd.extend(["-k", args.keyword])
    
    if args.no_cov:
        pytest_cmd.append("--no-cov")
    
    # Execute based on command
    if args.command == "all":
        # Run all tests
        return run_command(pytest_cmd)
    
    elif args.command == "unit":
        # Run only unit tests
        pytest_cmd.extend(["-m", "unit", "tests/unit/"])
        return run_command(pytest_cmd)
    
    elif args.command == "integration":
        # Run only integration tests
        pytest_cmd.extend(["-m", "integration", "tests/integration/"])
        return run_command(pytest_cmd)
    
    elif args.command == "coverage":
        # Run tests with coverage report
        pytest_cmd.extend([
            "--cov-report=html",
            "--cov-report=term-missing:skip-covered",
            "--cov-fail-under=80"
        ])
        exit_code = run_command(pytest_cmd)
        
        if exit_code == 0:
            print("\nCoverage report generated in htmlcov/index.html")
        
        return exit_code
    
    elif args.command == "data":
        # Generate test data
        print("Generating test data...")
        data_dir = Path("tests/fixtures/data")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        return run_command([
            sys.executable,
            "tests/fixtures/generate_test_data.py",
            "--output-dir", str(data_dir),
            "--n-properties", "500",
            "--counties", "06037", "06059"
        ])
    
    return 0


if __name__ == "__main__":
    sys.exit(main())