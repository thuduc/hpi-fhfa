#!/usr/bin/env python3
"""
Script to run PySpark tests with proper configuration.
"""

import os
import sys
import subprocess
import platform

def find_java_home():
    """Find suitable Java installation."""
    if platform.system() == "Darwin":  # macOS
        # Try to find Java using java_home utility
        for version in ["17", "19", "11", "1.8", "8"]:
            try:
                result = subprocess.run(
                    ["/usr/libexec/java_home", "-v", version],
                    capture_output=True,
                    text=True,
                    check=True
                )
                java_home = result.stdout.strip()
                if java_home and os.path.exists(java_home):
                    return java_home
            except subprocess.CalledProcessError:
                continue
    
    # Check common Java locations
    java_paths = [
        os.environ.get("JAVA_HOME"),
        "/usr/lib/jvm/java-11-openjdk-amd64",
        "/usr/lib/jvm/java-17-openjdk-amd64",
        "/usr/lib/jvm/java-8-openjdk-amd64",
        "/Library/Java/JavaVirtualMachines/openjdk-11.jdk/Contents/Home",
        "/Library/Java/JavaVirtualMachines/openjdk-17.jdk/Contents/Home",
    ]
    
    for path in java_paths:
        if path and os.path.exists(path):
            return path
    
    return None

def setup_environment():
    """Set up environment for PySpark tests."""
    # Find and set JAVA_HOME
    java_home = find_java_home()
    if not java_home:
        print("Error: Could not find Java installation.")
        print("Please install Java 8, 11, or 17 and set JAVA_HOME.")
        sys.exit(1)
    
    os.environ["JAVA_HOME"] = java_home
    print(f"Using Java at: {java_home}")
    
    # Set Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.environ["PYTHONPATH"] = project_root
    
    # Set Spark environment
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    
    # Reduce Spark verbosity
    os.environ["SPARK_CONF_DIR"] = ""
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 2g --executor-memory 2g pyspark-shell"
    
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {os.environ.get('PYTHONPATH')}")

def run_tests():
    """Run the test suite."""
    setup_environment()
    
    # Ensure we're in the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs(".pytest_cache", exist_ok=True)
    os.makedirs("/tmp/spark-warehouse", exist_ok=True)
    
    print("\nRunning PySpark tests...")
    print("-" * 60)
    
    # Run pytest with appropriate arguments
    pytest_args = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--disable-warnings",
        "-p", "no:warnings",
        "--log-cli-level=ERROR",
        "-x",  # Stop on first failure for debugging
        "--maxfail=5",  # Stop after 5 failures
        "--durations=10",  # Show slowest 10 tests
    ]
    
    # Run specific test file if provided
    if len(sys.argv) > 1:
        pytest_args.append(sys.argv[1])
    
    try:
        result = subprocess.run(pytest_args)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError running tests: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)