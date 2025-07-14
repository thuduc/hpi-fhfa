#!/usr/bin/env python3
"""
Script to run PySpark tests in a specific order to avoid session conflicts.
"""

import os
import sys
import subprocess
import platform
import time

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

def run_test_group(test_pattern, description):
    """Run a specific test group."""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    pytest_args = [
        sys.executable, "-m", "pytest",
        test_pattern,
        "-v",
        "--tb=short",
        "--disable-warnings",
        "-p", "no:warnings",
        "--log-cli-level=ERROR",
    ]
    
    try:
        result = subprocess.run(pytest_args, timeout=300)  # 5 min timeout per group
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False

def run_tests_ordered():
    """Run tests in a specific order to avoid conflicts."""
    setup_environment()
    
    # Ensure we're in the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs(".pytest_cache", exist_ok=True)
    os.makedirs("/tmp/spark-warehouse", exist_ok=True)
    
    print("\\nRunning PySpark RSAI tests in optimized order...")
    
    # Test groups in order of complexity and dependencies
    test_groups = [
        # 1. Non-Spark tests first
        ("tests/unit/test_data_models.py::TestDataModels", "Data Models (Non-Spark)"),
        ("tests/unit/test_data_models.py::TestEnums", "Enums (Non-Spark)"),
        
        # 2. Basic Spark schema tests
        ("tests/unit/test_data_models.py::TestSchemas", "Schema Creation"),
        
        # 3. Core functionality tests
        ("tests/unit/test_data_ingestion.py", "Data Ingestion"),
        
        # 4. Individual component tests
        ("tests/unit/test_weights.py", "Weight Calculation"),
        ("tests/unit/test_bmn_regression.py", "BMN Regression"),
        ("tests/unit/test_aggregation.py", "Index Aggregation"),
        
        # 5. Test infrastructure
        ("tests/test_spark_setup.py", "Spark Setup"),
        
        # 6. Integration tests last
        ("tests/integration/test_pipeline.py", "Pipeline Integration"),
    ]
    
    results = []
    failed_groups = []
    
    for test_pattern, description in test_groups:
        print(f"\\nWaiting 2 seconds for cleanup...")
        time.sleep(2)  # Give time for Spark sessions to clean up
        
        success = run_test_group(test_pattern, description)
        results.append((description, success))
        
        if not success:
            failed_groups.append(description)
    
    # Summary
    print(f"\\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description:40} {status}")
    
    print(f"\\nOverall: {passed}/{total} test groups passed")
    
    if failed_groups:
        print(f"\\nFailed groups: {', '.join(failed_groups)}")
        return 1
    else:
        print("\\n🎉 All test groups passed!")
        return 0

if __name__ == "__main__":
    exit_code = run_tests_ordered()
    sys.exit(exit_code)