"""Health check module for HPI-FHFA pipeline"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


class HealthChecker:
    """Performs health checks on pipeline data and processes"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(__name__)
        self.checks_passed = []
        self.checks_failed = []
        
    def check_all(
        self,
        transactions: Optional[DataFrame] = None,
        repeat_sales: Optional[DataFrame] = None,
        indices: Optional[DataFrame] = None
    ) -> Tuple[bool, Dict[str, any]]:
        """Run all health checks and return status"""
        
        checks = []
        
        # Data availability checks
        if transactions is not None:
            checks.append(self.check_transaction_data(transactions))
            
        if repeat_sales is not None:
            checks.append(self.check_repeat_sales_data(repeat_sales))
            
        if indices is not None:
            checks.append(self.check_index_output(indices))
            
        # System health checks
        checks.append(self.check_spark_health())
        checks.append(self.check_memory_usage())
        
        # Aggregate results
        all_passed = all(check[0] for check in checks)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "HEALTHY" if all_passed else "UNHEALTHY",
            "checks_passed": len(self.checks_passed),
            "checks_failed": len(self.checks_failed),
            "details": {
                check[1]: check[2] for check in checks
            }
        }
        
        return all_passed, report
    
    def check_transaction_data(self, df: DataFrame) -> Tuple[bool, str, Dict]:
        """Check transaction data quality"""
        check_name = "transaction_data_quality"
        
        try:
            # Count records
            total_count = df.count()
            if total_count == 0:
                self._record_failure(check_name, "No transaction data found")
                return False, check_name, {"error": "No data"}
            
            # Check for nulls in critical columns
            critical_columns = ["property_id", "transaction_date", "transaction_price", "census_tract"]
            null_counts = {}
            
            for col in critical_columns:
                null_count = df.filter(F.col(col).isNull()).count()
                null_counts[col] = null_count
                
            # Check for invalid prices
            invalid_prices = df.filter(
                (F.col("transaction_price") <= 0) | 
                (F.col("transaction_price") > 100000000)  # $100M cap
            ).count()
            
            # Check date ranges
            date_stats = df.select(
                F.min("transaction_date").alias("min_date"),
                F.max("transaction_date").alias("max_date")
            ).collect()[0]
            
            # Determine pass/fail
            has_nulls = any(count > 0 for count in null_counts.values())
            has_invalid_prices = invalid_prices > 0
            
            if has_nulls or has_invalid_prices:
                self._record_failure(check_name, "Data quality issues found")
                passed = False
            else:
                self._record_success(check_name)
                passed = True
                
            details = {
                "total_records": total_count,
                "null_counts": null_counts,
                "invalid_prices": invalid_prices,
                "date_range": {
                    "min": str(date_stats["min_date"]),
                    "max": str(date_stats["max_date"])
                }
            }
            
            return passed, check_name, details
            
        except Exception as e:
            self._record_failure(check_name, str(e))
            return False, check_name, {"error": str(e)}
    
    def check_repeat_sales_data(self, df: DataFrame) -> Tuple[bool, str, Dict]:
        """Check repeat sales data quality"""
        check_name = "repeat_sales_data_quality"
        
        try:
            total_count = df.count()
            
            # Check CAGR distribution
            cagr_stats = df.select(
                F.min("cagr").alias("min_cagr"),
                F.max("cagr").alias("max_cagr"),
                F.avg("cagr").alias("avg_cagr"),
                F.stddev("cagr").alias("std_cagr")
            ).collect()[0]
            
            # Check time differences
            time_diff_stats = df.select(
                F.min("time_diff_years").alias("min_years"),
                F.max("time_diff_years").alias("max_years"),
                F.avg("time_diff_years").alias("avg_years")
            ).collect()[0]
            
            # Check for extreme values
            extreme_cagr = df.filter(F.abs(F.col("cagr")) > 0.5).count()
            
            # Validation
            issues = []
            if abs(cagr_stats["min_cagr"]) > 0.5:
                issues.append("Extreme negative CAGR detected")
            if cagr_stats["max_cagr"] > 0.5:
                issues.append("Extreme positive CAGR detected")
            if time_diff_stats["min_years"] < 0:
                issues.append("Negative time differences found")
                
            passed = len(issues) == 0
            
            if passed:
                self._record_success(check_name)
            else:
                self._record_failure(check_name, "; ".join(issues))
                
            details = {
                "total_pairs": total_count,
                "cagr_stats": {
                    "min": float(cagr_stats["min_cagr"]),
                    "max": float(cagr_stats["max_cagr"]),
                    "avg": float(cagr_stats["avg_cagr"]),
                    "std": float(cagr_stats["std_cagr"]) if cagr_stats["std_cagr"] else 0
                },
                "time_diff_stats": {
                    "min_years": float(time_diff_stats["min_years"]),
                    "max_years": float(time_diff_stats["max_years"]),
                    "avg_years": float(time_diff_stats["avg_years"])
                },
                "extreme_cagr_count": extreme_cagr,
                "issues": issues
            }
            
            return passed, check_name, details
            
        except Exception as e:
            self._record_failure(check_name, str(e))
            return False, check_name, {"error": str(e)}
    
    def check_index_output(self, df: DataFrame) -> Tuple[bool, str, Dict]:
        """Check index output quality"""
        check_name = "index_output_quality"
        
        try:
            total_count = df.count()
            
            # Check for required columns
            required_columns = ["cbsa_code", "year", "weight_type", "index_value"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self._record_failure(check_name, f"Missing columns: {missing_columns}")
                return False, check_name, {"missing_columns": missing_columns}
            
            # Check index values
            index_stats = df.select(
                F.min("index_value").alias("min_index"),
                F.max("index_value").alias("max_index"),
                F.avg("index_value").alias("avg_index")
            ).collect()[0]
            
            # Check for negative indices
            negative_indices = df.filter(F.col("index_value") < 0).count()
            
            # Check weight types
            weight_types = df.select("weight_type").distinct().collect()
            weight_type_list = [row["weight_type"] for row in weight_types]
            
            # Check year coverage
            year_range = df.select(
                F.min("year").alias("min_year"),
                F.max("year").alias("max_year")
            ).collect()[0]
            
            # Validation
            issues = []
            if negative_indices > 0:
                issues.append(f"{negative_indices} negative index values found")
            if len(weight_type_list) < 6:
                issues.append(f"Only {len(weight_type_list)} weight types found (expected 6)")
            if index_stats["min_index"] < 50:
                issues.append("Unusually low index values detected")
            if index_stats["max_index"] > 500:
                issues.append("Unusually high index values detected")
                
            passed = len(issues) == 0
            
            if passed:
                self._record_success(check_name)
            else:
                self._record_failure(check_name, "; ".join(issues))
                
            details = {
                "total_indices": total_count,
                "index_stats": {
                    "min": float(index_stats["min_index"]) if index_stats["min_index"] else 0,
                    "max": float(index_stats["max_index"]) if index_stats["max_index"] else 0,
                    "avg": float(index_stats["avg_index"]) if index_stats["avg_index"] else 0
                },
                "weight_types": weight_type_list,
                "year_range": {
                    "min": year_range["min_year"],
                    "max": year_range["max_year"]
                },
                "negative_indices": negative_indices,
                "issues": issues
            }
            
            return passed, check_name, details
            
        except Exception as e:
            self._record_failure(check_name, str(e))
            return False, check_name, {"error": str(e)}
    
    def check_spark_health(self) -> Tuple[bool, str, Dict]:
        """Check Spark cluster health"""
        check_name = "spark_cluster_health"
        
        try:
            sc = self.spark.sparkContext
            
            # Get cluster info
            status_tracker = sc.statusTracker()
            
            # Check executors
            executor_infos = status_tracker.getExecutorInfos()
            num_executors = len(executor_infos)
            
            # Check for active jobs
            active_jobs = len(status_tracker.getActiveJobIds())
            active_stages = len(status_tracker.getActiveStageIds())
            
            # Get application info
            app_id = sc.applicationId
            app_name = sc.appName
            
            # Validation
            issues = []
            if num_executors == 0:
                issues.append("No executors available")
                
            passed = len(issues) == 0
            
            if passed:
                self._record_success(check_name)
            else:
                self._record_failure(check_name, "; ".join(issues))
                
            details = {
                "application_id": app_id,
                "application_name": app_name,
                "num_executors": num_executors,
                "active_jobs": active_jobs,
                "active_stages": active_stages,
                "spark_version": sc.version
            }
            
            return passed, check_name, details
            
        except Exception as e:
            self._record_failure(check_name, str(e))
            return False, check_name, {"error": str(e)}
    
    def check_memory_usage(self) -> Tuple[bool, str, Dict]:
        """Check system memory usage"""
        check_name = "memory_usage"
        
        try:
            import psutil
            
            # Get memory info
            memory = psutil.virtual_memory()
            
            # Validation
            issues = []
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            if memory.available < 2 * (1024**3):  # Less than 2GB available
                issues.append(f"Low available memory: {memory.available / (1024**3):.1f}GB")
                
            passed = len(issues) == 0
            
            if passed:
                self._record_success(check_name)
            else:
                self._record_failure(check_name, "; ".join(issues))
                
            details = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent_used": memory.percent
            }
            
            return passed, check_name, details
            
        except Exception as e:
            self._record_failure(check_name, str(e))
            return False, check_name, {"error": str(e)}
    
    def _record_success(self, check_name: str):
        """Record a successful check"""
        self.checks_passed.append({
            "name": check_name,
            "timestamp": datetime.now().isoformat()
        })
        self.logger.info(f"Health check passed: {check_name}")
    
    def _record_failure(self, check_name: str, reason: str):
        """Record a failed check"""
        self.checks_failed.append({
            "name": check_name,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        self.logger.warning(f"Health check failed: {check_name} - {reason}")