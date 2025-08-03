"""Metrics collection and monitoring for HPI-FHFA pipeline"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from functools import wraps
import psutil
import threading

from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkContext


class PipelineMetrics:
    """Collects and tracks metrics throughout pipeline execution"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.sc = spark.sparkContext
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            "start_time": datetime.now().isoformat(),
            "stages": {},
            "resources": {},
            "data_quality": {},
            "performance": {}
        }
        self._stage_stack = []
        
    def record_stage(self, stage_name: str):
        """Decorator to record metrics for a pipeline stage"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                stage_metrics = {
                    "start_time": time.time(),
                    "status": "running"
                }
                
                self._stage_stack.append(stage_name)
                self.metrics["stages"][stage_name] = stage_metrics
                
                try:
                    # Record initial metrics
                    stage_metrics["initial_metrics"] = self._get_spark_metrics()
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Record final metrics
                    stage_metrics["end_time"] = time.time()
                    stage_metrics["duration_seconds"] = stage_metrics["end_time"] - stage_metrics["start_time"]
                    stage_metrics["final_metrics"] = self._get_spark_metrics()
                    stage_metrics["status"] = "completed"
                    
                    # Calculate deltas
                    self._calculate_stage_deltas(stage_metrics)
                    
                    return result
                    
                except Exception as e:
                    stage_metrics["end_time"] = time.time()
                    stage_metrics["duration_seconds"] = stage_metrics["end_time"] - stage_metrics["start_time"]
                    stage_metrics["status"] = "failed"
                    stage_metrics["error"] = str(e)
                    raise
                    
                finally:
                    self._stage_stack.pop()
                    
            return wrapper
        return decorator
    
    def record_dataframe_stats(self, df: DataFrame, name: str):
        """Record statistics about a DataFrame"""
        try:
            stats = {
                "count": df.count(),
                "columns": len(df.columns),
                "partitions": df.rdd.getNumPartitions()
            }
            
            if not hasattr(self.metrics["data_quality"], name):
                self.metrics["data_quality"][name] = {}
            
            self.metrics["data_quality"][name].update(stats)
            
        except Exception as e:
            self.logger.warning(f"Failed to record DataFrame stats for {name}: {e}")
    
    def record_data_quality_check(self, check_name: str, passed: bool, details: Optional[Dict] = None):
        """Record results of a data quality check"""
        check_result = {
            "passed": passed,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        if "quality_checks" not in self.metrics["data_quality"]:
            self.metrics["data_quality"]["quality_checks"] = {}
            
        self.metrics["data_quality"]["quality_checks"][check_name] = check_result
    
    def record_resource_usage(self):
        """Record current resource usage"""
        try:
            # System resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Spark resources
            status_tracker = self.sc.statusTracker()
            
            resource_metrics = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_available_gb": memory.available / (1024**3)
                },
                "spark": {
                    "active_jobs": len(status_tracker.getActiveJobIds()),
                    "active_stages": len(status_tracker.getActiveStageIds()),
                    "executors": self._get_executor_metrics()
                }
            }
            
            # Add to time series
            if "time_series" not in self.metrics["resources"]:
                self.metrics["resources"]["time_series"] = []
                
            self.metrics["resources"]["time_series"].append(resource_metrics)
            
        except Exception as e:
            self.logger.warning(f"Failed to record resource usage: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics"""
        summary = {
            "pipeline_duration_seconds": time.time() - datetime.fromisoformat(
                self.metrics["start_time"]).timestamp(),
            "stages_completed": sum(1 for s in self.metrics["stages"].values() 
                                  if s.get("status") == "completed"),
            "stages_failed": sum(1 for s in self.metrics["stages"].values() 
                               if s.get("status") == "failed"),
            "total_shuffle_bytes": sum(
                s.get("final_metrics", {}).get("shuffle_write_bytes", 0) 
                for s in self.metrics["stages"].values()
            ),
            "data_quality_checks_passed": sum(
                1 for check in self.metrics["data_quality"].get("quality_checks", {}).values()
                if check["passed"]
            )
        }
        
        return {**self.metrics, "summary": summary}
    
    def export_metrics(self, output_path: str):
        """Export metrics to JSON file"""
        try:
            metrics_data = self.get_summary()
            with open(output_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            self.logger.info(f"Metrics exported to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
    
    def _get_spark_metrics(self) -> Dict[str, Any]:
        """Get current Spark metrics"""
        try:
            status = self.sc.statusTracker()
            
            # Get job and stage info
            active_jobs = status.getActiveJobIds()
            active_stages = status.getActiveStageIds()
            
            # Accumulate metrics
            metrics = {
                "active_jobs": len(active_jobs),
                "active_stages": len(active_stages),
                "shuffle_read_bytes": 0,
                "shuffle_write_bytes": 0,
                "input_bytes": 0,
                "output_bytes": 0
            }
            
            # Get stage metrics
            for stage_id in active_stages:
                stage_info = status.getStageInfo(stage_id)
                if stage_info:
                    metrics["shuffle_read_bytes"] += getattr(stage_info, "shuffleReadBytes", 0)
                    metrics["shuffle_write_bytes"] += getattr(stage_info, "shuffleWriteBytes", 0)
                    metrics["input_bytes"] += getattr(stage_info, "inputBytes", 0)
                    metrics["output_bytes"] += getattr(stage_info, "outputBytes", 0)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to get Spark metrics: {e}")
            return {}
    
    def _get_executor_metrics(self) -> Dict[str, Any]:
        """Get executor-level metrics"""
        try:
            # Get executor information from Spark context
            status = self.sc.statusTracker()
            executor_infos = status.getExecutorInfos()
            
            metrics = {
                "count": len(executor_infos),
                "total_cores": sum(e.totalCores for e in executor_infos),
                "total_memory_mb": sum(getattr(e, "maxMemory", 0) for e in executor_infos) / (1024*1024)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to get executor metrics: {e}")
            return {"count": 0, "total_cores": 0, "total_memory_mb": 0}
    
    def _calculate_stage_deltas(self, stage_metrics: Dict[str, Any]):
        """Calculate metric deltas for a stage"""
        initial = stage_metrics.get("initial_metrics", {})
        final = stage_metrics.get("final_metrics", {})
        
        stage_metrics["deltas"] = {
            "shuffle_read_bytes": final.get("shuffle_read_bytes", 0) - initial.get("shuffle_read_bytes", 0),
            "shuffle_write_bytes": final.get("shuffle_write_bytes", 0) - initial.get("shuffle_write_bytes", 0),
            "input_bytes": final.get("input_bytes", 0) - initial.get("input_bytes", 0),
            "output_bytes": final.get("output_bytes", 0) - initial.get("output_bytes", 0)
        }


class MetricsCollector:
    """Background metrics collector for continuous monitoring"""
    
    def __init__(self, metrics: PipelineMetrics, interval_seconds: int = 30):
        self.metrics = metrics
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread = None
        
    def start(self):
        """Start background metrics collection"""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._collect_loop)
            self._thread.daemon = True
            self._thread.start()
    
    def stop(self):
        """Stop background metrics collection"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
    
    def _collect_loop(self):
        """Main collection loop"""
        while not self._stop_event.is_set():
            try:
                self.metrics.record_resource_usage()
            except Exception as e:
                logging.warning(f"Error in metrics collection: {e}")
            
            # Wait for next collection interval
            self._stop_event.wait(self.interval)


def track_performance(spark: SparkSession):
    """Decorator factory for tracking function performance"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024**3)  # GB
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / (1024**3)  # GB
                
                perf_metrics = {
                    "function": func.__name__,
                    "duration_seconds": end_time - start_time,
                    "memory_delta_gb": end_memory - start_memory,
                    "timestamp": datetime.now().isoformat()
                }
                
                logging.info(f"Performance metrics: {json.dumps(perf_metrics)}")
                
                return result
                
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {e}")
                raise
                
        return wrapper
    return decorator