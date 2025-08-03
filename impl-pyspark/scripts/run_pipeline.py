#!/usr/bin/env python
"""
Main entry point for running the HPI-FHFA pipeline
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hpi_fhfa.pipeline.main_pipeline import HPIPipeline
from hpi_fhfa.utils.logging_config import setup_logging


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run HPI-FHFA PySpark Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument(
        "--transaction-path",
        required=True,
        help="Path to transaction data (parquet)"
    )
    parser.add_argument(
        "--geographic-path",
        required=True,
        help="Path to geographic data (parquet)"
    )
    parser.add_argument(
        "--weight-path",
        required=True,
        help="Path to weight data (parquet)"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path for output indices"
    )
    
    # Pipeline parameters
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Start year for processing"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End year for processing"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to configuration YAML file"
    )
    
    # Spark configuration
    parser.add_argument(
        "--spark-master",
        default=None,
        help="Spark master URL (e.g., spark://master:7077)"
    )
    parser.add_argument(
        "--executor-memory",
        default="16g",
        help="Executor memory"
    )
    parser.add_argument(
        "--executor-cores",
        type=int,
        default=4,
        help="Cores per executor"
    )
    parser.add_argument(
        "--num-executors",
        type=int,
        default=None,
        help="Number of executors"
    )
    
    # Environment
    parser.add_argument(
        "--env",
        choices=["local", "development", "staging", "production"],
        default="local",
        help="Deployment environment"
    )
    
    # Monitoring
    parser.add_argument(
        "--metrics-enabled",
        action="store_true",
        help="Enable metrics collection"
    )
    parser.add_argument(
        "--metrics-endpoint",
        default=None,
        help="Metrics endpoint URL"
    )
    
    return parser.parse_args()


def get_spark_config(args):
    """Build Spark configuration from arguments"""
    config = {}
    
    # Basic configuration
    config["spark.app.name"] = f"HPI-FHFA-Pipeline-{args.env}"
    
    # Memory settings
    config["spark.executor.memory"] = args.executor_memory
    config["spark.executor.cores"] = str(args.executor_cores)
    
    if args.num_executors:
        config["spark.executor.instances"] = str(args.num_executors)
    
    # Environment-specific settings
    if args.env == "production":
        config.update({
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            "spark.sql.shuffle.partitions": "2000",
            "spark.default.parallelism": "400",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.sql.execution.arrow.pyspark.enabled": "true"
        })
    elif args.env == "staging":
        config.update({
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.shuffle.partitions": "1000",
            "spark.default.parallelism": "200"
        })
    else:  # local/development
        config.update({
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.shuffle.partitions": "100",
            "spark.ui.showConsoleProgress": "true"
        })
    
    # Master URL
    if args.spark_master:
        config["spark.master"] = args.spark_master
    
    return config


def validate_paths(args):
    """Validate input paths exist"""
    paths_to_check = [
        ("Transaction data", args.transaction_path),
        ("Geographic data", args.geographic_path),
        ("Weight data", args.weight_path)
    ]
    
    # For S3 paths, skip validation (assume they exist)
    for name, path in paths_to_check:
        if path.startswith("s3://") or path.startswith("s3a://"):
            continue
        if path.startswith("hdfs://"):
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at: {path}")


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging("run_pipeline")
    logger.info(f"Starting HPI-FHFA Pipeline in {args.env} environment")
    
    try:
        # Validate paths
        validate_paths(args)
        
        # Get Spark configuration
        spark_config = get_spark_config(args)
        
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = HPIPipeline(
            config_path=args.config,
            spark_config=spark_config
        )
        
        # Run pipeline
        logger.info("Running pipeline...")
        start_time = datetime.now()
        
        metrics = pipeline.run_pipeline(
            transaction_path=args.transaction_path,
            geographic_path=args.geographic_path,
            weight_data_path=args.weight_path,
            output_path=args.output_path,
            start_year=args.start_year,
            end_year=args.end_year
        )
        
        # Log results
        if metrics["status"] == "SUCCESS":
            logger.info(
                f"Pipeline completed successfully in {metrics['duration_seconds']:.1f} seconds"
            )
            logger.info(f"Processed {metrics['repeat_sales_count']:,} repeat-sales pairs")
            logger.info(f"Generated {metrics['output_rows']:,} output rows")
            
            # Send metrics if enabled
            if args.metrics_enabled and args.metrics_endpoint:
                send_metrics(metrics, args.metrics_endpoint)
            
            return 0
        else:
            logger.error(f"Pipeline failed: {metrics.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return 1
    finally:
        # Cleanup
        if 'pipeline' in locals():
            pipeline.spark.stop()


def send_metrics(metrics: dict, endpoint: str):
    """Send metrics to monitoring endpoint"""
    import requests
    import json
    
    try:
        # Format metrics for monitoring system
        formatted_metrics = {
            "timestamp": datetime.now().isoformat(),
            "pipeline": "hpi-fhfa",
            "status": metrics["status"],
            "duration_seconds": metrics["duration_seconds"],
            "repeat_sales_count": metrics["repeat_sales_count"],
            "output_rows": metrics["output_rows"]
        }
        
        # Send to endpoint
        response = requests.post(
            endpoint,
            json=formatted_metrics,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        
    except Exception as e:
        logging.warning(f"Failed to send metrics: {str(e)}")


if __name__ == "__main__":
    sys.exit(main())