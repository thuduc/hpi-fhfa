#!/bin/bash

# Script to run HPI-FHFA pipeline with Spark

# Set defaults
MASTER="local[*]"
DRIVER_MEMORY="4g"
EXECUTOR_MEMORY="4g"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --master)
      MASTER="$2"
      shift 2
      ;;
    --driver-memory)
      DRIVER_MEMORY="$2"
      shift 2
      ;;
    --executor-memory)
      EXECUTOR_MEMORY="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

# Run spark-submit
spark-submit \
  --master "$MASTER" \
  --driver-memory "$DRIVER_MEMORY" \
  --executor-memory "$EXECUTOR_MEMORY" \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.adaptive.coalescePartitions.enabled=true \
  --py-files ../hpi_fhfa.zip \
  ../hpi_fhfa/pipeline/main_pipeline.py "$@"