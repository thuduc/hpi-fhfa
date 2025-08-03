#!/bin/bash
# Local development script for HPI-FHFA Pipeline

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Running HPI-FHFA Pipeline locally...${NC}"

# Default paths
DATA_DIR="${DATA_DIR:-./test_data}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# Create directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Check if test data exists
if [[ ! -d "$DATA_DIR" ]]; then
    echo -e "${YELLOW}Test data not found. Generating sample data...${NC}"
    python scripts/generate_sample_data.py --output-dir "$DATA_DIR"
fi

# Run the pipeline
spark-submit \
    --master "local[*]" \
    --driver-memory 4g \
    --conf spark.sql.adaptive.enabled=true \
    --conf spark.sql.adaptive.coalescePartitions.enabled=true \
    --conf spark.sql.shuffle.partitions=10 \
    --conf spark.ui.showConsoleProgress=true \
    --conf spark.sql.execution.arrow.pyspark.enabled=true \
    scripts/run_pipeline.py \
        --transaction-path "$DATA_DIR/transactions" \
        --geographic-path "$DATA_DIR/geographic" \
        --weight-path "$DATA_DIR/weights" \
        --output-path "$OUTPUT_DIR/indices" \
        --start-year 2015 \
        --end-year 2021 \
        --env local

echo -e "${GREEN}Pipeline completed! Check output at: $OUTPUT_DIR${NC}"