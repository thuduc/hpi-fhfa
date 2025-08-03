#!/bin/bash
# Deployment script for HPI-FHFA Pipeline

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="staging"
AWS_REGION="us-east-1"
S3_BUCKET=""
EMR_APPLICATION_ID=""
DOCKER_REGISTRY=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -r|--region)
            AWS_REGION="$2"
            shift 2
            ;;
        -b|--bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        -a|--app-id)
            EMR_APPLICATION_ID="$2"
            shift 2
            ;;
        -d|--docker-registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -e, --environment    Environment (staging|production) [default: staging]"
            echo "  -r, --region        AWS region [default: us-east-1]"
            echo "  -b, --bucket        S3 bucket for code deployment"
            echo "  -a, --app-id        EMR Serverless application ID"
            echo "  -d, --docker-registry Docker registry URL"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    echo -e "${RED}Error: Invalid environment. Must be 'staging' or 'production'${NC}"
    exit 1
fi

echo -e "${GREEN}Deploying HPI-FHFA Pipeline to ${ENVIRONMENT}${NC}"

# Function to check command availability
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        exit 1
    fi
}

# Check required commands
check_command aws
check_command docker
check_command python3

# Load environment-specific configuration
CONFIG_FILE="config/${ENVIRONMENT}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${YELLOW}Warning: Configuration file $CONFIG_FILE not found${NC}"
fi

# Step 1: Run tests
echo -e "${GREEN}Step 1: Running tests...${NC}"
python3 -m pytest tests/unit -v --tb=short
if [[ $? -ne 0 ]]; then
    echo -e "${RED}Tests failed. Aborting deployment.${NC}"
    exit 1
fi

# Step 2: Build Docker image
echo -e "${GREEN}Step 2: Building Docker image...${NC}"
VERSION=$(git describe --tags --always)
IMAGE_NAME="hpi-fhfa-pyspark"
IMAGE_TAG="${VERSION}-${ENVIRONMENT}"

docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest-${ENVIRONMENT}

# Step 3: Push to registry (if provided)
if [[ -n "$DOCKER_REGISTRY" ]]; then
    echo -e "${GREEN}Step 3: Pushing to Docker registry...${NC}"
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
    docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
fi

# Step 4: Package Python code
echo -e "${GREEN}Step 4: Packaging Python code...${NC}"
rm -rf dist/
mkdir -p dist/

# Create zip file with dependencies
cd hpi_fhfa
zip -r ../dist/hpi_fhfa.zip . -x "*.pyc" -x "__pycache__/*"
cd ..

# Step 5: Deploy to S3 (if bucket provided)
if [[ -n "$S3_BUCKET" ]]; then
    echo -e "${GREEN}Step 5: Deploying to S3...${NC}"
    
    # Upload code
    aws s3 cp dist/hpi_fhfa.zip s3://${S3_BUCKET}/code/${VERSION}/hpi_fhfa.zip
    aws s3 cp scripts/run_pipeline.py s3://${S3_BUCKET}/scripts/${VERSION}/run_pipeline.py
    
    # Upload configuration
    if [[ -f "$CONFIG_FILE" ]]; then
        aws s3 cp $CONFIG_FILE s3://${S3_BUCKET}/config/${ENVIRONMENT}.yaml
    fi
    
    # Create latest symlinks
    aws s3 cp s3://${S3_BUCKET}/code/${VERSION}/hpi_fhfa.zip \
              s3://${S3_BUCKET}/code/latest/hpi_fhfa.zip
    aws s3 cp s3://${S3_BUCKET}/scripts/${VERSION}/run_pipeline.py \
              s3://${S3_BUCKET}/scripts/latest/run_pipeline.py
fi

# Step 6: Deploy to EMR Serverless (if app ID provided)
if [[ -n "$EMR_APPLICATION_ID" ]]; then
    echo -e "${GREEN}Step 6: Submitting job to EMR Serverless...${NC}"
    
    # Prepare job configuration
    if [[ "$ENVIRONMENT" == "production" ]]; then
        EXECUTOR_INSTANCES=20
        EXECUTOR_MEMORY="32g"
        EXECUTOR_CORES=8
    else
        EXECUTOR_INSTANCES=10
        EXECUTOR_MEMORY="16g"
        EXECUTOR_CORES=4
    fi
    
    # Submit job
    JOB_RUN_ID=$(aws emr-serverless start-job-run \
        --region ${AWS_REGION} \
        --application-id ${EMR_APPLICATION_ID} \
        --execution-role-arn arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/EMRServerlessRole \
        --name "HPI-Pipeline-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)" \
        --job-driver "{
            \"sparkSubmit\": {
                \"entryPoint\": \"s3://${S3_BUCKET}/scripts/latest/run_pipeline.py\",
                \"entryPointArguments\": [
                    \"--transaction-path\", \"s3://${S3_BUCKET}/data/${ENVIRONMENT}/transactions/\",
                    \"--geographic-path\", \"s3://${S3_BUCKET}/data/${ENVIRONMENT}/geographic/\",
                    \"--weight-path\", \"s3://${S3_BUCKET}/data/${ENVIRONMENT}/weights/\",
                    \"--output-path\", \"s3://${S3_BUCKET}/output/${ENVIRONMENT}/\",
                    \"--env\", \"${ENVIRONMENT}\"
                ],
                \"sparkSubmitParameters\": \"--py-files s3://${S3_BUCKET}/code/latest/hpi_fhfa.zip --conf spark.executor.instances=${EXECUTOR_INSTANCES} --conf spark.executor.memory=${EXECUTOR_MEMORY} --conf spark.executor.cores=${EXECUTOR_CORES}\"
            }
        }" \
        --configuration-overrides "{
            \"monitoringConfiguration\": {
                \"s3MonitoringConfiguration\": {
                    \"logUri\": \"s3://${S3_BUCKET}/logs/${ENVIRONMENT}/\"
                }
            }
        }" \
        --query 'jobRunId' \
        --output text)
    
    echo -e "${GREEN}Job submitted with ID: ${JOB_RUN_ID}${NC}"
    
    # Wait for job to start
    echo "Waiting for job to start..."
    sleep 10
    
    # Monitor job status
    while true; do
        STATUS=$(aws emr-serverless get-job-run \
            --region ${AWS_REGION} \
            --application-id ${EMR_APPLICATION_ID} \
            --job-run-id ${JOB_RUN_ID} \
            --query 'jobRun.state' \
            --output text)
        
        echo "Job status: $STATUS"
        
        if [[ "$STATUS" == "SUCCESS" ]]; then
            echo -e "${GREEN}Job completed successfully!${NC}"
            break
        elif [[ "$STATUS" == "FAILED" || "$STATUS" == "CANCELLED" ]]; then
            echo -e "${RED}Job failed with status: $STATUS${NC}"
            exit 1
        fi
        
        sleep 30
    done
fi

# Step 7: Update deployment metadata
echo -e "${GREEN}Step 7: Updating deployment metadata...${NC}"

# Create deployment record
cat > dist/deployment.json <<EOF
{
    "environment": "${ENVIRONMENT}",
    "version": "${VERSION}",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "git_commit": "$(git rev-parse HEAD)",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD)",
    "deployed_by": "$(whoami)",
    "image_tag": "${IMAGE_TAG}"
}
EOF

# Upload deployment metadata
if [[ -n "$S3_BUCKET" ]]; then
    aws s3 cp dist/deployment.json s3://${S3_BUCKET}/deployments/${ENVIRONMENT}/${VERSION}.json
    aws s3 cp dist/deployment.json s3://${S3_BUCKET}/deployments/${ENVIRONMENT}/latest.json
fi

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "Version: ${VERSION}"
echo -e "Environment: ${ENVIRONMENT}"
echo -e "Image: ${IMAGE_NAME}:${IMAGE_TAG}"