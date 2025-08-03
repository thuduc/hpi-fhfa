# Deployment Guide

## Overview

This guide covers deployment options for the HPI-FHFA PySpark pipeline across different environments.

## Deployment Options

### 1. Local Development

For development and testing on a single machine.

```bash
# Using spark-submit
spark-submit \
  --master local[*] \
  --driver-memory 4g \
  --executor-memory 4g \
  --conf spark.sql.adaptive.enabled=true \
  scripts/run_pipeline.py \
    --transaction-path data/transactions.parquet \
    --geographic-path data/geographic.parquet \
    --weight-path data/weights.parquet \
    --output-path output/indices.parquet
```

### 2. Docker Deployment

#### Single Container
```bash
docker run -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  hpi-fhfa-pyspark:latest
```

#### Docker Compose Cluster
```bash
# Start the cluster
docker-compose up -d

# Scale workers
docker-compose up -d --scale spark-worker=4

# Submit job to cluster
docker exec -it hpi-spark-master spark-submit \
  --master spark://spark-master:7077 \
  --deploy-mode client \
  scripts/run_pipeline.py
```

### 3. AWS EMR Serverless

#### Prerequisites
- AWS CLI configured
- EMR Serverless application created
- S3 buckets for data and logs

#### Deployment Steps

1. **Upload code to S3**
```bash
aws s3 cp scripts/run_pipeline.py s3://your-bucket/scripts/
aws s3 cp --recursive hpi_fhfa/ s3://your-bucket/code/hpi_fhfa/
```

2. **Submit job**
```bash
aws emr-serverless start-job-run \
  --application-id app-xxxxxxxxxxxx \
  --execution-role-arn arn:aws:iam::account:role/EMRServerlessRole \
  --job-driver '{
    "sparkSubmit": {
      "entryPoint": "s3://your-bucket/scripts/run_pipeline.py",
      "entryPointArguments": [
        "--transaction-path", "s3://your-bucket/data/transactions/",
        "--geographic-path", "s3://your-bucket/data/geographic/",
        "--weight-path", "s3://your-bucket/data/weights/",
        "--output-path", "s3://your-bucket/output/"
      ],
      "sparkSubmitParameters": "--py-files s3://your-bucket/code/hpi_fhfa.zip"
    }
  }' \
  --configuration-overrides '{
    "applicationConfiguration": [{
      "classification": "spark-defaults",
      "properties": {
        "spark.executor.instances": "20",
        "spark.executor.memory": "16g",
        "spark.executor.cores": "4"
      }
    }]
  }'
```

### 4. Kubernetes Deployment

#### Using Spark Operator

1. **Install Spark Operator**
```bash
helm repo add spark-operator https://googlecloudplatform.github.io/spark-on-k8s-operator
helm install spark-operator spark-operator/spark-operator \
  --namespace spark-operator --create-namespace
```

2. **Create SparkApplication**
```yaml
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: hpi-fhfa-pipeline
  namespace: default
spec:
  type: Python
  pythonVersion: "3"
  mode: cluster
  image: "hpi-fhfa-pyspark:latest"
  imagePullPolicy: Always
  mainApplicationFile: "local:///app/scripts/run_pipeline.py"
  sparkVersion: "3.5.0"
  driver:
    cores: 2
    memory: "8g"
    serviceAccount: spark
  executor:
    cores: 4
    instances: 10
    memory: "16g"
  deps:
    pyFiles:
      - "local:///app/hpi_fhfa.zip"
  arguments:
    - "--transaction-path"
    - "s3a://bucket/data/transactions/"
    - "--geographic-path"
    - "s3a://bucket/data/geographic/"
    - "--weight-path"
    - "s3a://bucket/data/weights/"
    - "--output-path"
    - "s3a://bucket/output/"
```

3. **Submit application**
```bash
kubectl apply -f spark-application.yaml
```

### 5. Databricks Deployment

1. **Create Databricks job**
```json
{
  "name": "HPI-FHFA Pipeline",
  "tasks": [{
    "task_key": "main",
    "spark_python_task": {
      "python_file": "dbfs:/scripts/run_pipeline.py",
      "parameters": [
        "--transaction-path", "dbfs:/data/transactions/",
        "--geographic-path", "dbfs:/data/geographic/",
        "--weight-path", "dbfs:/data/weights/",
        "--output-path", "dbfs:/output/"
      ]
    },
    "new_cluster": {
      "spark_version": "13.3.x-scala2.12",
      "node_type_id": "i3.xlarge",
      "num_workers": 10,
      "spark_conf": {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true"
      }
    }
  }]
}
```

## Configuration Management

### Environment Variables
```bash
# Spark configuration
export SPARK_MASTER_URL=spark://master:7077
export SPARK_EXECUTOR_MEMORY=16g
export SPARK_EXECUTOR_CORES=4

# Data paths
export HPI_TRANSACTION_PATH=s3://bucket/data/transactions/
export HPI_GEOGRAPHIC_PATH=s3://bucket/data/geographic/
export HPI_WEIGHT_PATH=s3://bucket/data/weights/
export HPI_OUTPUT_PATH=s3://bucket/output/

# Pipeline configuration
export HPI_START_YEAR=2015
export HPI_END_YEAR=2021
export HPI_MIN_HALF_PAIRS=40
```

### Configuration Files

#### config/production.yaml
```yaml
spark:
  master: spark://master:7077
  executor:
    instances: 20
    memory: 16g
    cores: 4
  driver:
    memory: 8g
    cores: 2
  conf:
    spark.sql.adaptive.enabled: true
    spark.sql.shuffle.partitions: 2000

pipeline:
  data:
    min_half_pairs: 40
    start_year: 1989
    end_year: 2021
  filters:
    max_cagr: 0.30
    min_appreciation_ratio: 0.25
    max_appreciation_ratio: 10.0

storage:
  input:
    format: parquet
    compression: snappy
  output:
    format: parquet
    compression: snappy
    partitionBy: ["cbsa_code", "year"]
```

## Monitoring and Logging

### Spark UI Access

#### Port Forwarding (Kubernetes)
```bash
kubectl port-forward service/spark-ui-svc 4040:4040
```

#### SSH Tunnel (EMR)
```bash
ssh -i key.pem -L 4040:localhost:4040 hadoop@emr-master
```

### Log Collection

#### CloudWatch (AWS)
```json
{
  "logGroup": "/aws/emr-serverless/hpi-fhfa",
  "logStreamPrefix": "driver",
  "awslogs-region": "us-east-1"
}
```

#### Fluentd (Kubernetes)
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/spark/*.log
      tag spark.*
      <parse>
        @type json
      </parse>
    </source>
    
    <match spark.**>
      @type elasticsearch
      host elasticsearch.default.svc.cluster.local
      port 9200
      index_name spark-logs
    </match>
```

## Performance Tuning

### Memory Settings
```bash
# Driver memory (coordination)
--driver-memory 8g

# Executor memory (processing)
--executor-memory 16g

# Memory overhead (off-heap)
--conf spark.executor.memoryOverhead=4g
```

### Parallelism Settings
```bash
# Number of executors
--num-executors 20

# Cores per executor
--executor-cores 4

# Default parallelism
--conf spark.default.parallelism=400

# Shuffle partitions
--conf spark.sql.shuffle.partitions=2000
```

### Adaptive Query Execution
```bash
--conf spark.sql.adaptive.enabled=true
--conf spark.sql.adaptive.coalescePartitions.enabled=true
--conf spark.sql.adaptive.skewJoin.enabled=true
--conf spark.sql.adaptive.localShuffleReader.enabled=true
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Increase executor memory
   - Reduce partition size
   - Enable off-heap memory

2. **Slow Shuffles**
   - Increase shuffle partitions
   - Enable adaptive query execution
   - Use SSD-backed instances

3. **Data Skew**
   - Enable adaptive skew join
   - Salt keys for heavily skewed data
   - Use custom partitioner

### Debug Commands
```bash
# Check application status
kubectl get sparkapplications

# View driver logs
kubectl logs -f spark-driver-pod

# Access Spark shell
kubectl exec -it spark-driver-pod -- spark-shell

# Monitor resource usage
kubectl top pods -l spark-role=executor
```

## Security Considerations

### Network Security
- Use VPC endpoints for S3 access
- Configure security groups for Spark ports
- Enable SSL/TLS for Spark UI

### Access Control
- IAM roles for AWS resources
- RBAC for Kubernetes
- Kerberos for Hadoop clusters

### Data Encryption
- Enable S3 server-side encryption
- Use encrypted EBS volumes
- Configure Spark SSL settings

## Backup and Recovery

### Checkpoint Strategy
```python
# Configure checkpointing
spark.sparkContext.setCheckpointDir("s3://bucket/checkpoints/")

# Checkpoint DataFrames
df.checkpoint()
```

### Recovery Procedure
1. Identify last successful checkpoint
2. Resume from checkpoint
3. Reprocess failed partitions
4. Validate output consistency