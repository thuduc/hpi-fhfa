# HPI-FHFA PySpark Implementation

A distributed implementation of the FHFA House Price Index (HPI) calculation using Apache Spark and MLlib.

## 🏠 Overview

This project implements the FHFA Repeat-Sales Aggregation Index (RSAI) methodology using PySpark for distributed processing. It processes millions of repeat-sales pairs across census tracts and CBSAs to produce balanced panel indices with dynamic aggregation and flexible weighting schemes.

### Key Features

- **Distributed Processing**: Leverages Apache Spark for scalable data processing
- **BMN Regression**: Implements Bailey, Muth, and Nourse regression using MLlib
- **Dynamic Aggregation**: Supertract algorithm for census tract aggregation
- **Multiple Weighting Schemes**: 6 different index variants (sample, value, unit, UPB, college, nonwhite)
- **Production Ready**: Containerized deployment with CI/CD pipeline

## 📋 Requirements

- Python 3.9+
- Apache Spark 3.5.0+
- Java 17
- Docker (for containerized deployment)

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/hpi-fhfa-pyspark.git
   cd hpi-fhfa-pyspark
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run tests**
   ```bash
   # Unit tests
   pytest tests/unit -v

   # Integration tests
   pytest tests/integration -v -m integration

   # All tests with coverage
   pytest --cov=hpi_fhfa --cov-report=html
   ```

5. **Run the pipeline**
   ```bash
   python -m hpi_fhfa.pipeline.main_pipeline \
     --transaction-path data/transactions.parquet \
     --geographic-path data/geographic.parquet \
     --weight-path data/weights.parquet \
     --output-path output/indices.parquet \
     --start-year 2015 \
     --end-year 2021
   ```

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t hpi-fhfa-pyspark .
   ```

2. **Run with docker-compose**
   ```bash
   docker-compose up
   ```

3. **Access services**
   - Spark Master UI: http://localhost:8080
   - Spark Application UI: http://localhost:4040
   - Jupyter Lab: http://localhost:8888

## 📁 Project Structure

```
hpi-fhfa-pyspark/
├── hpi_fhfa/               # Main package
│   ├── core/              # Core algorithms (BMN, Supertract, Aggregation)
│   ├── etl/               # Data processing and validation
│   ├── pipeline/          # Main pipeline orchestrator
│   ├── schemas/           # Data schemas
│   ├── utils/             # Utilities
│   └── monitoring/        # Metrics and monitoring
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── performance/      # Performance tests
├── notebooks/            # Jupyter notebooks
├── scripts/              # Deployment scripts
├── config/               # Configuration files
└── docs/                 # Documentation
```

## 🔧 Configuration

### Spark Configuration

Key Spark settings for optimal performance:

```python
{
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.shuffle.partitions": "2000",
    "spark.executor.memory": "16g",
    "spark.executor.cores": "4",
    "spark.default.parallelism": "400"
}
```

### Pipeline Configuration

See `config/pipeline_config.yaml` for pipeline-specific settings:

```yaml
pipeline:
  name: HPI-FHFA-Pipeline
  version: 0.1.0

data:
  min_half_pairs: 40
  start_year: 1989
  end_year: 2021
  base_year: 1989

filters:
  max_cagr: 0.30
  min_appreciation_ratio: 0.25
  max_appreciation_ratio: 10.0
```

## 📊 Data Requirements

### Input Data Schemas

1. **Transaction Data**
   - property_id (string)
   - transaction_date (date)
   - transaction_price (double)
   - census_tract (string)
   - cbsa_code (string)
   - distance_to_cbd (double)

2. **Geographic Data**
   - census_tract (string)
   - cbsa_code (string)
   - centroid_lat (double)
   - centroid_lon (double)
   - adjacent_tracts (array<string>)

3. **Weight Data**
   - census_tract (string)
   - cbsa_code (string)
   - year (int)
   - value_measure (double)
   - unit_measure (double)
   - upb_measure (double)
   - college_share (double, nullable)
   - nonwhite_share (double, nullable)

## 🧪 Testing

The project includes comprehensive test coverage:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test end-to-end pipeline functionality
- **Performance Tests**: Benchmark processing speed and scalability

Run specific test categories:

```bash
# Only unit tests
pytest -m unit

# Only integration tests
pytest -m integration

# Only performance tests
pytest -m performance
```

## 🚢 Deployment

### AWS EMR Serverless

The recommended production deployment uses AWS EMR Serverless:

```bash
# Deploy to staging
./scripts/deploy.sh staging

# Deploy to production
./scripts/deploy.sh production
```

### Kubernetes

For Kubernetes deployment, use the provided Helm chart:

```bash
helm install hpi-fhfa ./k8s/helm/hpi-fhfa \
  --namespace data-processing \
  --values ./k8s/helm/values.prod.yaml
```

## 📈 Performance

Expected performance metrics:

- **Processing Time**: ~3-4 hours for 63.3M repeat-sales pairs
- **Memory Usage**: ~128GB total cluster memory
- **Scalability**: Linear scaling with data volume

## 🔍 Monitoring

The pipeline includes comprehensive monitoring:

- **Metrics**: Processing time, record counts, memory usage
- **Logging**: Structured JSON logging with correlation IDs
- **Alerts**: Configurable alerts for failures and SLA breaches

Access monitoring dashboards:
- Spark UI: View job progress and resource usage
- Application logs: Check `logs/` directory
- Metrics: Exported to configured monitoring system

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Ensure all tests pass before submitting PR

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FHFA for the HPI methodology
- Apache Spark community
- Contributors and maintainers

## 📞 Support

For issues and questions:
- Create an issue in GitHub
- Check existing documentation in `docs/`
- Contact the development team

---

Built with ❤️ using Apache Spark