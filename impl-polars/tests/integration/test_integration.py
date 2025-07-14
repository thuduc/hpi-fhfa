"""
Integration tests for the complete RSAI pipeline.

Tests the full workflow from data ingestion through index calculation
and output generation using Polars DataFrames.
"""

import pytest
from datetime import date, timedelta
from pathlib import Path
import polars as pl
import numpy as np
import tempfile
import shutil

from rsai.src.data.models import (
    RSAIConfig,
    GeographyLevel,
    WeightingScheme,
    TransactionType,
    PropertyType
)
from rsai.src.data.ingestion import DataIngestion
from rsai.src.geography.supertract import SupertractGenerator
from rsai.src.index.bmn_regression import BMNRegression
from rsai.src.index.weights import WeightCalculator
from rsai.src.output.export import OutputExporter


@pytest.mark.integration
class TestRSAIPipeline:
    """Test complete RSAI pipeline."""
    
    @pytest.fixture
    def integration_config(self):
        """Create configuration for integration tests."""
        return RSAIConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            frequency="monthly",
            geography_levels=[GeographyLevel.TRACT, GeographyLevel.COUNTY],
            weighting_scheme=WeightingScheme.BMN,
            min_pairs_threshold=5,
            max_holding_period_years=10,
            min_price=50000,
            max_price=2000000,
            outlier_std_threshold=3.0,
            output_format="parquet",
            include_diagnostics=True
        )
    
    @pytest.fixture
    def test_data_dir(self, tmp_path):
        """Create directory with test data files."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        
        # Generate realistic test data
        np.random.seed(42)
        
        # Create properties
        n_properties = 500
        properties = pl.DataFrame({
            "property_id": [f"P{i:06d}" for i in range(1, n_properties + 1)],
            "latitude": np.random.uniform(33.5, 34.5, n_properties),
            "longitude": np.random.uniform(-118.8, -117.8, n_properties),
            "zip_code": [f"900{np.random.randint(10, 99)}" for _ in range(n_properties)],
            "tract": [f"06037{np.random.randint(1000, 2000):04d}" for _ in range(n_properties)],
            "county_fips": ["06037"] * n_properties,
            "living_area": np.random.uniform(800, 5000, n_properties),
            "bedrooms": np.random.choice([1, 2, 3, 4, 5], n_properties),
            "year_built": np.random.randint(1950, 2020, n_properties),
            "property_type": np.random.choice(
                [PropertyType.SINGLE_FAMILY.value, PropertyType.CONDO.value],
                n_properties,
                p=[0.7, 0.3]
            )
        })
        
        # Save properties
        properties_path = data_dir / "properties.parquet"
        properties.write_parquet(properties_path)
        
        # Create transactions with realistic patterns
        transactions = []
        transaction_id = 1
        
        for prop_row in properties.iter_rows(named=True):
            prop_id = prop_row["property_id"]
            base_price = np.random.lognormal(12.5, 0.5)  # Log-normal price distribution
            
            # Number of sales for this property (most have 1-2, some have more)
            n_sales = np.random.choice([1, 2, 3, 4], p=[0.4, 0.4, 0.15, 0.05])
            
            sale_date = date(2020, 1, 1) + timedelta(days=np.random.randint(0, 365))
            
            for sale_num in range(n_sales):
                # Price appreciation
                years_passed = sale_num * np.random.uniform(1, 3)
                appreciation = 1 + 0.05 * years_passed + np.random.normal(0, 0.02)
                sale_price = base_price * appreciation
                
                transactions.append({
                    "transaction_id": f"T{transaction_id:08d}",
                    "property_id": prop_id,
                    "sale_price": sale_price,
                    "sale_date": sale_date,
                    "transaction_type": np.random.choice(
                        [TransactionType.ARMS_LENGTH.value, TransactionType.NON_ARMS_LENGTH.value],
                        p=[0.95, 0.05]
                    ),
                    "tract": prop_row["tract"],
                    "county_fips": prop_row["county_fips"]
                })
                
                transaction_id += 1
                sale_date = sale_date + timedelta(days=int(365 * years_passed))
        
        transactions_df = pl.DataFrame(transactions)
        
        # Save transactions
        transactions_path = data_dir / "transactions.parquet"
        transactions_df.write_parquet(transactions_path)
        
        return data_dir
    
    def test_full_pipeline(self, integration_config, test_data_dir, tmp_path):
        """Test complete pipeline from data loading to index generation."""
        output_dir = tmp_path / "output"
        
        # Step 1: Data Ingestion
        ingestion = DataIngestion(integration_config)
        
        # Load data
        transactions_df = ingestion.load_transactions(test_data_dir / "transactions.parquet")
        properties_df = ingestion.load_properties(test_data_dir / "properties.parquet")
        
        assert len(transactions_df) > 0
        assert len(properties_df) > 0
        
        # Identify repeat sales
        repeat_sales_df = ingestion.identify_repeat_sales()
        assert len(repeat_sales_df) > 0
        
        # Merge geographic data
        repeat_sales_df = ingestion.merge_geographic_data()
        assert "latitude" in repeat_sales_df.columns
        
        # Filter outliers
        repeat_sales_df = ingestion.filter_outliers(method="iqr")
        initial_count = len(repeat_sales_df)
        
        # Step 2: Geographic Aggregation
        supertract_gen = SupertractGenerator(
            min_transactions=20,
            max_distance_km=5.0,
            min_tracts=1,
            max_tracts=10,
            method="hierarchical"
        )
        
        tract_stats = supertract_gen.prepare_tract_data(transactions_df, properties_df)
        supertracts = supertract_gen.generate_supertracts(tract_stats)
        
        assert len(supertracts) > 0
        
        # Add supertract mapping to repeat sales
        mapping_df = supertract_gen.export_mapping()
        repeat_sales_df = repeat_sales_df.join(
            mapping_df.select(["tract_id", "supertract_id"]).rename({"tract_id": "tract"}),
            on="tract",
            how="left"
        )
        
        # Step 3: Weight Calculation
        weight_calc = WeightCalculator(integration_config.weighting_scheme)
        repeat_sales_df = weight_calc.calculate_weights(repeat_sales_df)
        
        assert "weight" in repeat_sales_df.columns
        assert all(w > 0 for w in repeat_sales_df["weight"])
        
        # Step 4: BMN Regression
        bmn = BMNRegression(
            frequency=integration_config.frequency,
            min_pairs_per_period=integration_config.min_pairs_threshold,
            weighted=True
        )
        
        # Fit models for different geographic levels
        regression_results = {}
        
        # County level
        county_results = bmn.fit_multiple_geographies(
            repeat_sales_df,
            "county_fips",
            GeographyLevel.COUNTY,
            weights_df=repeat_sales_df.select(["pair_id", "weight", "county_fips"]),
            min_pairs=10
        )
        regression_results.update(county_results)
        
        assert len(regression_results) > 0
        
        # Extract all index values
        all_index_values = []
        for result in regression_results.values():
            all_index_values.extend(result.index_values)
        
        assert len(all_index_values) > 0
        
        # Step 5: Output Generation
        exporter = OutputExporter(output_dir, integration_config)
        
        # Export index values
        index_path = exporter.export_index_values(
            all_index_values,
            format="parquet"
        )
        assert index_path.exists()
        
        # Export regression results
        regression_path = exporter.export_regression_results(
            regression_results,
            include_diagnostics=True
        )
        assert regression_path.exists()
        
        # Generate summary report
        index_df = pl.read_parquet(index_path)
        report_path = exporter.generate_summary_report(
            index_df,
            format="html"
        )
        assert report_path.exists()
        
        # Create plots
        plot_paths = exporter.create_index_plots(
            index_df,
            interactive=False
        )
        assert len(plot_paths) > 0
        
        # Verify final outputs
        assert len(list(output_dir.rglob("*.parquet"))) >= 1
        assert len(list(output_dir.rglob("*.json"))) >= 1
        assert len(list(output_dir.rglob("*.html"))) >= 1
    
    def test_pipeline_with_multiple_counties(self, integration_config, tmp_path):
        """Test pipeline with data from multiple counties."""
        # Create multi-county test data
        np.random.seed(42)
        
        counties = ["06037", "06059", "06073"]  # LA, Orange, San Diego
        transactions = []
        properties = []
        
        prop_id = 1
        trans_id = 1
        
        for county in counties:
            # Create properties for each county
            for i in range(100):
                prop_id_str = f"P{prop_id:06d}"
                properties.append({
                    "property_id": prop_id_str,
                    "latitude": 34.0 + np.random.uniform(-1, 1),
                    "longitude": -118.0 + np.random.uniform(-1, 1),
                    "tract": f"{county}{np.random.randint(1000, 2000):04d}",
                    "county_fips": county
                })
                
                # Create 2-3 transactions per property
                n_trans = np.random.choice([2, 3])
                base_price = np.random.lognormal(12.5, 0.5)
                sale_date = date(2020, 1, 1)
                
                for j in range(n_trans):
                    price = base_price * (1 + 0.05 * j + np.random.normal(0, 0.02))
                    transactions.append({
                        "transaction_id": f"T{trans_id:08d}",
                        "property_id": prop_id_str,
                        "sale_price": price,
                        "sale_date": sale_date + timedelta(days=365 * j),
                        "transaction_type": TransactionType.ARMS_LENGTH.value,
                        "tract": properties[-1]["tract"],
                        "county_fips": county
                    })
                    trans_id += 1
                
                prop_id += 1
        
        # Create DataFrames
        properties_df = pl.DataFrame(properties)
        transactions_df = pl.DataFrame(transactions)
        
        # Run pipeline
        ingestion = DataIngestion(integration_config)
        ingestion.transactions_df = transactions_df
        ingestion.properties_df = properties_df
        
        repeat_sales_df = ingestion.identify_repeat_sales()
        repeat_sales_df = ingestion.merge_geographic_data()
        
        # Verify we have data from all counties
        counties_in_data = repeat_sales_df["county_fips"].unique().to_list()
        assert all(county in counties_in_data for county in counties)
        
        # Run regression for each county
        bmn = BMNRegression(frequency="monthly", min_pairs_per_period=5)
        results = bmn.fit_multiple_geographies(
            repeat_sales_df,
            "county_fips",
            GeographyLevel.COUNTY,
            min_pairs=20
        )
        
        # Should have results for counties with enough data
        assert len(results) >= 2
    
    def test_pipeline_error_handling(self, integration_config, tmp_path):
        """Test pipeline error handling and recovery."""
        # Create minimal data that will cause issues
        transactions_df = pl.DataFrame({
            "transaction_id": ["T1", "T2"],
            "property_id": ["P1", "P2"],  # No repeat sales
            "sale_price": [100000, 200000],
            "sale_date": [date(2020, 1, 1), date(2020, 6, 1)],
            "transaction_type": [TransactionType.ARMS_LENGTH.value] * 2,
            "tract": ["06037001", "06037002"],
            "county_fips": ["06037"] * 2
        })
        
        ingestion = DataIngestion(integration_config)
        ingestion.transactions_df = transactions_df
        
        # Should handle no repeat sales gracefully
        repeat_sales_df = ingestion.identify_repeat_sales()
        assert len(repeat_sales_df) == 0
        
        # BMN regression should fail with no data
        bmn = BMNRegression()
        with pytest.raises(ValueError):
            bmn.fit(repeat_sales_df, GeographyLevel.COUNTY, "06037")
    
    def test_pipeline_with_different_frequencies(self, test_data_dir):
        """Test pipeline with different time frequencies."""
        for frequency in ["monthly", "quarterly"]:
            config = RSAIConfig(
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
                frequency=frequency
            )
            
            ingestion = DataIngestion(config)
            transactions_df = ingestion.load_transactions(test_data_dir / "transactions.parquet")
            repeat_sales_df = ingestion.identify_repeat_sales()
            
            if len(repeat_sales_df) > 20:
                bmn = BMNRegression(frequency=frequency, min_pairs_per_period=5)
                result = bmn.fit(
                    repeat_sales_df.head(50),  # Use subset for speed
                    GeographyLevel.COUNTY,
                    "06037"
                )
                
                assert result.num_periods > 0
                assert len(result.index_values) > 0
                
                # Check that periods match frequency
                periods = [iv.period for iv in result.index_values]
                if frequency == "monthly":
                    assert all(p.day == 1 for p in periods)
                elif frequency == "quarterly":
                    assert all(p.month in [1, 4, 7, 10] for p in periods)
    
    def test_pipeline_with_different_weights(self, test_data_dir):
        """Test pipeline with different weighting schemes."""
        config = RSAIConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31)
        )
        
        ingestion = DataIngestion(config)
        transactions_df = ingestion.load_transactions(test_data_dir / "transactions.parquet")
        repeat_sales_df = ingestion.identify_repeat_sales()
        
        results = {}
        
        for scheme in [WeightingScheme.EQUAL, WeightingScheme.VALUE, WeightingScheme.BMN]:
            weight_calc = WeightCalculator(scheme)
            weighted_df = weight_calc.calculate_weights(repeat_sales_df.head(100))
            
            # Verify weights were applied
            assert "weight" in weighted_df.columns
            assert weighted_df["weight_type"][0] == scheme.value
            
            # Check weight diagnostics
            diagnostics = weight_calc.diagnose_weights(weighted_df)
            results[scheme.value] = diagnostics
        
        # Compare weight distributions
        assert results["equal"]["std"] < results["value"]["std"]  # Equal weights have no variation
    
    def test_end_to_end_performance(self, test_data_dir, tmp_path):
        """Test pipeline performance with larger dataset."""
        import time
        
        config = RSAIConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            frequency="monthly"
        )
        
        start_time = time.time()
        
        # Load data
        ingestion = DataIngestion(config)
        transactions_df = ingestion.load_transactions(test_data_dir / "transactions.parquet")
        properties_df = ingestion.load_properties(test_data_dir / "properties.parquet")
        
        # Process repeat sales
        repeat_sales_df = ingestion.identify_repeat_sales()
        repeat_sales_df = ingestion.merge_geographic_data()
        
        # Calculate weights
        weight_calc = WeightCalculator(WeightingScheme.EQUAL)
        repeat_sales_df = weight_calc.calculate_weights(repeat_sales_df)
        
        # Run regression (subset for speed)
        bmn = BMNRegression(min_pairs_per_period=5)
        if len(repeat_sales_df) > 50:
            result = bmn.fit(
                repeat_sales_df.head(50),
                GeographyLevel.COUNTY,
                "06037"
            )
            
            # Export results
            exporter = OutputExporter(tmp_path / "output", config)
            exporter.export_index_values(result.index_values)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Pipeline should complete in reasonable time
        assert execution_time < 30  # 30 seconds for test data
        
        # Log performance metrics
        print(f"\nPipeline Performance:")
        print(f"- Total transactions: {len(transactions_df)}")
        print(f"- Repeat sales pairs: {len(repeat_sales_df)}")
        print(f"- Execution time: {execution_time:.2f} seconds")