#!/usr/bin/env python3
"""
Batch processing example for multiple datasets.

This script demonstrates how to process multiple datasets
in batch with consistent configuration and reporting.
"""

import pandas as pd
from pathlib import Path
from hpi_fhfa.processing.pipeline import HPIPipeline
from hpi_fhfa.config.settings import HPIConfig
from hpi_fhfa.validation import HPIValidator

def process_batch_datasets(data_directory: Path, output_base: Path):
    """Process multiple datasets in batch."""
    
    # Find all transaction files
    transaction_files = list(data_directory.glob("transactions_*.parquet"))
    geographic_file = data_directory / "geographic.parquet"
    
    if not geographic_file.exists():
        raise FileNotFoundError(f"Geographic file not found: {geographic_file}")
    
    print(f"Found {len(transaction_files)} datasets to process")
    
    results_summary = []
    
    for txn_file in transaction_files:
        dataset_name = txn_file.stem
        print(f"\nProcessing {dataset_name}...")
        
        try:
            # Configure pipeline
            config = HPIConfig(
                transaction_data_path=txn_file,
                geographic_data_path=geographic_file,
                output_path=output_base / dataset_name,
                start_year=2020,
                end_year=2023,
                weight_schemes=["sample", "value"],
                n_jobs=4,
                validate_data=True
            )
            
            # Run pipeline
            pipeline = HPIPipeline(config)
            results = pipeline.run()
            
            # Validate results
            validator = HPIValidator(tolerance=0.001)
            validation_results = validator.validate_all(
                results.tract_indices,
                results.city_indices
            )
            
            # Collect summary
            passed_validations = sum(1 for r in validation_results if r.passed)
            total_validations = len(validation_results)
            
            summary = {
                "dataset": dataset_name,
                "status": "success",
                "n_transactions": results.metadata["n_transactions"],
                "n_tract_indices": len(results.tract_indices),
                "n_city_indices": sum(len(df) for df in results.city_indices.values()),
                "processing_time": results.metadata["processing_time"],
                "validation_pass_rate": passed_validations / total_validations if total_validations > 0 else 0,
                "avg_tract_appreciation": None,
                "avg_city_appreciation": None
            }
            
            # Calculate average appreciations
            if not results.tract_indices.is_empty():
                tract_appreciation = results.tract_indices.filter(
                    pl.col("appreciation_rate").is_not_null()
                )["appreciation_rate"]
                if len(tract_appreciation) > 0:
                    summary["avg_tract_appreciation"] = tract_appreciation.mean()
            
            if results.city_indices:
                city_appreciations = []
                for city_df in results.city_indices.values():
                    if not city_df.is_empty():
                        city_appreciation = city_df.filter(
                            pl.col("appreciation_rate").is_not_null()
                        )["appreciation_rate"]
                        if len(city_appreciation) > 0:
                            city_appreciations.extend(city_appreciation.to_list())
                
                if city_appreciations:
                    summary["avg_city_appreciation"] = sum(city_appreciations) / len(city_appreciations)
            
            results_summary.append(summary)
            
            print(f"  ✓ Success: {results.metadata['n_transactions']:,} transactions, "
                  f"{results.metadata['processing_time']:.1f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results_summary.append({
                "dataset": dataset_name,
                "status": "failed",
                "error": str(e),
                "n_transactions": None,
                "processing_time": None
            })
    
    # Create summary report
    summary_df = pd.DataFrame(results_summary)
    
    # Save summary
    summary_file = output_base / "batch_processing_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 50)
    
    successful = summary_df[summary_df["status"] == "success"]
    failed = summary_df[summary_df["status"] == "failed"]
    
    print(f"Total datasets: {len(summary_df)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if len(successful) > 0:
        total_transactions = successful["n_transactions"].sum()
        total_time = successful["processing_time"].sum()
        avg_throughput = total_transactions / total_time if total_time > 0 else 0
        
        print(f"\nSuccessful Processing:")
        print(f"  Total transactions: {total_transactions:,}")
        print(f"  Total processing time: {total_time:.1f}s")
        print(f"  Average throughput: {avg_throughput:.0f} txn/s")
        
        avg_tract_app = successful["avg_tract_appreciation"].mean()
        avg_city_app = successful["avg_city_appreciation"].mean()
        
        if not pd.isna(avg_tract_app):
            print(f"  Average tract appreciation: {avg_tract_app:.2f}%")
        if not pd.isna(avg_city_app):
            print(f"  Average city appreciation: {avg_city_app:.2f}%")
    
    if len(failed) > 0:
        print(f"\nFailed Datasets:")
        for _, row in failed.iterrows():
            print(f"  {row['dataset']}: {row['error']}")
    
    print(f"\nSummary saved to: {summary_file}")
    return summary_df

def main():
    data_dir = Path("data/batch_datasets/")
    output_dir = Path("output/batch_results/")
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please create the directory and add transaction files named 'transactions_*.parquet'")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = process_batch_datasets(data_dir, output_dir)

if __name__ == "__main__":
    main()
