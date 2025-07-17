# -*- coding: utf-8 -*-
"""
Main execution script for the battery performance classification project.

This script orchestrates the entire workflow, including data preprocessing
and model training.

Author: lunazhang
"""
import logging
from pathlib import Path

from src.data_pipeline import run_pipeline
from src.train import main as train_main

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main execution block."""
    raw_data_dir = Path("data/raw")
    processed_data_dir = Path("data/processed")
    raw_data_csv = raw_data_dir / "battery_performance_data.csv"
    processed_data_parquet = processed_data_dir / "battery_data.parquet"
    target_cols = ["target_capacity", "target_resistance"]

    # Ensure data directories exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run preprocessing pipeline if processed data does not exist
    if not processed_data_parquet.exists():
        logging.info("Processed data not found. Running data pipeline...")
        if not raw_data_csv.exists():
            logging.error(
                "Raw data file not found at %s. Please provide the data.", raw_data_csv
            )
            return
        run_pipeline(csv_path=raw_data_csv, parquet_path=processed_data_parquet)
    else:
        logging.info(
            "Processed data found at %s. Skipping pipeline.", processed_data_parquet
        )

    # 2. Run training on processed data
    logging.info("Starting model training process...")
    train_main(data_path=processed_data_parquet, target_cols=target_cols)
    logging.info("Model training and evaluation completed.")


if __name__ == "__main__":
    main()