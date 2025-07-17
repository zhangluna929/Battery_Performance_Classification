# -*- coding: utf-8 -*-
"""
Data processing pipeline for battery performance data.

This module handles reading, cleaning, and serializing the raw battery data.

Author: lunazhang
"""
import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd
from scipy import stats


def read_csv(input_path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads raw battery data from a CSV file.

    Args:
        input_path: Path to the input CSV file.

    Returns:
        A pandas DataFrame containing the raw data.
    """
    input_path = Path(input_path)
    logging.info("Reading raw CSV data from %s", input_path)
    df = pd.read_csv(input_path)
    logging.info("Loaded raw data with shape: %s", df.shape)
    return df


def clean_data(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Cleans the DataFrame by removing duplicates, missing values, and outliers.

    Outliers are identified using the z-score method for all numeric columns.

    Args:
        df: The input DataFrame to clean.
        z_thresh: The absolute z-score threshold for outlier removal.

    Returns:
        A cleaned pandas DataFrame.
    """
    logging.info("Cleaning data...")
    initial_rows = df.shape[0]

    # Drop duplicates and missing values
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    # Outlier detection and removal for numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if not numeric_cols.empty:
        z_scores = stats.zscore(df[numeric_cols], nan_policy="omit")
        abs_z_scores = abs(z_scores)
        filtered_entries = (abs_z_scores < z_thresh).all(axis=1)
        df = df[filtered_entries]

    final_rows = df.shape[0]
    logging.info(
        "Removed %d rows during cleaning. Final shape: %s",
        initial_rows - final_rows,
        df.shape,
    )
    return df.reset_index(drop=True)


def to_parquet(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    partition_cols: Optional[Sequence[str]] = None,
):
    """
    Saves a DataFrame to a Parquet file.

    Args:
        df: The DataFrame to save.
        output_path: The path for the output Parquet file.
        partition_cols: Optional list of columns to partition by.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing cleaned data to Parquet at %s", output_path)
    df.to_parquet(output_path, engine="pyarrow", index=False, partition_cols=partition_cols)


def run_pipeline(
    csv_path: Union[str, Path],
    parquet_path: Union[str, Path],
    partition_cols: Optional[Sequence[str]] = None,
):
    """
    Executes the full data processing pipeline.

    Args:
        csv_path: Path to the input CSV file.
        parquet_path: Path for the output Parquet file.
        partition_cols: Optional list of columns to partition by.
    """
    df_raw = read_csv(csv_path)
    df_clean = clean_data(df_raw)
    to_parquet(df_clean, parquet_path, partition_cols=partition_cols) 