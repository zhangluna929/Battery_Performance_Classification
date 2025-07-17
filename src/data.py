# -*- coding: utf-8 -*-
"""
Data loading and preprocessing for model training.

This module provides utilities to load the processed data and prepare it
for the machine learning model.

Author: lunazhang
"""
import logging
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(
    filepath: Union[str, Path], target_cols: Sequence[str]
) -> Tuple[pd.DataFrame, pd.Series, StandardScaler, List[str]]:
    """
    Loads Parquet data, separates features and targets, and scales features.

    Args:
        filepath: Path to the Parquet data file.
        target_cols: A sequence of strings with the names of the target columns.

    Returns:
        A tuple containing:
        - X_scaled: A DataFrame of scaled features.
        - y: A Series or DataFrame of target values.
        - scaler: The fitted StandardScaler instance.
        - feature_names: A list of the feature column names.
    """
    fp = Path(filepath)
    logging.info("Loading processed data from %s", fp)

    if fp.suffix != ".parquet":
        raise ValueError("The data file must be in Parquet format.")

    df = pd.read_parquet(fp)
    logging.info("Data shape before preprocessing: %s", df.shape)

    # Drop rows with any missing values
    df = df.dropna().reset_index(drop=True)
    logging.info("Data shape after dropping NA: %s", df.shape)

    y = df[list(target_cols)]
    X = df.drop(columns=list(target_cols))

    feature_names = list(X.columns)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    return X_scaled, y, scaler, feature_names 