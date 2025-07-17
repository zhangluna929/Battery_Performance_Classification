# -*- coding: utf-8 -*-
"""
Main training script for the battery performance classification model.

This script orchestrates the model training and evaluation pipeline, including
data loading, feature selection, model tuning, and final evaluation.

Author: lunazhang
"""
import logging
from pathlib import Path
from typing import Sequence, Union

from sklearn.model_selection import train_test_split

from .data import load_and_preprocess_data
from .features import select_features
from .models import build_random_forest, build_xgboost, evaluate, tune_model_optuna


def main(data_path: Union[str, Path], target_cols: Sequence[str]):
    """
    Executes the full training and evaluation pipeline.

    Args:
        data_path: Path to the processed Parquet data file.
        target_cols: Sequence of target column names.
    """
    # 1. Load & preprocess data
    # For classification, we'll use the first target column as the label.
    X, y, _, _ = load_and_preprocess_data(data_path, target_cols=target_cols)
    
    # Assuming the first target column is the primary label for classification
    if y.ndim > 1:
        main_target_col = y.columns[0]
        y_label = y[main_target_col]
        logging.info("Using '%s' as the primary target for classification.", main_target_col)
    else:
        y_label = y

    # 2. Feature selection
    X_selected, _, _ = select_features(X, y_label, k=min(20, X.shape[1]))

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_label, test_size=0.25, random_state=42, stratify=y_label if y_label.nunique() > 1 else None
    )

    # 4. Tune and Train Models
    logging.info("--- Tuning and Training RandomForest ---")
    rf_best, _ = tune_model_optuna("rf", X_train, y_train, n_trials=50)
    
    logging.info("--- Tuning and Training XGBoost ---")
    xgb_best, _ = tune_model_optuna("xgb", X_train, y_train, n_trials=50)
    
    # 5. Evaluate final models on the test set
    logging.info("--- Evaluating Final Models on Test Set ---")
    evaluate(rf_best, X_test, y_test)
    evaluate(xgb_best, X_test, y_test)

    logging.info("Training and evaluation pipeline completed successfully.") 