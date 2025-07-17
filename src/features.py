# -*- coding: utf-8 -*-
"""
Feature selection module.

This module provides utilities for selecting the most relevant features
from the dataset to improve model performance and interpretability.

Author: lunazhang
"""
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression


def select_features(
    X: pd.DataFrame, y: pd.Series, k: int = 10
) -> Tuple[pd.DataFrame, List[str], SelectKBest]:
    """
    Selects the top-k features using the F-regression score.

    Args:
        X: DataFrame of features.
        y: Series of target values.
        k: The number of top features to select.

    Returns:
        A tuple containing:
        - X_selected: DataFrame with only the selected features.
        - selected_features: List of names of the selected features.
        - selector: The fitted SelectKBest instance.
    """
    logging.info("Performing feature selection to find the top %d features...", k)
    
    # Handle multi-output targets by averaging them for scoring
    if y.ndim > 1 and y.shape[1] > 1:
        y_for_scoring = y.mean(axis=1)
        logging.info("Target has multiple columns; using their mean for feature scoring.")
    else:
        y_for_scoring = y

    selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    X_selected_array = selector.fit_transform(X, y_for_scoring)
    
    selected_feature_names = list(X.columns[selector.get_support(indices=True)])
    
    X_selected = pd.DataFrame(X_selected_array, columns=selected_feature_names)

    logging.info("Selected features: %s", selected_feature_names)
    
    return X_selected, selected_feature_names, selector 