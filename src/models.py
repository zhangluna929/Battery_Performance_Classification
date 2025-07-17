# -*- coding: utf-8 -*-
"""
Machine learning model definitions, training, and evaluation.

This module contains the core components for building, tuning, and evaluating
the classification models used in this project. It supports RandomForest and
XGBoost classifiers.

Author: lunazhang
"""
import logging
from typing import Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- Model Builders ---

def build_random_forest(
    n_estimators: int = 100, random_state: int = 42, **kwargs
) -> RandomForestClassifier:
    """Builds a RandomForestClassifier with specified parameters."""
    return RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, **kwargs
    )


def build_xgboost(
    n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 5, **kwargs
) -> xgb.XGBClassifier:
    """Builds an XGBClassifier with specified parameters."""
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        use_label_encoder=False,
        eval_metric="mlogloss",
        **kwargs
    )


# --- Hyperparameter Tuning ---

def tune_model_optuna(
    model_type: str, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 50
) -> Tuple[any, dict]:
    """
    Hyperparameter tuning using Optuna for a specified model type.

    Args:
        model_type: Either 'rf' for RandomForest or 'xgb' for XGBoost.
        X_train: Training features.
        y_train: Training target.
        n_trials: Number of optimization trials.

    Returns:
        A tuple of the best fitted model and its parameters.
    """
    logging.info("Tuning %s with Optuna...", model_type)

    def objective(trial):
        if model_type == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
            model = build_random_forest(**params)
        elif model_type == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 1e-1, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
            model = build_xgboost(**params)
        else:
            raise ValueError("Unsupported model type for tuning.")

        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        return accuracy_score(y_train, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    logging.info("Best trial for %s: %s", model_type, best_params)

    if model_type == "rf":
        best_model = build_random_forest(**best_params)
    else: # xgb
        best_model = build_xgboost(**best_params)

    best_model.fit(X_train, y_train)
    return best_model, best_params


# --- Evaluation ---

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float]:
    """Evaluates the model and returns accuracy and F1 score."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    logging.info(
        "%s - Accuracy: %.4f | F1-score (weighted): %.4f",
        model.__class__.__name__,
        accuracy,
        f1,
    )
    return float(accuracy), float(f1)


def cross_validate(model, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> np.ndarray:
    """Performs cross-validation and returns accuracy scores."""
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    logging.info("Cross-validation accuracy scores: %s", scores)
    logging.info("Mean CV accuracy: %.4f (+/- %.4f)", scores.mean(), scores.std() * 2)
    return scores 