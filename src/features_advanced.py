# -*- coding: utf-8 -*-
"""
Advanced Feature Engineering for Solid-State Battery Analysis.

This module provides functions for extracting sophisticated, domain-specific
features from raw electrochemical data, such as dQ/dV analysis, EIS fitting,
and degradation rate calculations. These features are crucial for building
high-fidelity predictive models.

Author: lunazhang
"""
import logging
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def extract_dqdv_features(cycle_data: pd.DataFrame, voltage_range=(2.5, 4.2), min_peak_height=0.1) -> dict:
    """
    Extracts features from differential capacity (dQ/dV) curves for a single cycle.

    This analysis reveals phase transitions and is sensitive to degradation mechanisms.

    Args:
        cycle_data: DataFrame with 'voltage' and 'charge_capacity' for one cycle.
        voltage_range: Tuple specifying the voltage window for analysis.
        min_peak_height: Minimum height for a peak to be considered.

    Returns:
        A dictionary of extracted dQ/dV features (e.g., peak count, locations, heights).
        Returns an empty dict if data is insufficient.
    """
    logging.info("Extracting dQ/dV features...")
    required_cols = ['voltage', 'charge_capacity']
    if not all(col in cycle_data.columns for col in required_cols):
        logging.warning("dQ/dV feature extraction requires 'voltage' and 'charge_capacity'.")
        return {}

    cycle_data = cycle_data.dropna(subset=required_cols).sort_values('voltage')
    if cycle_data.empty:
        return {}

    # Interpolate to a uniform voltage grid
    interp_func = interp1d(cycle_data['voltage'], cycle_data['charge_capacity'], kind='linear', bounds_error=False, fill_value=0)
    v_uniform = np.linspace(voltage_range[0], voltage_range[1], num=1000)
    q_uniform = interp_func(v_uniform)

    # Calculate dQ/dV
    dq_dv = np.gradient(q_uniform, v_uniform)

    # Find peaks
    peaks, properties = find_peaks(dq_dv, height=min_peak_height)

    if peaks.size == 0:
        return {"dqdv_peak_count": 0}

    return {
        "dqdv_peak_count": len(peaks),
        "dqdv_peak1_voltage": v_uniform[peaks[0]] if len(peaks) > 0 else np.nan,
        "dqdv_peak1_height": properties["peak_heights"][0] if len(peaks) > 0 else np.nan,
    }


def extract_eis_features(eis_data: pd.DataFrame) -> dict:
    """
    Extracts features from Electrochemical Impedance Spectroscopy (EIS) data.
    
    This is a placeholder for a full equivalent circuit model fitting. In a real
    scenario, this would involve using a library like `impedance.py` to fit a model
    (e.g., Randles circuit) and extract parameters like solution resistance (R_s)
    and charge-transfer resistance (R_ct).

    Args:
        eis_data: DataFrame with 'frequency', 'real_impedance', 'imag_impedance'.

    Returns:
        A dictionary of placeholder EIS features.
    """
    logging.info("Extracting EIS features (placeholder)...")
    required_cols = ['frequency', 'real_impedance', 'imag_impedance']
    if not all(col in eis_data.columns for col in required_cols):
        logging.warning("EIS feature extraction requires 'frequency', 'real_impedance', 'imag_impedance'.")
        return {}

    # Placeholder logic: estimate R_s from high-frequency intercept
    r_s_estimate = eis_data.loc[eis_data['frequency'].idxmax()]['real_impedance']

    return {
        "eis_r_s_estimate": r_s_estimate,
        "eis_r_ct_placeholder": 0.05,  # Placeholder for charge-transfer resistance
    }


def calculate_decay_rate(summary_data: pd.DataFrame) -> dict:
    """
    Calculates the capacity fade rate over a series of cycles.

    Args:
        summary_data: DataFrame indexed by cycle number with a 'capacity' column.

    Returns:
        A dictionary containing the linear decay rate of capacity.
    """
    logging.info("Calculating capacity decay rate...")
    if 'capacity' not in summary_data.columns or not isinstance(summary_data.index, (pd.RangeIndex, pd.Int64Index)):
        logging.warning("Decay calculation requires a 'capacity' column and a numeric cycle index.")
        return {}
    
    # Ensure at least two points to fit a line
    if summary_data.shape[0] < 2:
        return {"capacity_decay_rate": np.nan}

    # Linear fit of capacity vs. cycle number
    coeffs = np.polyfit(summary_data.index, summary_data['capacity'], 1)
    slope = coeffs[0]

    return {
        "capacity_decay_rate": slope
    } 