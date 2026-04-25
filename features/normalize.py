"""
features/normalize.py — Per-window z-score normalization.
==========================================================
The STRICTEST causal normalization: each 45-day window is
normalized using ONLY the 45 values within that window.
No information from outside the window contaminates statistics.

This automatically adapts to volatility regimes without any
global parameters.
"""

import numpy as np


def normalize_window_zscore(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalize a single window independently per feature.

    Args:
        window: 2D array of shape (timesteps, n_features).
        eps: Small constant to prevent division by zero.

    Returns:
        Normalized array of same shape.
    """
    mean = np.nanmean(window, axis=0, keepdims=True)
    std = np.nanstd(window, axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)  # Prevent div-by-zero for constant features
    normalized = (window - mean) / std
    # Replace any remaining NaN/inf with 0
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized


def normalize_fundamentals_train_fit(fund_array: np.ndarray, eps: float = 1e-8):
    """
    Compute mean and std from training set fundamentals.

    Returns:
        (mean, std) arrays for use in normalize_fundamentals_apply.
    """
    mean = np.nanmean(fund_array, axis=0)
    std = np.nanstd(fund_array, axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean, std


def normalize_fundamentals_apply(
    fund_array: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Apply training-set normalization to fundamental features.
    """
    normalized = (fund_array - mean) / std
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
