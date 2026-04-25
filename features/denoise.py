"""
features/denoise.py — Wavelet denoising for price/technical features.
=====================================================================
Applies Discrete Wavelet Transform (DWT) with soft thresholding
to remove high-frequency noise while preserving trend, momentum,
and volatility structure.

Based on the VisuShrink universal threshold (Donoho & Johnstone, 1994):
    λ = σ̂ √(2 ln N)

where σ̂ is estimated from the finest-scale wavelet coefficients
using the Median Absolute Deviation (MAD).
"""

import numpy as np
import pandas as pd
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import WAVELET_CONFIG
from utils import setup_logger

logger = setup_logger(__name__)


def wavelet_denoise_signal(
    signal: np.ndarray,
    wavelet: str = None,
    level: int = None,
    mode: str = None,
) -> np.ndarray:
    """
    Denoise a 1D signal using DWT with soft thresholding.

    Args:
        signal: 1D numpy array.
        wavelet: Wavelet name (default: 'db4').
        level: Decomposition level (default: 3).
        mode: Thresholding mode ('soft' or 'hard').

    Returns:
        Denoised signal of same length.
    """
    try:
        import pywt
    except ImportError:
        logger.warning("pywt not installed. Returning original signal.")
        return signal

    if wavelet is None:
        wavelet = WAVELET_CONFIG["wavelet"]
    if level is None:
        level = WAVELET_CONFIG["level"]
    if mode is None:
        mode = WAVELET_CONFIG["mode"]

    # Handle short signals or constant signals
    if len(signal) < 2 ** level:
        return signal
    if np.std(signal) < 1e-10:
        return signal

    # Auto-clamp level to the maximum safe value for this signal length.
    # Prevents boundary effects that corrupt all coefficients when the
    # window is shorter than the wavelet's support at the requested level.
    max_safe_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    if level > max_safe_level:
        level = max(max_safe_level, 1)

    # Decompose
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Estimate noise level from finest detail coefficients (MAD estimator)
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745

    # Universal threshold (VisuShrink)
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Apply soft thresholding to detail coefficients only
    # Keep approximation coefficients (low-frequency trend) untouched
    denoised_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail, threshold, mode=mode))

    # Reconstruct
    denoised = pywt.waverec(denoised_coeffs, wavelet)

    # pywt may return signal 1 element longer due to padding
    return denoised[: len(signal)]


def denoise_feature_matrix(
    features: np.ndarray,
) -> np.ndarray:
    """
    Denoise each feature column in a (timesteps, n_features) matrix.

    Applied per-window: each 45-day window's features are denoised independently.

    Args:
        features: 2D array of shape (timesteps, n_features).

    Returns:
        Denoised array of same shape.
    """
    denoised = np.copy(features)
    n_timesteps, n_features = features.shape

    for j in range(n_features):
        col = features[:, j]
        if np.all(np.isnan(col)) or np.std(col) < 1e-10:
            continue
        # Fill NaN for wavelet (restore after)
        mask = np.isnan(col)
        if mask.any():
            col_filled = pd.Series(col).ffill().bfill().values
        else:
            col_filled = col
        denoised[:, j] = wavelet_denoise_signal(col_filled)
        # Restore NaN positions
        if mask.any():
            denoised[mask, j] = np.nan

    return denoised


if __name__ == "__main__":
    # Test with synthetic noisy sine wave
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 60)
    clean = np.sin(t)
    noisy = clean + np.random.normal(0, 0.3, 60)
    denoised = wavelet_denoise_signal(noisy)

    print(f"Original noise std: {np.std(noisy - clean):.4f}")
    print(f"Denoised noise std: {np.std(denoised - clean):.4f}")
    print(f"Noise reduction: {1 - np.std(denoised - clean)/np.std(noisy - clean):.1%}")

    # Test matrix denoising
    mat = np.column_stack([noisy, noisy * 2 + 1, np.random.randn(60)])
    mat_denoised = denoise_feature_matrix(mat)
    print(f"\nMatrix denoising: ({mat.shape}) → ({mat_denoised.shape})")
    print(f"Feature 0 noise reduction: {1 - np.std(mat_denoised[:,0] - clean)/np.std(mat[:,0] - clean):.1%}")
