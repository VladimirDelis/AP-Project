"""
Signal processing model.

Pure numpy/scipy signal transforms, shared by both the live VisPy view and
the offline Matplotlib view. No GUI code (MVVM: Model layer).

Parameters used (document these in the README too):
    RMS window:      50 samples (moving RMS)
    Filter:          4th-order Butterworth band-pass, 1-40 Hz
                     (typical EMG/EEG-like signal band; adjust to your
                     actual signal source and sample rate)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

RMS_WINDOW = 50
FILTER_ORDER = 4
FILTER_LOW_HZ = 1.0
FILTER_HIGH_HZ = 40.0


def compute_rms(signal: np.ndarray, window: int = RMS_WINDOW) -> np.ndarray:
    """Moving RMS along the last axis. signal shape: (..., n_samples)."""
    if signal.shape[-1] == 0:
        return signal.copy()
    window = max(1, min(window, signal.shape[-1]))
    squared = signal ** 2
    kernel = np.ones(window) / window
    if signal.ndim == 1:
        mean_sq = np.convolve(squared, kernel, mode="same")
        return np.sqrt(mean_sq)
    # 2D: (n_channels, n_samples) -> apply per channel
    out = np.empty_like(signal, dtype=float)
    for ch in range(signal.shape[0]):
        out[ch] = np.sqrt(np.convolve(squared[ch], kernel, mode="same"))
    return out


def compute_filtered(
    signal: np.ndarray,
    sample_rate_hz: float,
    low_hz: float = FILTER_LOW_HZ,
    high_hz: float = FILTER_HIGH_HZ,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter along the last axis.

    Falls back to returning the input unchanged if there are too few
    samples for stable filtering (filtfilt needs > ~3x filter order).
    """
    n_samples = signal.shape[-1]
    min_len = 3 * (order * 2 + 1)  # filtfilt padding requirement, roughly
    if n_samples < min_len:
        return signal.copy()

    nyquist = sample_rate_hz / 2.0
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.999)
    if low >= high:
        return signal.copy()

    b, a = butter(order, [low, high], btype="band")

    if signal.ndim == 1:
        return filtfilt(b, a, signal)

    out = np.empty_like(signal, dtype=float)
    for ch in range(signal.shape[0]):
        out[ch] = filtfilt(b, a, signal[ch])
    return out


def apply_mode(signal: np.ndarray, mode: str, sample_rate_hz: float) -> np.ndarray:
    """Dispatch helper: mode in {'original', 'rms', 'filtered'}."""
    if mode == "original":
        return signal
    if mode == "rms":
        return compute_rms(signal)
    if mode == "filtered":
        return compute_filtered(signal, sample_rate_hz)
    raise ValueError(f"Unknown signal mode: {mode!r}")
