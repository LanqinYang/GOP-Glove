"""Shared feature extraction helpers."""

import numpy as np


def _ricker_wavelet(points: int, width: float) -> np.ndarray:
    """Return a Ricker wavelet compatible with the former scipy.signal.ricker."""
    points = int(points)
    if points <= 0:
        return np.zeros(0, dtype=np.float64)

    width = float(width)
    xs = np.arange(points, dtype=np.float64) - (points - 1.0) / 2.0
    width_sq = width * width
    amplitude = 2.0 / (np.sqrt(3.0 * width) * np.power(np.pi, 0.25))
    wavelet = amplitude * (1.0 - xs * xs / width_sq) * np.exp(-(xs * xs) / (2.0 * width_sq))
    return wavelet


def extract_wavelet_energies(channel_data, scales=np.arange(1, 9)) -> list:
    """Compute 8 scale-wise Ricker-CWT energies without relying on removed SciPy APIs."""
    data = np.asarray(channel_data, dtype=np.float64)
    if data.size == 0:
        return [0.0] * len(scales)

    energies = []
    for scale in scales:
        wavelet_len = min(int(10 * scale), data.size)
        wavelet = _ricker_wavelet(wavelet_len, float(scale))
        coeff = np.convolve(data, wavelet[::-1], mode="same")
        energy = float(np.sum(coeff * coeff))
        if np.isnan(energy) or np.isinf(energy):
            energy = 0.0
        energies.append(energy)

    return energies
