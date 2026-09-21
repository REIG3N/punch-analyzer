import math

import numpy as np
import pandas as pd

from punch_analyzer.smoothing import savgol_smooth_run, smooth_series


def test_savgol_smooth_run_preserves_linear_signal():
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    smoothed = savgol_smooth_run(values, window_length=5, polyorder=2)

    assert np.allclose(smoothed, values)


def test_savgol_smooth_run_reduces_noise_around_a_ramp():
    base = np.linspace(0.0, 1.0, 21)
    rng = np.random.default_rng(0)
    noisy = base + rng.normal(0, 0.05, size=base.shape)

    smoothed = savgol_smooth_run(noisy, window_length=9, polyorder=2)

    assert np.abs(smoothed - base).mean() < np.abs(noisy - base).mean()


def test_savgol_smooth_run_falls_back_on_too_short_runs():
    values = np.array([1.0, 2.0])

    smoothed = savgol_smooth_run(values, window_length=9, polyorder=3)

    assert np.array_equal(smoothed, values)


def test_smooth_series_resets_on_nan_gap():
    series = pd.Series([0.0, 0.0, math.nan, 0.9, 0.9, 0.9, 0.9])

    smoothed = smooth_series(series, window_length=3, polyorder=1)

    assert math.isnan(smoothed.iloc[2])
    assert smoothed.notna().sum() == 6


def test_smooth_series_is_non_causal_symmetric_around_a_spike():
    # Un pic isolé entouré de plat : un lissage non causal (contrairement à un
    # filtre causal type One Euro) doit l'atténuer symétriquement, pas juste
    # après coup.
    series = pd.Series([0.0] * 5 + [1.0] + [0.0] * 5)

    smoothed = smooth_series(series, window_length=5, polyorder=2)

    before = smoothed.iloc[4]
    after = smoothed.iloc[6]
    assert before == after
