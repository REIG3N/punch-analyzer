import math

import pandas as pd

from punch_analyzer.smoothing import OneEuroFilter, smooth_series


def test_one_euro_filter_passthrough_on_first_sample():
    filt = OneEuroFilter(mincutoff=1.0, beta=0.0)
    assert filt.filter(0.5, t=0.0) == 0.5


def test_one_euro_filter_smooths_step_less_than_raw_jump():
    filt = OneEuroFilter(mincutoff=1.0, beta=0.0)
    filt.filter(0.0, t=0.0)
    smoothed = filt.filter(1.0, t=1 / 30)

    assert 0.0 < smoothed < 1.0


def test_smooth_series_resets_on_nan_gap():
    values = pd.Series([0.0, 0.0, math.nan, 0.9, 0.9])
    timestamps_ms = pd.Series([0.0, 33.3, 66.6, 100.0, 133.3])

    smoothed = smooth_series(values, timestamps_ms, mincutoff=1.0, beta=0.0)

    assert math.isnan(smoothed.iloc[2])
    # Après le trou, le filtre redémarre à froid : la valeur brute passe telle quelle.
    assert smoothed.iloc[3] == 0.9
