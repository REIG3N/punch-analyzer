import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def savgol_smooth_run(values: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """Lisse un run continu (sans trou) de valeurs. Réduit la fenêtre/l'ordre pour un
    run trop court plutôt que de planter ; laisse passer tel quel si trop court pour
    qu'un polynôme ait un sens (< 3 points)."""
    n = len(values)
    effective_window = min(window_length, n if n % 2 == 1 else n - 1)
    if effective_window < 3:
        return values
    effective_polyorder = min(polyorder, effective_window - 1)
    return savgol_filter(values, effective_window, effective_polyorder)


def smooth_series(
    series: pd.Series,
    window_length: int,
    polyorder: int,
) -> pd.Series:
    """Lissage non causal (Savitzky-Golay) sur toute la série, run par run de valeurs
    non-NaN contiguës (les runs sont déjà issus de _fill_short_gaps en amont : seuls
    les vrais trous longs restent NaN et ne sont pas pontés par le lissage)."""
    is_na = series.isna()
    run_id = (is_na != is_na.shift()).cumsum()
    result = series.copy()
    for _, group_na in is_na.groupby(run_id):
        if bool(group_na.iloc[0]):
            continue
        idx = group_na.index
        result.loc[idx] = savgol_smooth_run(series.loc[idx].to_numpy(), window_length, polyorder)
    return result
