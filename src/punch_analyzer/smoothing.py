import math

import pandas as pd


class OneEuroFilter:
    """Casiez et al. 2012. Alpha s'adapte à la vitesse du signal filtré."""

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev: float | None = None
        self.dx_prev = 0.0
        self.t_prev: float | None = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x: float, t: float) -> float:
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            self.dx_prev = 0.0
            return x

        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.t_prev = t
        self.dx_prev = dx_hat
        return x_hat


def smooth_series(
    values: pd.Series,
    timestamps_ms: pd.Series,
    mincutoff: float = 1.0,
    beta: float = 0.0,
    dcutoff: float = 1.0,
) -> pd.Series:
    """Filtre One Euro appliqué en série. Réinitialisé à chaque NaN (pas de pont sur les trous)."""
    filt = OneEuroFilter(mincutoff, beta, dcutoff)
    out = []
    for value, t_ms in zip(values, timestamps_ms):
        if pd.isna(value):
            filt.reset()
            out.append(math.nan)
        else:
            out.append(filt.filter(float(value), t_ms / 1000.0))
    return pd.Series(out, index=values.index)
