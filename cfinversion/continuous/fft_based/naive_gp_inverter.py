from typing import Callable, Optional, NoReturn, Union
import numpy as np
from numpy.typing import NDArray
from ..continuous_inverter import ContinuousInverter


class NaiveGPInverter(ContinuousInverter):
    def __init__(self, N: float = 1e3, delta: float = 1e-1, num_points: int | None = None) -> None:
        super().__init__()
        self.N: int = int(N)
        self.delta: float = delta
        self.num_points: int = int(N / delta) if num_points is None else num_points

    def fit(self, cf) -> None:
        """cf = characteristic function"""
        self.cf = cf
        self.t = np.linspace(-self.N, self.N, self.num_points)
        self.cf_values = self.cf(self.t)
    def cdf(self, x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        if self.cf is None:
            raise ValueError("Characteristic function (phi) is not set. Call fit() first.")
        x_arr = np.asarray(x)
        t = self.t
        tq = self.cf_values * np.exp(-1j * t * x_arr[:, np.newaxis])
        tq[:, t != 0] /= (1j * t[t != 0])
        tq[:, t == 0] = -x_arr.reshape(-1,1)
        result =  1 / 2 - (1 / (2 * np.pi)) * np.real(np.trapezoid(tq, t, axis=1))
        if isinstance(x, float):
            return result.item()
        return result
    def pdf(self, x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        if self.cf is None:
            raise ValueError("Characteristic function (phi) is not set. Call fit() first.")
        x_arr = np.asarray(x)
        t = self.t
        result = (1.0 / (2 * np. pi)) * np.real(np.trapezoid(
            self.cf_values * np.exp(-1j * t * x_arr[:, np.newaxis]), t, axis=1
        ))
        if isinstance(x, float):
            return result.item()
        return result
