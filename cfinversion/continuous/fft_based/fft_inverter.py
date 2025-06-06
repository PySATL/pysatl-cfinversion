from typing import Callable, Union
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from ..continuous_inverter import ContinuousInverter


class FFTInverter(ContinuousInverter):

    def __init__(self, N: float = 2 ** 8, A: float = -6, B: float = 6) -> None:
        super().__init__()
        self.N: int = int(N)
        self.A: float = A
        self.B: float = B
        self.dist = B - A
        self.dy = self.dist / N
        self.dt = (2 * np.pi) / self.dist
        self.T = (N / 2) * self.dt
        self.k = np.arange(N)
        self.j = np.arange(N)
        self.Y = A + self.k * self.dy
        self.t = -self.T + self.j * self.dt
        self.phi = None #type: Callable[[np.ndarray], np.ndarray] | None

    def fit(self, cf: Callable) -> None:
        """cf = characteristic function"""
        self.cf = cf

        f = np.exp(-1j * self.j * self.dt * self.A) * self.cf(self.t)
        C = (self.dt / (2 * np.pi)) * np.exp(1j * self.T * self.Y)

        self.pdf_values = np.real(C * np.fft.fft(f)) #type: np.ndarray
        self.pdf_interp = interp1d(self.Y, self.pdf_values, kind='linear')

        self.cdf_values = np.cumsum(self.pdf_values) * self.dy #type: np.ndarray
        self.cdf_interp = interp1d(self.Y, self.cdf_values, kind='linear') 

    def cdf(self, x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        if self.cf is None:
            raise ValueError("Characteristic function (phi) is not set. Call fit() first.")
        result = np.real(self.cdf_interp(x))
        if isinstance(x, float):
            return result.item()
        return result

    def pdf(self, x: float | NDArray[np.float64]) ->  float | NDArray[np.float64]:
        if self.cf is None:
            raise ValueError("Characteristic function (phi) is not set. Call fit() first.")
        result = np.real(self.pdf_interp(x))
        if isinstance(x, float):
            return result.item()
        return result

