from typing import Callable, Union
import numpy as np
from scipy.interpolate import interp1d

from cfinversion.continuous import ContinuousInverter


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
        self.phi = None
        self.pdf_values = None

    def fit(self, phi: Callable[[np.ndarray], np.ndarray]) -> None:
        """phi = characteristic function"""
        self.phi = phi

        f = np.exp(-1j * self.j * self.dt * self.A) * self.phi(self.t)
        C = (self.dt / (2 * np.pi)) * np.exp(1j * self.T * self.Y)

        self.pdf_values = np.real(C * np.fft.fft(f))
        self.pdf_interp = interp1d(self.Y, self.pdf_values, kind='linear')

        self.cdf_values = np.cumsum(self.pdf_values) * self.dy
        self.cdf_interp = interp1d(self.Y, self.cdf_values, kind='linear')

    def cdf(self, x: np.ndarray) -> Union[float, np.ndarray]:
        if self.phi is None:
            raise ValueError("Characteristic function (phi) is not set. Call fit() first.")
        return self.cdf_interp(x)

    def pdf(self, x: np.ndarray) -> Union[float, np.ndarray]:
        if self.phi is None:
            raise ValueError("Characteristic function (phi) is not set. Call fit() first.")

        return self.pdf_interp(x)
