from typing import Callable, Union
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from ..continuous_inverter import ContinuousInverter
from ...tools import Standardizer

class FFTInverter(ContinuousInverter):

    def __init__(self, N: float = 2 ** 8, A: float = -6, B: float = 6) -> None:
        super().__init__()
        self.N: int = int(N)
        self.A: float = A
        self.B: float = B
        self.D: float = B - A
        self.dy = self.D / N
        self.du = 1 / self.D
        self.dt = 2 * np.pi * self.du
        self.T = N * self.du / 2

        self.j = np.arange(N)
        self.u =  -self.T + self.j * self.du
        self.t = 2 * np.pi * self.u

        self.k = np.arange(N)
        self.y = self.A + self.k * self.dy
        
        self.stdr = Standardizer()

    def fit(self, cf: Callable) -> None:
        """cf = characteristic function"""
        self.stdr.fit(cf)
        self.cf = self.stdr.cf
        
        tmask = (self.t != 0)
        C = np.exp(2 * np.pi * 1.0j * self.T * self.y) * self.du
        
        f_pdf = np.exp(-1j * self.j * self.dt * self.A) * self.cf(self.t)
        
        f_cdf = np.zeros_like(f_pdf)
        f_cdf[tmask]   = np.exp(-1j * self.j[tmask] * self.dt * self.A) * self.cf(self.t[tmask]) / self.t[tmask]
        f_cdf[~tmask]  = 0
        
        self.pdf_values = np.real(C * np.fft.fft(f_pdf)) #type: np.ndarray
        self.pdf_interp = self.stdr.unstandardize_pdf(interp1d(self.y, self.pdf_values, kind='linear', fill_value = 0, bounds_error=False))

        self.cdf_values = 0.5 - np.imag(C * np.fft.fft(f_cdf)) #type: np.ndarray
        self.cdf_values += self.y * self.du # I have absolutely no idea why this works, but it works though
        self.cdf_interp = self.stdr.unstandardize_cdf(interp1d(self.y, self.cdf_values, kind='linear', fill_value = (0,1), bounds_error=False))

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

