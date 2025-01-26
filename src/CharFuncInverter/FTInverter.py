from typing import NoReturn

import numpy as np

from src.CharFuncInverter.CharFuncInverter import CharFuncInverter


class FFTInverterNaive(CharFuncInverter):

    def __init__(self, N=1e3, delta=1e-1) -> NoReturn:
        super().__init__()
        self.phi = None
        self.N = int(N)
        self.delta = delta
        self.num_points = N / delta

    def fit(self, phi) -> NoReturn:
        self.phi = phi

    def cdf(self, x):
        t = np.linspace(-self.N, self.N, self.num_points)

        phi_t = self.phi(t)

        integral = np.trapezoid((phi_t * np.exp(-1j * t * x)) / (1j * t), t)
        return 1 / 2 - (1 / (2 * np.pi)) * integral

    def pdf(self, x):
        t = np.linspace(-self.N, self.N, self.num_points)

        phi_t = self.phi(t)

        integral = np.trapezoid(phi_t * np.exp(1j * t * x), t)
        return (1 / (2 * np.pi)) * integral
