from typing import NoReturn

import numpy as np

from cfinvert.CharFuncInverter.CharFuncInverter import CharFuncInverter


class FTInverterNaive(CharFuncInverter):

    def __init__(self, N=1e3, delta=1e-1, num_points=None):
        super().__init__()
        self.N = int(N)
        self.delta = delta
        if num_points is None:
            self.num_points = int(N // delta)
        else:
            self.num_points = num_points

    def fit(self, phi):
        """phi = characteristic function"""
        self.phi = phi

    def cdf(self, x):
        t = np.linspace(-self.N, self.N, self.num_points)

        phi_t = self.phi(t)

        integral = np.trapezoid((phi_t * np.exp(-1j * t * x[:, np.newaxis])) / (1j * t), t, axis=1)
        return 1 / 2 - (1 / (2 * np.pi)) * integral

    def pdf(self, x):
        t = np.linspace(-self.N, self.N, self.num_points)

        phi_t = self.phi(t)

        integral = np.trapezoid(phi_t * np.exp(-1j * t * x[:, np.newaxis]), t, axis=1)

        return (1 / (2 * np.pi)) * integral