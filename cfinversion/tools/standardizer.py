from typing import Callable

import numpy as np


class Standardizer:

    def __init__(self, loc: float | None = None, scale: float | None = None) -> None:
        """
        the class makes transitions between random variable and standardized random variable

        :param loc: location
        :param scale: scale
        """
        self.loc = loc or 0.0
        self.scale = scale or 1.0

    def fit(self, cf: Callable, tol_diff : float = 1e-4) -> None: 
        self.cf_true = cf
        cft = [cf(i * tol_diff) for i in range(1,5)]
        mean = (-3 * np.imag(cft[3]) + 32 * np.imag(cft[2]) - 168 * np.imag(cft[1]) + 672 * np.imag(cft[0])) / (420 * tol_diff) # estimated mean
        var  = (9 * np.real(cft[3]) - 128 * np.real(cft[2]) + 1008 * np.real(cft[1]) - 8064 * np.real(cft[0]) + 7175) / (2520 * tol_diff**2) # estimated variance
        self.loc = mean
        self.scale = np.sqrt(var - mean**2)

    def cf(self, x): 
        """
        :return: characteristic function of standardized
                 random variable at point x
        """
        return np.exp(-1j * x * self.loc / self.scale) * self.cf_true(x / self.scale)

    def standardize_cdf(self, F: Callable) -> Callable:
        """
        Returns the distribution function of the standard random variable Z.

        :param F: distribution of the original random variable X.
        :return: distribution function of the standardized random variable Z.
        """
        z_F = lambda x : F(x * self.scale + self.loc)
        return z_F

    def unstandardize_cdf(self, z_F: Callable) -> Callable:
        """
        Returns the distribution function of the original random variable Z.

        :param z_F: distribution function of the standardized random variable Z.
        :return: distribution of the original random variable X.
        """
        F = lambda x : z_F((x - self.loc) / self.scale)
        return F

    def standardize_pdf(self, f: Callable) -> Callable:
        z_f = lambda x: self.scale * f(self.loc + x * self.scale)
        return z_f

    def unstandardize_pdf(self, z_f: Callable) -> Callable:
        f = lambda x : 1 / self.scale * z_f((x - self.loc) / self.scale)
        return f





