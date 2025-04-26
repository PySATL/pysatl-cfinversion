from typing import Callable

import numpy as np


class Standardizer:

    def __init__(self, m: float, sd: float) -> None:
        """
        the class makes transitions between random variable and standardized random variable

        :param m: mean
        :param sd: standard deviation
        """
        self.m = m
        self.sd = sd

    def standardize_chf(self, phi: Callable) -> Callable:
        """
        :param phi: characteristic function
        :return: characteristic function of standardized
                 random variable
        """
        z_phi = lambda args: np.exp(-1j * args * self.m / self.sd) * phi(args / self.sd)
        return z_phi

    def unstandardize_chf(self, z_phi: Callable) -> Callable:
        """
        :param z_phi: characteristic function of standardized
        :return: characteristic function
random variable
        """
        phi = lambda args: np.exp(1j * args * self.m) * z_phi(args * self.sd)
        return phi

    def standardize_cdf(self, F: Callable) -> Callable:
        """
        Returns the distribution function of the standard random variable Z.

        :param F: distribution of the original random variable X.
        :return: distribution function of the standardized random variable Z.
        """
        z_F = lambda args : F(self.m + self.sd * args)
        return z_F

    def unstandardize_cdf(self, z_F: Callable) -> Callable:
        """
        Returns the distribution function of the original random variable Z.

        :param z_F: distribution function of the standardized random variable Z.
        :return: distribution of the original random variable X.
        """
        F = lambda args : z_F((args - self.m) / self.sd)
        return F







