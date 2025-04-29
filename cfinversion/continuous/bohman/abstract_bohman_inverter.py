from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from ..continuous_inverter import ContinuousInverter
from ...tools import Standardizer

class AbstractBohmanInverter(ContinuousInverter, ABC):
    """Abstract class for characteristic function inverter,
    which are implemented using the methods described by Harald Bohman in 1975

    Bohmans methods allows to implement only cumulative distribution function
    """
    def __init__(self):
        self.standardizer = Standardizer()

    @staticmethod
    def _C(t: np.ndarray) -> np.ndarray:
        result = np.zeros_like(t)

        t_negative = t[(t >= -1) & (t <= 0)]
        result[(t >= -1) & (t <= 0)] = (1 + t_negative) * np.cos(np.pi * -t_negative) + np.sin(
            np.pi * -t_negative) / np.pi

        t_positive = t[(0 <= t) & (t <= 1)]
        result[(0 <= t) & (t <= 1)] = (1 - t_positive) * np.cos(np.pi * t_positive) + np.sin(np.pi * t_positive) / np.pi

        return result
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Function return cumulative distribution function

        Attributes
        ----------
        x : np.ndarray
            Data for which we want to calculate
            the value of the cumulative distribution function

        Return
        ------
        np.ndarray
            The value of the cumulative distribution function for each element x
        """
        return self.standardizer.unstandardize_cdf(self.cdf_)(x)

    def pdf(self, x: np.ndarray, tol_diff:float = 1e-4) -> np.ndarray:
        """Function return probability density function

        Attributes
        ----------
        x : np.ndarray
            Data for which we want to calculate
            the value of the probability density function

        Return
        ------
        np.ndarray
            The value of the probability density function for each element x
        """
        return (self.cdf(x+tol_diff)-self.cdf(x-tol_diff))/(2 * tol_diff)


    @abstractmethod
    def cdf_(self, x: np.ndarray) -> np.ndarray:
        """Internal variant for standardized cf"""
        pass
