from abc import abstractmethod
from typing import Callable

import numpy as np


class CharFuncInverter:
    """Abstract class for characteristic function inverter"""

    @abstractmethod
    def fit(self, phi: Callable) -> None:
        """Function for setting or changing characteristic function

        Attributes
        ----------
        phi : Callable
              characteristic function
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
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
        raise NotImplementedError
