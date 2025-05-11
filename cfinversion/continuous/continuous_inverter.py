from abc import abstractmethod
from typing import Callable, Union
import numpy as np
from numpy.typing import NDArray


class ContinuousInverter:
    """Abstract class for characteristic function inverter"""

    @abstractmethod
    def fit(self, cf: Callable) -> None:
        """Function for setting or changing characteristic function

        Attributes
        ----------
        cf : Callable
              characteristic function
        """

        raise NotImplementedError

    @abstractmethod
    def cdf(self, x: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
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
    def pdf(self, x: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
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
