from abc import abstractmethod


import numpy as np


class AbstractDistribution:
    """Abstract class for distributions"""

    @abstractmethod
    def chr(self,  x: np.ndarray) -> np.ndarray:
        """
        Function return characteristic function of distribution

        :param x: An input value or an array of values.
        :return: The value of the characteristic function.
        """
        raise NotImplementedError

    @abstractmethod
    def cdf(self,  x: np.ndarray) -> np.ndarray:
        """
        Function return cumulative distribution function

        :param x: An input value or an array of values.
        :return: The value of the distribution function.
        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self,  x: np.ndarray) -> np.ndarray:
        """
        Function return probability density function

        :param x: An input value or an array of values.
        :return: The value of the probability density function.
        """
        raise NotImplementedError