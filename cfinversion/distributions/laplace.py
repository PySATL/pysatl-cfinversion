import numpy as np
from numpy.typing import NDArray
from .abstract_distribution import AbstractDistribution

class Laplace(AbstractDistribution):
    def __init__(self, m: float, b: float) -> None:
        """
        Initialization of Laplace distribution parameters.

        :param m: Shift parameter (average value).
        :param b: Scale parameter (positive number).
        """
        self.m: float = m
        self.b: float = b

    def chr_(self, x: NDArray[np.float64]) -> NDArray[np.complex128]:
        """
        Characteristic Laplace distribution function.

        :param x: Input value or result array.
        :return: The value of the characteristic function.
        """
        return np.exp(self.m * 1j * x) / (1 + (self.b * x) ** 2)

    def cdf_(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Distribution function (CDF) of the Laplace distribution.

        :param x: Input value or array of values.
        :return: The value of the distribution function.
        """
        result = np.zeros_like(x)
        result[x <= self.m] = 0.5 * np.exp((x[x <= self.m] - self.m) / self.b)
        result[x  > self.m] = 1 - 0.5 * np.exp(-(x[x > self.m] - self.m) / self.b)
        return result

    def pdf_(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Функция плотности вероятности (PDF) распределения Лапласа.

        :param x: Входное значение или массив значений.
        :return: Значение функции плотности вероятности.
        """
        return (1 / (2 * self.b)) * np.exp(-np.abs(x - self.m) / self.b)