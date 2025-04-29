import numpy as np
from numpy import exp, abs
from typing import Union


class Laplace():
    def __init__(self, m: float, b: float) -> None:
        """
        Initialization of Laplace distribution parameters.

        :param m: Shift parameter (average value).
        :param b: Scale parameter (positive number).
        """
        self.m: float = m
        self.b: float = b

    def chr(self, x: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
        """
        Characteristic Laplace distribution function.

        :param x: Input value or result array.
        :return: The value of the characteristic function.
        """
        return exp(self.m * 1j * x) / (1 + (self.b * x) ** 2)

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Distribution function (CDF) of the Laplace distribution.

        :param x: Input value or array of values.
        :return: The value of the distribution function.
        """
        result: np.ndarray = np.zeros_like(x)
        result[x <= self.m] = 0.5 * exp((x[x <= self.m] - self.m) / self.b)
        result[x > self.m] = 1 - 0.5 * exp(-(x[x > self.m] - self.m) / self.b)
        return result

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Функция плотности вероятности (PDF) распределения Лапласа.

        :param x: Входное значение или массив значений.
        :return: Значение функции плотности вероятности.
        """
        return (1 / (2 * self.b)) * exp(-abs(x - self.m) / self.b)