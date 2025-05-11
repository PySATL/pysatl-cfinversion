import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from .abstract_distribution import AbstractDistribution

class Norm(AbstractDistribution):
    def __init__(self, m: float, sd: float) -> None:
        """
        Конструктор класса Norm.

        :param m: математическое ожидание (среднее значение) нормального распределения
        :param sd: стандартное отклонение
        """
        self.m = m
        self.sd = sd

    def chr_(self, x: NDArray[np.float64]) -> NDArray[np.complex128]:
        """
        Метод для вычисления характеристической функции нормального распределения.

        :param x: аргумент характеристической функции
        :return: значение характеристической функции в точке x
        """
        return np.exp(self.m * 1j * x - ((self.sd * x) ** 2 / 2))

    def cdf_(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Метод для вычисления функции распределения (CDF) нормального распределения.

        :param x: аргумент функции распределения
        :return: значение функции распределения в точке x
        """
        return norm.cdf(x, loc=self.m, scale=self.sd)

    def pdf_(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Метод для вычисления плотности вероятности (PDF) нормального распределения.

        :param x: аргумент плотности вероятности
        :return: значение плотности вероятности в точке x
        """
        return norm.pdf(x, loc=self.m, scale=self.sd)