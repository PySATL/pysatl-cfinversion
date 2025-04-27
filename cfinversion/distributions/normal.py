import numpy as np
from numpy import exp
from scipy.stats import norm

from cfinversion.Distributions.AbstractDistribution import AbstractDistribution

class Normal(AbstractDistribution):
    def __init__(self, m: float, var: float) -> None:
        """
        Конструктор класса Norm.

        :param m: математическое ожидание (среднее значение) нормального распределения
        :param var: дисперсия нормального распределения
        """
        self.m = m
        self.var = var

    def chr(self,  x: np.ndarray) -> np.ndarray:
        """
        Метод для вычисления характеристической функции нормального распределения.

        :param x: аргумент характеристической функции
        :return: значение характеристической функции в точке x
        """
        return exp(self.m * 1j * x - (self.var * (x ** 2)) / 2)

    def cdf(self,  x: np.ndarray) -> np.ndarray:
        """
        Метод для вычисления функции распределения (CDF) нормального распределения.

        :param x: аргумент функции распределения
        :return: значение функции распределения в точке x
        """
        return norm.cdf(x, loc=self.m, scale=np.sqrt(self.var))

    def pdf(self,  x: np.ndarray) -> np.ndarray:
        """
        Метод для вычисления плотности вероятности (PDF) нормального распределения.

        :param x: аргумент плотности вероятности
        :return: значение плотности вероятности в точке x
        """
        return norm.pdf(x, loc=self.m, scale=self.var)