import numpy as np
from numpy import exp, abs


class Laplace:
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def chr(self, x):
        return exp(self.m * 1j * x) / (1 + (self.b * x) ** 2)

    def cdf(self, x):
        result = np.zeros_like(x)
        result[x <= self.m] = 0.5 * exp((x - self.m) / self.b)
        result[x > self.m] = 1 - 0.5 * exp(-(x - self.m) / self.b)
        return result

    def pdf(self, x):
        return (1 / (2 * self.b)) * exp(-abs(x - self.m) / self.b)
