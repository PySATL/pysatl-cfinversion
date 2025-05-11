import numpy as np
from numpy.typing import NDArray
from .abstract_distribution import AbstractDistribution

class Uniform(AbstractDistribution):
    def __init__(self, a: float, b: float) -> None:
        self.a: float = a
        self.b: float = b

    def chr_(self, x: NDArray[np.float64]) -> NDArray[np.complex128]:
        result = np.ones_like(x, np.complex128)
        result[x != 0] = (np.exp(1j * x[x != 0] * self.b) - np.exp(1j * x[ x!= 0] * self.a)) / (1j * x[x != 0] * (self.b - self.a))
        return result

    def cdf_(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        result = np.zeros_like(x)
        result[x >= self.b] = 1
        result[(x >= self.a) & (x < self.b)] = (
                (x[(x >= self.a) & (x < self.b)] - self.a) / (self.b - self.a)
        )
        return result

    def pdf_(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        result = np.zeros_like(x)
        result[(x >= self.a) & (x < self.b)] = 1 / (self.b - self.a)
        return result
