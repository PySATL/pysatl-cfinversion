import numpy as np
from numpy.typing import NDArray
from scipy.special import erf
from .abstract_distribution import AbstractDistribution
class UniformSquared(AbstractDistribution):
    def __init__(self, a: float, b: float) -> None:
        self.a: float = a
        self.b: float = b

    def chr_(self, x: NDArray[np.float64]) -> NDArray[np.complex128]:
        result = np.ones_like(x, np.complex128)
        result[x != 0] = np.exp(1.0j * x[x != 0] * self.a) * np.sqrt(np.pi) *\
                erf(np.sqrt(-1.0j*(self.b - self.a) * x[x != 0] )) / (2 * np.sqrt(-1.0j * (self.b-self.a) * x[x != 0]))
        return result

    def cdf_(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        result = np.zeros_like(x)
        result[x >= self.b] = 1
        result[(x >= self.a) & (x < self.b)] = np.sqrt(
                (x[(x >= self.a) & (x < self.b)] - self.a) / (self.b - self.a)
        )
        return result

    def pdf_(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        result = np.zeros_like(x)
        result[(x > self.a) & (x < self.b)] = 1 / (2 * np.sqrt(self.b - self.a)  * np.sqrt(x[(x > self.a) & (x < self.b)] - self.a))
        return result
