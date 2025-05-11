import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc
from .abstract_distribution import AbstractDistribution

class Levy(AbstractDistribution):
    def __init__(self, c: float, mu: float) -> None:
        self.c = c
        self.mu = mu

    def chr_(self, x: NDArray[np.float64]) -> NDArray[np.complex128]:
        return np.exp(1j * self.mu * x - np.sqrt(-2*1j*self.c*x))

    def cdf_(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return erfc(np.sqrt(self.c / (2 * (x - self.mu))))

    def pdf_(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sqrt(self.c/(2 * np.pi)) * (np.exp(-self.c/(2*(x-self.mu))) / ((x-self.mu) ** 1.5))
