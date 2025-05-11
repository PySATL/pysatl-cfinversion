
import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial
from scipy.stats import poisson
from .abstract_distribution import AbstractDistribution


class Poisson(AbstractDistribution):
    def __init__(self, mean: float) -> None:
        self.mean = mean

    def chr_(self, x: NDArray[np.float64]) -> NDArray[np.complex128]:
        return np.exp(self.mean * (np.exp(1j * x) - 1))
    def cdf_(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return poisson.cdf(x, self.mean)
    def pdf_(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return poisson.ppf(x, self.mean)
