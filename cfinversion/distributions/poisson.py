import math

import numpy as np
from typing import Optional, Union
from numpy import exp


class Poisson():
    def __init__(self, mean: float) -> None:
        self.mean = mean

    def chr(self,  x: np.ndarray) -> np.ndarray:
        return exp(self.mean * (exp(1j * x) - 1))

    def cdf(self,  x: np.ndarray) -> np.ndarray:
        if x < 0 or not isinstance(x, int):
            raise ValueError("x must be positive integer value")

        cdf_value = 0.0
        for i in range(x + 1):
            cdf_value += self.pdf(i)
        return cdf_value

    def pdf(self,  x: Union[float, np.ndarray]) -> np.ndarray:
        if x < 0 or not isinstance(x, int):
            raise ValueError("x must be positive integer value")

        return  (self.mean**x * np.exp(-self.mean)) / math.factorial(x)
