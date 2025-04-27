import numpy as np
from numpy import exp
from scipy.special import erf
from typing import Union


class UniformSquared:
    def __init__(self, a: float, b: float) -> None:
        self.a: float = a
        self.b: float = b

    def chr(self, x: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
        if isinstance(x, np.ndarray):
            result = np.ones_like(x, np.complex128)
            result[x != 0] = np.exp(1.0j * x[x != 0] * self.a) * np.sqrt(np.pi) * erf(np.sqrt(-1.0j*(self.b - self.a) * x[x != 0] )) / (2 * np.sqrt(-1.0j * (self.b-self.a) * x[x != 0]))
            return result
        if x == 0.0:
            return 1.0 + 0.0j
        return np.exp(1.0j * x * self.a) * np.sqrt(np.pi) * erf(np.sqrt(-1.0j*(self.b - self.a) * x )) / (2 * np.sqrt(-1.0j * (self.b-self.a) * x))

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x_arr = np.asarray(x)

        result = np.zeros_like(x_arr)

        result[x_arr >= self.b] = 1
        result[(x_arr >= self.a) & (x_arr < self.b)] = np.sqrt(
                (x_arr[(x_arr >= self.a) & (x_arr < self.b)] - self.a) / (self.b - self.a)
        )

        if isinstance(x, float):
            return float(result)
        return result

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        result = np.zeros_like(x)
        result[(x > self.a) & (x < self.b)] = 1 / (2 * np.sqrt(self.b - self.a)  * np.sqrt(x[(x > self.a) & (x < self.b)] - self.a))
        return result
