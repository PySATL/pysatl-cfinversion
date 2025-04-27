import numpy as np
from numpy import exp
from typing import Union


class Uniform:
    def __init__(self, a: float, b: float) -> None:
        self.a: float = a
        self.b: float = b

    def chr(self, x: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
        return (exp(1j * x * self.b) - exp(1j * x * self.a)) / (1j * x * (self.b - self.a))

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x_arr = np.asarray(x)

        result = np.zeros_like(x_arr)

        result[x_arr >= self.b] = 1
        result[(x_arr >= self.a) & (x_arr < self.b)] = (
                (x_arr[(x_arr >= self.a) & (x_arr < self.b)] - self.a) / (self.b - self.a)
        )

        if isinstance(x, float):
            return float(result)
        return result

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        result = np.zeros_like(x)
        result[(x >= self.a) & (x < self.b)] = 1 / (self.b - self.a)
        return result
