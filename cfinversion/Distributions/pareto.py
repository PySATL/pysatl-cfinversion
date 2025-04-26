import numpy as np
import scipy.special as sc
from typing import Union


class Pareto:
    def __init__(self, shape: float, scale: float) -> None:
        self.shape: float = shape
        self.scale: float = scale

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 1 - (self.scale / x) ** self.shape

    def chr(self, x: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
        return self.shape * ((-self.shape * self.scale * x) ** self.shape) * sc.gammainc(
            -self.shape, -1j * self.scale * x
        )

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.shape * (self.scale ** self.shape) / (x ** (self.shape + 1))