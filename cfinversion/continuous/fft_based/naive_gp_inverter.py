from typing import Callable, Optional, NoReturn, Union
import numpy as np
from ..continuous_inverter import ContinuousInverter


class NaiveGPInverter(ContinuousInverter):
    def __init__(self, N: float = 1e3, delta: float = 1e-1, num_points: Optional[int] = None) -> None:
        super().__init__()
        self.N: int = int(N)
        self.delta: float = delta
        self.num_points: int = int(N // delta) if num_points is None else num_points
        self.phi: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def fit(self, phi: Callable[[np.ndarray], np.ndarray]) -> None:
        """phi = characteristic function"""
        self.phi = phi

    def cdf(self, x: np.ndarray) -> Union[float, np.ndarray]:
        if self.phi is None:
            raise ValueError("Characteristic function (phi) is not set. Call fit() first.")

        t: np.ndarray = np.linspace(-self.N, self.N, self.num_points)
        phi_t = self.phi(t)

        integral = np.trapezoid(
            (phi_t * np.exp(-1j * t * x[:, np.newaxis])) / (1j * t), t, axis=1
        )

        return 1 / 2 - (1 / (2 * np.pi)) * integral

    def pdf(self, x: np.ndarray) -> Union[float, np.ndarray]:
        if self.phi is None:
            raise ValueError("Characteristic function (phi) is not set. Call fit() first.")

        t: np.ndarray = np.linspace(-self.N, self.N, self.num_points)
        phi_t: np.ndarray = self.phi(t)

        integral = np.trapezoid(
            phi_t * np.exp(-1j * t * x[:, np.newaxis]), t, axis=1
        )
        return (1 / (2 * np.pi)) * integral
