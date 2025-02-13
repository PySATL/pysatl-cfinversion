from typing import Callable

import numpy as np
from numpy import pi, exp
from scipy.stats import norm

from src.CharFuncInverter.Bohman.BohmanMethod import BohmanMethod


class BohmanA(BohmanMethod):
    """Straight on"""

    """
    N - positive integer
    delta - positive quantity
    d = 2pi / N*delta
    """

    def __init__(self, N: int = int(1e3), delta: float = 1e-1) -> None:
        super().__init__()
        self.N = N
        self.delta = delta

    def fit(self, phi: Callable[[np.ndarray], complex]) -> None:
        self.coeff_0 = 0.5
        self.coeff_1 = self.delta / (2 * pi)

        v_values = np.arange(1 - self.N, self.N)
        v_values = v_values[v_values != 0]

        self.coeff = phi(self.delta * v_values) / (2 * pi * 1j * v_values)

    def cdf(self, X: np.ndarray) -> np.ndarray[float]:
        v = np.arange(1 - self.N, self.N)
        v_non_zero = v[v != 0]

        x_vect = np.outer(X, v_non_zero)

        F_x = self.coeff_0 + X * self.coeff_1 + (-exp(-1j * self.delta * x_vect) @ self.coeff)

        return F_x.real


class BohmanB(BohmanMethod):
    """Battling the truncation error by deforming F"""

    def __init__(self, N: int = int(1e3), delta: float = 1e-1) -> None:
        super().__init__()
        self.N = N
        self.delta = delta

    def fit(self, phi: Callable[[np.ndarray], complex]) -> None:
        self.coeff_0 = 0.5
        self.coeff_1 = self.delta / (2 * pi)

        v_values = np.arange(1 - self.N, self.N)
        v_values = v_values[v_values != 0]
        self.coeff = super()._C(v_values / self.N) * phi(self.delta * v_values) / (2 * pi * 1j * v_values)

    def cdf(self, X: np.ndarray) -> np.ndarray[float]:
        v = np.arange(1 - self.N, self.N)
        v_non_zero = v[v != 0]

        x_vect = np.outer(X, v_non_zero)

        F_x = self.coeff_0 + X * self.coeff_1 + (-exp(-1j * self.delta * x_vect) @ self.coeff)

        return F_x.real


class BohmanC(BohmanMethod):
    """Reducing importance of trigonometric series by considering difference between F and <I>"""

    def __init__(self, N: float = 1e3, delta: float = 1e-1) -> None:
        super().__init__()
        self.N = int(N)
        self.delta = delta

    def fit(self, phi: Callable[[np.ndarray], complex]) -> None:
        self.coeff = np.array([((exp(- ((self.delta * v) ** 2) / 2) - phi(self.delta * v)) / (2 * pi * 1j * v)) for v in
                               range(1 - self.N, self.N) if v != 0])

    def cdf(self, X: np.ndarray) -> np.ndarray:
        v = np.arange(1 - self.N, self.N)
        v_non_zero = v[v != 0]

        x_vect = np.outer(X, v_non_zero)

        F_x = norm.cdf(X, loc=0, scale=1) + (exp(-1j * self.delta * x_vect) @ self.coeff)

        return F_x


class BohmanD(BohmanMethod):
    """Reducing the aliasing error and reducing importance of trigonometric series"""

    def __init__(self, N: int = int(1e3), delta: float = 1e-1, K: int = 2) -> None:
        super().__init__()
        self.N = N
        self.delta = delta
        self.K = K
        self.delta_1 = self.delta / self.K

    def fit(self, phi: Callable[[np.ndarray], complex]) -> None:
        self.coeff_1 = np.array([(exp(-((self.delta * v) ** 2) / 2) - phi(self.delta * v)) / (2 * pi * 1j * v) for v in
                                 range(1 - self.N, self.N) if v != 0])
        L = self.N // self.K
        d = (2 * pi) / (self.N * self.delta)
        d_1 = self.K * d

        v_values = np.arange(1 - self.N, self.N)
        v_values = v_values[v_values != 0]
        i_values = np.arange(1, self.K)

        exp_matrix = np.exp(-1j * self.delta_1 * v_values[:, np.newaxis] * i_values * L * d_1)
        exp_coeff = np.sum(exp_matrix, axis=1)

        self.coeff_2 = - (exp(-((self.delta_1 * v_values) ** 2) / 2) - phi(self.delta_1 * v_values)) / (
                2 * pi * 1j * v_values) * exp_coeff

    def cdf(self, X):
        v = np.arange(1 - self.N, self.N)
        v_non_zero = v[v != 0]

        x_vect = np.outer(X, v_non_zero)

        F_x = norm.cdf(X, loc=0, scale=1) + (exp(-1j * x_vect * self.delta) @ self.coeff_1) + (
                exp(-1j * x_vect * self.delta_1) @ self.coeff_2)

        return F_x


class BohmanE(BohmanMethod):
    """Reducing the aliasing error and Reducing importance of trigonometric
    series and Battling the truncation error by deforming F"""

    def __init__(self, N: int = int(1e3), delta: float = 1e-1, K: int = 4) -> None:
        super().__init__()
        self.N = N
        self.delta = delta
        self.K = K
        self.delta_1 = self.delta / self.K

    def fit(self, phi: Callable[[np.ndarray], complex]) -> None:
        v_values = np.arange(1 - self.N, self.N)
        v_values = v_values[v_values != 0]

        С_values = super()._C(v_values / self.N)

        self.coeff_1 = С_values * (exp(-((self.delta * v_values) ** 2) / 2) - phi(self.delta * v_values)) / (
                2 * pi * 1j * v_values)

        L = self.N // self.K
        d = (2 * pi) / (self.N * self.delta)
        d_1 = self.K * d

        i_values = np.arange(1, self.K)

        exp_matrix = np.exp(-1j * self.delta_1 * v_values[:, np.newaxis] * i_values * L * d_1)
        exp_coeff = np.sum(exp_matrix, axis=1)

        self.coeff_2 = -С_values * ((exp(-((self.delta_1 * v_values) ** 2) / 2) - phi(self.delta_1 * v_values)) / (
                2 * pi * 1j * v_values)) * exp_coeff

    def cdf(self, X: np.ndarray) -> None:
        v = np.arange(1 - self.N, self.N)
        v_non_zero = v[v != 0]

        x_vect = np.outer(X, v_non_zero)

        F_x = norm.cdf(X, loc=0, scale=1) + (exp(-1j * x_vect * self.delta) @ self.coeff_1) + (
                exp(-1j * x_vect * self.delta_1) @ self.coeff_2)
        return F_x
