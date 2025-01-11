import numpy as np
from numpy import pi, exp, sin, cos
from scipy.stats import norm

from src.CharFuncInverter import CharFuncInverter


class BohmanA(CharFuncInverter):
    """Straight on"""

    def __init__(self, N=1e3, delta=1e-1):
        super().__init__()
        self.N = int(N)
        self.delta = delta

    def fit(self, phi):
        self.coeff_0 = 0.5
        self.coeff_1 = self.delta / (2 * pi)
        self.coeff = np.array([phi(self.delta * v) / (2 * pi * 1j * v) for v in range(1 - self.N, self.N) if v != 0])

    def cdf(self, X):
        v = np.arange(1 - self.N, self.N)
        v_non_zero = v[v != 0]

        x_vect = np.outer(X, v_non_zero)

        F_x = self.coeff_0 + X * self.coeff_1 + (-exp(-1j * self.delta * x_vect) @ self.coeff)

        return F_x


class BohmanB(CharFuncInverter):
    """Battling the truncation error by deforming F"""

    def __init__(self, N=1e3, delta=1e-1):
        super().__init__()
        self.N = int(N)
        self.delta = delta

    def __C(self, t):
        if t > 1:
            return 0
        if t < 0:
            return self.__C(-t)
        return (1 - t) * cos(pi * t) + sin(pi * t) / pi

    def fit(self, phi):
        self.coeff_0 = 0.5
        self.coeff_1 = self.delta / (2 * pi)
        self.coeff = np.array(
            [self.__C(v / self.N) * phi(self.delta * v) / (2 * pi * 1j * v) for v in range(1 - self.N, self.N) if
             v != 0])

    def cdf(self, X):
        v = np.arange(1 - self.N, self.N)
        v_non_zero = v[v != 0]

        x_vect = np.outer(X, v_non_zero)

        F_x = self.coeff_0 + X * self.coeff_1 + (-exp(-1j * self.delta * x_vect) @ self.coeff)

        return F_x


class BohmanC(CharFuncInverter):
    """Reducing importance of trigonometric series by considering difference between F and <I>"""

    def __init__(self, N=1e3, delta=1e-1):
        super().__init__()
        self.N = int(N)
        self.delta = delta

    def fit(self, phi):
        self.coeff = np.array([((exp(- ((self.delta * v) ** 2) / 2) - phi(self.delta * v)) / (2 * pi * 1j * v)) for v in
                               range(1 - self.N, self.N) if v != 0])

    def cdf(self, X):
        v = np.arange(1 - self.N, self.N)
        v_non_zero = v[v != 0]

        x_vect = np.outer(X, v_non_zero)

        F_x = norm.cdf(X, loc=0, scale=1) + (exp(-1j * self.delta * x_vect) @ self.coeff)

        return F_x


class BohmanD(CharFuncInverter):
    """Reducing the aliasing error and reducing importance of trigonometric series"""

    def __init__(self, N=1e3, delta=1e-1, K=2):
        super().__init__()
        self.N = int(N)
        self.delta = delta
        self.K = K
        self.delta_1 = self.delta / self.K

    def fit(self, phi):
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
                2 * pi * 1j * self.delta_1)

    def cdf(self, X):
        v = np.arange(1 - self.N, self.N)
        v_non_zero = v[v != 0]

        x_vect = np.outer(X, v_non_zero)

        F_x = norm.cdf(X, loc=0, scale=1) + (exp(-1j * x_vect * self.delta) @ self.coeff_1) + (
                exp(-1j * x_vect * self.delta_1) @ self.coeff_2)

        return F_x


class BohmanE(CharFuncInverter):
    """Reducing the aliasing error and Reducing importance of trigonometric series and Battling the truncation error by deforming F"""

    def __init__(self, N=1e3, delta=1e-1, K=4):
        super().__init__()
        self.N = int(N)
        self.delta = delta
        self.K = K
        self.delta_1 = self.delta / self.K

    def __C(self, t):
        result = np.zeros_like(t)

        t_negative = t[t < 0]
        result[t < 0] = (1 + t_negative) * cos(pi * -t_negative) + sin(pi * -t_negative) / pi

        t_between = t[(0 <= t) & (t <= 1)]
        result[(0 <= t) & (t <= 1)] = (1 - t_between) * cos(pi * t_between) + sin(pi * t_between) / pi

        return result

    def fit(self, phi):
        v_values = np.arange(1 - self.N, self.N)
        v_values = v_values[v_values != 0]

        С_values = self.__C(v_values / self.N)
        C_values = np.ones_like(v_values)

        self.coeff_1 = С_values * (exp(-((self.delta * v_values) ** 2) / 2) - phi(self.delta * v_values)) / (
                2 * pi * 1j * v_values)

        L = self.N // self.K
        d = (2 * pi) / (self.N * self.delta)
        d_1 = self.K * d

        i_values = np.arange(1, self.K)

        exp_matrix = np.exp(-1j * self.delta_1 * v_values[:, np.newaxis] * i_values * L * d_1)
        exp_coeff = np.sum(exp_matrix, axis=1)

        self.coeff_2 = -(exp(-((self.delta_1 * v_values) ** 2) / 2) - phi(self.delta_1 * v_values)) / (
                2 * pi * 1j * self.delta_1)

    def cdf(self, X):
        v = np.arange(1 - self.N, self.N)
        v_non_zero = v[v != 0]

        x_vect = np.outer(X, v_non_zero)

        F_x = norm.cdf(X, loc=0, scale=1) + (exp(-1j * x_vect * self.delta) @ self.coeff_1) + (
                exp(-1j * x_vect * self.delta_1) @ self.coeff_2)
        return F_x
