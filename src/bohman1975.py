import numpy as np
import matplotlib.pyplot as plt
from mpmath import *

class ChrFunc:
    def __init__(self, phi):
        self.phi = phi

    def __get_error__(self, true_chr):
        pass

    def cdf(self, x):
        pass

    def make_plot(self, start, stop, num):
        x = np.linspace(start, stop, num)
        F_x = np.array([self.cdf(y) for y in x])

        plt.plot(x, F_x, label='Функция распределения')
        plt.title('График функции распределения')
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.grid()
        plt.legend()
        plt.show()


import scipy.stats as stats

# Straight on
class A(ChrFunc):
    def __init__(self, N, delta, phi):
        super().__init__(phi)
        self.N = int(N)
        self.delta = delta

    def cdf(self, x):
        F = 0.5 + (self.delta * x) / (2 * pi)
        for v in range(1 - self.N, self.N):
            if v == 0:
                continue
            F -= (self.phi(self.delta * v) / (2 * pi * 1j * v)) * exp(-1j * self.delta * v * x)
        return F


# Battling the truncation error by deforming F
class B(ChrFunc):
    def __init__(self, N, delta, phi):
        super().__init__(phi)
        self.N = int(N)
        self.delta = delta

    def __C(self, t):
        if t > 1:
            return 0
        if t < 0:
            return self.__C(-t)
        return (1 - t) * cos(pi * t) + sin(pi * t) / pi

    def cdf(self, x):
        F = 0.5 + (self.delta * x) / (2 * pi)
        for v in range(1 - self.N, self.N):
            if v == 0:
                continue
            F -= self.__C(v / self.N) * (self.phi(self.delta * v) / (2 * pi * 1j * v)) * exp(-1j * self.delta * v * x)
        return F


# Reducing importance of trigonometric series by considering difference between F and 
class C(ChrFunc):
    def __init__(self, N, delta, phi):
        super().__init__(phi)
        self.N = int(N)
        self.delta = delta

    def cdf(self, x):
        F = stats.norm.cdf(x, loc=0, scale=1)
        for v in range(1 - self.N, self.N):
            if v == 0:
                continue
            p = self.delta * v
            F += ((exp(- (p ** 2) / 2) - self.phi(p)) / (2 * pi * 1j * v)) * exp(-1j * p * x)
        return F


# Reducing the aliasing error and reducing importance of trigonometric series
class D(ChrFunc):
    def __init__(self, N, delta, phi, K):
        super().__init__(phi)
        self.N = int(N)
        self.delta = delta
        self.K = K

    def __H(self, x, delta):
        H = 0
        for v in range(1 - self.N, self.N):
            if v == 0:
                continue
            p = delta * v
            H += ((exp(- (p ** 2) / 2) - self.phi(p)) / (2 * pi * 1j * v)) * exp(-1j * p * x)
        return H

    def cdf(self, x):
        F = stats.norm.cdf(x, loc=0, scale=1) + self.__H(x, self.delta)
        d = (2 * pi) / (self.N * self.delta)
        for v in range(1, self.K):
            L = self.N // self.K
            delta_1 = self.delta / self.K
            d_1 = self.K * d
            F -= self.__H(x + v * L * d_1, delta_1)
        return F


# Reducing the aliasing error and Reducing importance of trigonometric
# series and Battling the truncation error by deforming F
class E(ChrFunc):
    def __init__(self, N, delta, phi, K):
        super().__init__(phi)
        self.N = int(N)
        self.delta = delta
        self.K = K

    def __C(self, t):
        if t > 1:
            return 0
        if t < 0:
            return self.__C(-t)
        return (1 - t) * cos(pi * t) + sin(pi * t) / pi

    def __G(self, x, delta):
        G = 0
        for v in range(1 - self.N, self.N):
            if v == 0:
                continue
            p = delta * v
            G += self.__C(v / self.N) * ((exp(- (p ** 2) / 2) - self.phi(p)) / (2 * pi * 1j * v)) * exp(-1j * p * x)
        return G

    def cdf(self, x):
        F = stats.norm.cdf(x, loc=0, scale=1) + self.__G(x, self.delta)
        d = (2 * pi) / (self.N * self.delta)
        for v in range(1, self.K):
            L = self.N // self.K
            delta_1 = self.delta / self.K
            d_1 = self.K * d
            F -= self.__G(x + v * L * d_1, delta_1)
        return F
