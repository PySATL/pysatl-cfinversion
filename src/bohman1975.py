from math import *
import scipy.stats as stats
from scipy.special import delta


class simple():
  def __init__(self, function, N, delta):
    self.N = N
    self.delta = delta

  def straight_on(self, x, phi):
    F = 0.5 + (self.delta * x) / (2 * pi)
    for v in range(1-self.N, self.N):
        if v == 0:
            continue
        F -= (phi(self.delta * v) / (2 * pi * 1j * v)) * exp(-1j * self.delta * v * x)
    return F

  def __C(self, t):
    if t > 1:
      return 0
    if t < 0:
      return self.__C(-t)
    return (1-t) * cos(pi * t) + sin(pi * t) / pi

  def deforming_F(self, x, phi):
    F = 0.5 + (self.delta * x) / (2 * pi)
    for v in range(1 - self.N, self.N):
        if v == 0:
            continue
        F -= self.__C(v / self.N) * (phi(self.delta * v) / (2 * pi * 1j * v)) * exp(-1j * self.delta * v * x)
    return F

  def trigonometric_series(self, x, phi):
    F = stats.norm.cdf(x, loc=0, scale=1)
    for v in range(1 - self.N, self.N):
        if v == 0:
            continue
        p = self.delta * v
        F += ((exp(- (p ** 2)) - phi(p))/(2 * pi * 1j * v)) * exp(-1j * p * x)
    return F



  def reduce_aliad_error(self, x, phi):

  def __G(self, x, phi):
      G = 0
      for v in range(1 - self.N, self.N):
          if v == 0:
              continue
          p = self.delta * v
          G += self.__C(v / self.N) * ((exp(- (p ** 2)) - phi(p)) / (2 * pi * 1j * v)) * exp(-1j * p * x)
      return G

  def the_best(self, x, phi, K):
      F = stats.norm.cdf(x, loc=0, scale=1) + self.__G(x, phi)
      for v in range(1, K):
          if v == 0:
              continue
          F -= self.__G(x + v)

