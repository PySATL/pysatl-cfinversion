import numpy as np
from numpy import exp

class Unif:
  def __init__(self, a, b):
      self.a = a
      self.b = b

  def chr(self, x):
      return (exp(1j * x * self.b) - exp(1j * x * self.a)) / (1j * x * (self.b - self.a))

  def cdf(self, x):
      x = np.asarray(x)
      result = np.zeros_like(x)
      result[x >= self.b] = 1
      result[(x >= self.a) & (x < self.b)] = (x[(x >= self.a) & (x < self.b)] - self.a) / (self.b - self.a)

      return result
