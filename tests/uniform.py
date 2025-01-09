from numpy import exp

class Unif:
  def __init__(self, a, b):
      self.a = a
      self.b = b

  def chr(self, x):
      return (exp(1j * x * self.b) - exp(1j * x * self.a)) / (1j * x * (self.b - self.a))

  def cdf(self, x):
      if x < self.a:
        return 0
      if x >= self.b:
        return 1
      return (x - self.a) / (self.b - self.a)