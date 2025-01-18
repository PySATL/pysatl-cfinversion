from numpy import exp
from scipy.stats import norm

class Norm:
  def __init__(self, m, var):
      self.m = m
      self.var = var

  def chr(self, x):
      return exp(self.m * 1j * x - (self.var * (x ** 2)) / 2)

  def cdf(self, x):
      return norm.cdf(x, loc=self.m, scale=self.var)

  def pdf(self, x):
      return norm.pdf(x, loc=self.m, scale=self.var)