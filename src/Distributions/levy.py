import numpy as np
from scipy.special import erfc

class Levy:
    def __init__(self, c, mu):
        self.c = c
        self.mu = mu

    def chr(self, x):
        return np.exp(1j * self.mu * x - np.sqrt(-2*1j*self.c*x))

    def cdf(self, x):
        return erfc(np.sqrt(self.c / (2 * (x - self.mu))))

    def pdf(self, x):
        return np.sqrt(self.c/(2 * np.pi)) * (np.exp(-self.c/(2*(x-self.mu))) / ((x-self.mu) ** 1.5))
