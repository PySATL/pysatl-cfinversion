from numpy import exp


class Poisson:
    def __init__(self, mean):
        self.mean = mean

    def cdf(self, x):
        pass

    def chr(self, x):
        return exp(self.mean * (exp(1j * x) - 1))

    def pdf(self, x):
        pass
