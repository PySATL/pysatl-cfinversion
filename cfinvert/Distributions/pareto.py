import scipy.special as sc


class Pareto:
    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def cdf(self, x):
        return 1 - (self.scale / x) ** self.shape

    def chr(self, x):
        return self.shape * ((-self.shape * self.scale * x) ** self.shape) * sc.gammainc(-self.shape,
                                                                                        -1j * self.scale * x)

    def pdf(self, x):
        return self.shape * (self.scale ** self.shape) / (x ** (self.shape + 1))
