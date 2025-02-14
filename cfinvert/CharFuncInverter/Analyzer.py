import numpy as np
from matplotlib import pyplot as plt

def lre(v_true, v):
    """
     Log Relative Error gives an approximation
     for the number of correct digits in predicted value (v).
     If the error is 10^(−𝑘), the logarithm tells the 𝑘.

    :param v_true: true value
    :param v: predicted value
    :return: log relative error
    """
    return -np.log10(np.abs((v_true - v) / v_true))

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


