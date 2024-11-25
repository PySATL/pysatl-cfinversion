import numpy as np
import matplotlib.pyplot as plt

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
        print(x)
        print(F_x)

        plt.plot(x, F_x, label='Функция распределения')
        plt.title('График функции распределения')
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.grid()
        plt.legend()
        plt.show()


