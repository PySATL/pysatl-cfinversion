import numpy as np


def lre(v_true: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
     Log Relative Error gives an approximation
     for the number of correct digits in predicted value (v).
     If the error is 10^(−𝑘), the logarithm tells the 𝑘.

    :param v_true: true value
    :param v: predicted value
    :return: log relative error
    """
    return -np.log10(np.abs((v_true - v) / v_true))

