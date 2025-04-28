import numpy as np
import scipy as sp
from typing import Callable
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

def l0_err(f: Callable, tol_diff:float = 1e-3) -> float:
    return 1 - sp.integrate.quad(f, -np.inf, np.inf, epsabs = tol_diff)[0]
