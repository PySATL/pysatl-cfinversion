import numpy as np
import pytest

import CFInvert.CharFuncInverter.Bohman.BohmansInverters as BI



@pytest.fixture(scope="function")
def parameters_setup(request):
    N = 512
    K = 4
    d = 0.35 / 16
    delta = (2 * np.pi) / (N * d)
    return N, delta, K


xs = np.array([-5.6, -4.55, -3.5, -2.45, -1.4, -1.05, -0.7, -0.35, 0, 0.35, 0.7, 1.4, 1.75, 2.1, 3.15, 4.2, 5.25])
F_exact = {
    -5.60: 0,
    -4.55: 0,
    -3.50: 0,
    -2.45: 0,
    -1.40: 0,
    -1.05: 0,
    -0.70: 79586,
    -0.35: 522700,
    0: 682690,
    0.35: 778553,
    0.70: 841654,
    1.40: 915695,
    1.75: 937693,
    2.10: 953678,
    3.15: 980485,
    4.20: 991570,
    5.25: 996298
}


def bochmans_testcase(inv, eps):
    """Общий тест для инверторов."""
    inv.fit(lambda t: ((1 - 1j * t * np.sqrt(2)) ** (-0.5)) * np.exp((-1j * t) / np.sqrt(2)))

    difference = inv.cdf(xs).real * 1e6 - np.array(list(F_exact.values()))
    MSE = np.sum(difference ** 2) / len(xs)

    assert MSE < eps


def test_bochman_a(parameters_setup):
    N, delta, _ = parameters_setup
    inv = BI.BohmanA(N, delta)
    bochmans_testcase(inv, 164e4)


def test_bochman_b(parameters_setup):
    N, delta, _ = parameters_setup
    inv = BI.BohmanB(N, delta)
    bochmans_testcase(inv, 479e4)


def test_bochman_c(parameters_setup):
    N, delta, _ = parameters_setup
    inv = BI.BohmanC(N, delta)
    bochmans_testcase(inv, 163e4)


def test_bochman_d(parameters_setup):
    N, delta, K = parameters_setup
    inv = BI.BohmanD(N, delta, K)  # wtf
    bochmans_testcase(inv, 105e4)


def test_bochman_e(parameters_setup):
    N, delta, K = parameters_setup
    inv = BI.BohmanE(N, delta, K)
    bochmans_testcase(inv, 417e4)

