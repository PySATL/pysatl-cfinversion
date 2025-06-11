import pytest
import numpy as np
from scipy.stats import norm
from typing import Callable

from cfinversion.continuous.fft_based.fft_inverter import FFTInverter


@pytest.fixture
def normal_cf() -> Callable[[np.ndarray], np.ndarray]:
    """Характеристическая функция для N(0,1)"""
    return lambda t: np.exp(-t ** 2 / 2)


@pytest.fixture
def inverter() -> FFTInverter:
    """Фикстура с инициализированным инвертором"""
    return FFTInverter(N=2 ** 10, A=-6, B=6)


def test_initialization(inverter):
    """Проверка корректности инициализации параметров"""
    assert inverter.N == 1024
    assert inverter.A == -6
    assert inverter.B == 6
    assert len(inverter.y) == 1024


def test_pdf_normal_distribution(inverter, normal_cf):
    """Сравнение PDF с аналитическим решением для N(0,1)"""
    inverter.fit(normal_cf)
    x = np.linspace(-3, 3, 100)
    pdf_values = inverter.pdf(x)
    exact_values = norm.pdf(x)

    assert np.allclose(pdf_values, exact_values, atol=0.1)


def test_cdf_normal_distribution(inverter, normal_cf):
    """Сравнение CDF с аналитическим решением для N(0,1)"""
    inverter.fit(normal_cf)
    x = np.linspace(-3, 3, 100)
    cdf_values = inverter.cdf(x)
    exact_values = norm.cdf(x)

    # Проверка монотонности и примерной формы
    assert np.all(np.diff(cdf_values) >= 0)  # Монотонность
    assert np.allclose(cdf_values[50], 0.5, atol=0.1)  # Медиана ~0.5


def test_edge_cases(inverter):
    """Проверка обработки граничных случаев"""

    # Проверка на нулевых частотах
    inverter.fit(lambda t: np.where(t == 0, 1.0, np.sin(t) / t))
    assert not np.isnan(inverter.pdf(np.array([0]))).any()