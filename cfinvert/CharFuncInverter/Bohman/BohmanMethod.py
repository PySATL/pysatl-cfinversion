from abc import ABC
from typing import Type

import numpy as np

from cfinvert.CharFuncInverter.CharFuncInverter import CharFuncInverter


class BohmanMethod(CharFuncInverter, ABC):
    """Abstract class for characteristic function inverter,
    which are implemented using the methods described by Harald Bohman in 1975"""

    @staticmethod
    def _C(t: np.ndarray) -> np.ndarray:
        result = np.zeros_like(t)

        t_negative = t[(t >= -1) & (t <= 0)]
        result[(t >= -1) & (t <= 0)] = (1 + t_negative) * np.cos(np.pi * -t_negative) + np.sin(
            np.pi * -t_negative) / np.pi

        t_positive = t[(0 <= t) & (t <= 1)]
        result[(0 <= t) & (t <= 1)] = (1 - t_positive) * np.cos(np.pi * t_positive) + np.sin(np.pi * t_positive) / np.pi

        return result

    def pdf(self, x: np.ndarray) -> Type[NotImplementedError]:
        """Bohmans methods allows to implement only cumulative distribution function"""
        return NotImplementedError
