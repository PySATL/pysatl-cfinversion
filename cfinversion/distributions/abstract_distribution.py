import numpy as np
from abc import abstractmethod, ABC
from numpy.typing import NDArray

class AbstractDistribution(ABC):
    @abstractmethod
    def chr_(self, x: NDArray[np.float64]) -> NDArray[np.complex128]:
        pass
    @abstractmethod
    def cdf_(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        pass
    @abstractmethod
    def pdf_(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        pass
    
    def chr(self, x: float | NDArray[np.float64]) -> complex | NDArray[np.complex128]:
        result = self.chr_(np.asarray(x))
        if isinstance(x, float):
            return result.item()
        return result
    def cdf(self, x: float | NDArray[np.float64]) -> float |NDArray[np.float64]:
        result = self.cdf_(np.asarray(x))
        if isinstance(x, float):
            return result.item()
        return result
    def pdf(self, x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        result = self.pdf_(np.asarray(x))
        if isinstance(x, float):
            return result.item()
        return result