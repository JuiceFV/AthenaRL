import abc

import torch
from typing_extensions import overload

from athena.core.registry import RegistryMeta


class MetricBase(metaclass=RegistryMeta):
    @abc.abstractmethod
    def compute(self, *args, **kwargs) -> torch.Tensor:
        pass

    @overload
    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.compute(*args, **kwargs)
