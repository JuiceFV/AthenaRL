from typing import Callable, Dict, List

import torch

from athena.preprocessing.transforms.base import Transformation


class Lambda(Transformation):
    def __init__(self, keys: List[str], fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__(keys)
        self.fn = fn

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key in self.keys:
            data[key] = self.fn(data[key])
        return data


class FlattenSlateView(Transformation):
    def __init__(self, keys: List[str], candidate_dim: int) -> None:
        super().__init__(keys)
        self.candidate_dim = candidate_dim

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key in self.keys:
            data[key] = data[key].view(-1, self.candidate_dim)
        return data


class UnflattenSlateView(Transformation):
    def __init__(self, keys: List[str], dim_size: int, is_slate_dim: bool = False) -> None:
        super().__init__(keys)
        self.dim_size = dim_size
        self.axis = int(is_slate_dim)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        shape = [-1, -1, -1]
        for key in self.keys:
            value = data[key]
            shape[-1] = value.shape[1]
            shape[self.axis] = self.dim_size
            data[key] = data[key].view(*shape)
        return data
