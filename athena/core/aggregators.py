from typing import List, Tuple, Union
import torch
from athena.core.tracker import Aggregator


class TensorAggregator(Aggregator):
    def __call__(self, field: str, values: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]) -> None:
        values = torch.cat(values, dim=0).cpu()
        return super().__call__(field, values)


class MeanAggregator(TensorAggregator):
    def __init__(self, field: str) -> None:
        super().__init__(field)
        self.values: List[float] = []

    def aggregate(self, values: torch.Tensor) -> None:
        mean = values.mean().item()
        self.info(f"{self.field}: {mean}")
        self.values.append(mean)
