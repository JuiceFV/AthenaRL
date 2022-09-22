from typing import Any, List, Tuple, Union
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


class ListAggregator(TensorAggregator):
    def __init__(self, field: str) -> None:
        super().__init__(field)
        self.values: List[Any] = []

    def aggregate(self, values: List[Any]):
        self.values.extend(values)


class LastEpochListAggregator(TensorAggregator):
    def __init__(self, field: str) -> None:
        super().__init__(field)
        self.values: List[Any] = []
        self.epoch_values: List[Any] = []

    def aggregate(self, values: torch.Tensor):
        flattened = torch.flatten(values).tolist()
        self.values.extend(flattened)

    def flush(self):
        if self.values:
            self.epoch_values = self.values
            self.values.clear()
