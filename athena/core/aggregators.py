from typing import Any, Callable, Dict, List, Tuple, Union

import torch

from athena.core.tensorboard import SummaryWriterContext
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


class TensorboardHistogramAndMeanAggregator(TensorAggregator):
    def __init__(self, field: str, log_key: str) -> None:
        super().__init__(field)
        self.log_key = log_key

    def aggregate(self, values: torch.Tensor) -> None:
        if len(values.shape) != 1 and (len(values.shape) != 2 or values.shape[1] != 1):
            raise RuntimeError(f"Unexpected shape for {self.field}: {values.shape}")
        try:
            SummaryWriterContext.add_histogram(self.log_key, values)
            SummaryWriterContext.add_scalar(f"{self.log_key}/mean", values.mean())
        except ValueError:
            raise ValueError(
                f"Cannot create histogram for key: {self.log_key}; "
                f"this is likely because you have NULL value in your input; value: {values}"
            )


class LambdaAggregator(TensorAggregator):
    def __init__(self, field: str, log_keys: List[str], funcs: Dict[str, Callable]) -> None:
        super().__init__(field)
        self.log_keys = log_keys
        self.values: Dict[str, Dict[str, List[float]]] = {
            func: {key: [] for key in self.log_keys}
            for func in funcs
        }
        self.funcs = funcs

    def aggregate(self, values: torch.Tensor):
        for func_name, func in self.funcs.items():
            agg_values = func(values, dim=0)
            for log_key, value in zip(self.log_keys, agg_values):
                value = value.item()
                self.values[func_name][log_key].append(value)
