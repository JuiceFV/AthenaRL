from typing import Dict, Final

import numpy as np
import torch
from source.core.dataclasses import dataclass
from source.core.logger import LoggerMixin

STATIC_KEYS = ["action", "state", "score"]


class SimBuffer(LoggerMixin):
    def __init__(
        self,
        sim_capacity: int = 10000,
        batch_size: int = 1
    ) -> None:
        self._initialized: bool = False
        self._batch_size: int = batch_size
        self._sim_capacity: Final[int] = sim_capacity

        self.add_calls = np.array(0)

        self._storage: Dict[str, torch.Tensor] = {}

    def initialize_storage(self, **kwargs):
        kwargs_keys = set(kwargs.keys())
        if not set(STATIC_KEYS).issubset(kwargs_keys):
            raise ValueError(
                f"{kwargs_keys} doesn't contain all of {STATIC_KEYS}"
            )

    def _add(self, **kwargs):
        pass

    def add(self, **kwargs):
        if not self._initialized:
            self.initialize_storage(**kwargs)

    @property
    def size(self) -> int:
        pass

    @property
    def capacity(self) -> int:
        return self._sim_capacity

    def iter(self) -> int:
        return self.add_calls % self._sim_capacity
