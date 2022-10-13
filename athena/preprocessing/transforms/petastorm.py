from typing import Callable, List

import numpy as np
import pandas as pd

from athena.preprocessing.transforms.base import Transformation


class Lambda(Transformation):
    def __init__(self, keys: List[str], fn: Callable[[pd.Series], pd.Series]) -> None:
        super().__init__(keys)
        self.fn = fn

    def forward(self, data: pd.DataFrame) -> pd.DataFrame:
        for key in self.keys:
            data[key] = self.fn(data[key])
        return data


class VectorPadding(Transformation):
    def __init__(self, keys: List[str], max_lengths: List[int]) -> None:
        super().__init__(keys)
        if len(keys) != len(max_lengths):
            raise RuntimeError(
                f"Expected max_length for each vector column; Got {len(keys)} != {len(max_lengths)}"
            )
        self.max_lengths = max_lengths

    def forward(self, data: pd.DataFrame) -> pd.DataFrame:
        for key, max_len in zip(self.keys, self.max_lengths):
            data[key] = data[key].map(
                lambda vec: np.pad(
                    vec, (0, max_len - len(vec)),
                    'constant', constant_values=vec.dtype.type(0)
                )
            )
        return data
