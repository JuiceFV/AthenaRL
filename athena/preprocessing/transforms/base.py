from typing import List

import pandas as pd
import torch.nn as nn


class Transformation(nn.Module):
    def __init__(self, keys: List[str]) -> None:
        super().__init__()
        self.keys = keys


class Compose:
    def __init__(self, transformations: List[Transformation]) -> None:
        self.transformations = transformations

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for trans in self.transformations:
            data = trans(data)
        return data
