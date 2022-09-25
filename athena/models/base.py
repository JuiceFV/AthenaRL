import abc
from copy import deepcopy
from typing import Any, Optional

import torch.nn as nn

from athena.core.dtypes.base import ModelFeatureConfig


class BaseModel(nn.Module):
    @abc.abstractmethod
    def input_prototype(self) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def get_distributed_data_parallel_model(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def cpu_model(self) -> "BaseModel":
        return deepcopy(self).cpu()

    @abc.abstractmethod
    def requires_model_parallel(self) -> bool:
        return False

    @abc.abstractmethod
    def feature_config(self) -> Optional[ModelFeatureConfig]:
        return None
