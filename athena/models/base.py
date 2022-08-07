import abc
from copy import deepcopy
from typing import Any

import torch.nn as nn


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
