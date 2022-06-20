import abc
from copy import deepcopy

import torch.nn as nn


class BaseModel(nn.Module, abc.ABCMeta):
    @abc.abstractmethod
    def get_distributed_data_parallel_model(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def cpu_model(self) -> "BaseModel":
        return deepcopy(self).cpu()

    @abc.abstractmethod
    def requires_model_parallel(self) -> bool:
        return False
