import inspect
from typing import Dict, List, Union, Iterable, Any

import torch
from source.core.dataclasses import dataclass, field
from source.core.registry import RegistryMeta

from source.optim.scheduler import LRSchedulerConfig
from source.optim.utils import is_torch_optimizer


@dataclass(frozen=True)
class OptimizerConfig(metaclass=RegistryMeta):
    lr_schedulers: List[LRSchedulerConfig] = field(default_factory=list)

    def create_optimizer_scheduler(
        self, params: Iterable[Union[torch.Tensor, Dict[str, Any]]]
    ) -> Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]]:
        if len(self.lr_schedulers) > 1:
            raise ValueError(
                "Multiple schedulers for one optimizer not supported yet")
        optimizer_class = getattr(torch.optim, type(self).__name__)
        if not is_torch_optimizer(optimizer_class):
            raise TypeError(f"{optimizer_class} is not an optimizer.")
        kwargs = {
            kwarg: getattr(self, kwarg)
            for kwarg in inspect.signature(optimizer_class).parameters
            if kwarg != "params"
        }
        optimizer = optimizer_class(params=params, **kwargs)
        if len(self.lr_schedulers) == 0:
            return {"optimizer": optimizer}
        else:
            lr_scheduler = self.lr_schedulers[0].create_from_optimizer(
                optimizer)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
