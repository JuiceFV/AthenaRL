"""Every torch optimizer is stored in optimizer factory s.t.
the only optimizer could be defined at once.

Usage:

Once you want to use an optimizer you need to define the 
`OptimizerRoster` one is registered via `OptimizerConfig.register()`.
After register you should define an optimizer you're wanna to use:
E.g.
>>> class ModelTrainer:
>>>     def __init__(
>>>         self,
>>>         optimizer_factory: OptimizerRoster = field(default_factory=OptimizerRoster.default)
>>>         ...
>>>     ):
>>>         self.opt_factory = optimizer_factory
>>>         ...

To specify non-default optimizer you have to be sure that the optimizer
is instatiated from `torch.optim.Optimizer` and inferrable by dataclasses
otherwise exception appears"
E.g.
>>> from source.optim.optimizer_roster import opt_classes
>>> optimizer = OptimizerRoster(SGD=opt_classes["SGD"](lr=0.01))
>>> ...

Normally, the training process is configured with YAML config file.
YAML optimizer definition looks as following:
E.g.
>>> model:
>>>   Seq2Slate:
>>>     trainer_param:
>>>       ...
>>>       minibatch_size: 512
>>>       optimizer:
>>>         SGD:
>>>           lr: 0.01
>>>           ...
>>>           lr_schedulers:
>>>               - OneCycleLR:
>>>                   ...

According to the example above, since we don't know which network param
we want to optimize we build optimizer specified in the factory. Optimizer
creation looks like follows:
E.g.
>>> class ModelTrainer:
>>>     ...
>>>     def configure_optimizers(self):
>>>         optimizers = []
>>>         optimizers.append(
>>>             self.opt_factory.make_optimizer_scheduler(model.parameters())
>>>         )
"""
import inspect
from typing import Dict, List, Union, Iterable, Any

import torch
from athena.core.dataclasses import dataclass, field
from athena.core.registry import RegistryMeta

from athena.optim.scheduler import LRSchedulerConfig
from athena.optim.utils import is_torch_optimizer


@dataclass(frozen=True)
class OptimizerConfig(metaclass=RegistryMeta):
    """Base optimizer config mold.
    It creates optimizer scheduler if any is passed for any PyTorch optimizer.
    """
    lr_schedulers: List[LRSchedulerConfig] = field(default_factory=list)

    def create_optimizer_scheduler(
        self, params: Iterable[Union[torch.Tensor, Dict[str, Any]]]
    ) -> Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]]:
        if len(self.lr_schedulers) > 1:
            raise ValueError(
                "Multiple schedulers for one optimizer not supported yet"
            )
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
                optimizer
            )
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
