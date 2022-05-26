"""The basic torch optimizers or instantiated from ones are wrapped with pydantic dataclass
around. Besides, these optimizers form the registry where the only could be picked.
Since we don't know which network parameters we want to optimize, we create a factory 
one contains following optimizers: Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, 
ASGD, SGD, RAdam, Rprop, RMSprop, NAdam, LBFGS. Due to the factory bases on class implementation
one is specified via YAML config, some of these optimizers were redefined s.t. we can handle
their parameters' type like `tuple` and `None`.
"""
from typing import Any, Dict, Iterable, List, Union

import source.optim.uninferrable_optimizers as cbi
import torch
from source.core.config import create_config_class, param_hash
from source.core.discriminated_union import DiscriminatedUnion
from source.optim.optimizer import OptimizerConfig
from source.optim.utils import is_torch_optimizer


def retrieve_torch_optimizers() -> List[str]:
    return [
        optimizer
        for optimizer in dir(torch.optim)
        if is_torch_optimizer(getattr(torch.optim, optimizer))
    ]


_opt_classes = {}
for opt_name in retrieve_torch_optimizers():
    if hasattr(cbi, opt_name):
        subclass = getattr(cbi, opt_name)
    else:
        opt_class = getattr(torch.optim, opt_name)

        subclass = type(
            opt_name,
            (OptimizerConfig,),
            {}
        )
        create_config_class(opt_class, blocklist=["params"])(subclass)

    subclass.__hash__ = param_hash
    _opt_classes[opt_name] = subclass


@OptimizerConfig.register()
class OptimizerRoster(DiscriminatedUnion):
    @classmethod
    def default(cls, **kwargs) -> "OptimizerRoster":
        return cls(Adam=_opt_classes["Adam"](**kwargs))

    def create_optimizer_scheduler(
        self, params: Iterable[Union[torch.Tensor, Dict[str, Any]]]
    ) -> Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]]:
        return self.value.create_optimizer_scheduler(params)
