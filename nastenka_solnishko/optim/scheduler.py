"""
"""
import inspect
import torch
from typing import Dict, Any
from nastenka_solnishko.core.dataclasses import dataclass
from nastenka_solnishko.core.registry import RegistryMeta

from nastenka_solnishko.optim.utils import is_torch_lr_scheduler


@dataclass(frozen=True)
class LRSchedulerConfig(metaclass=RegistryMeta):
    def create_from_optimizer(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        lr_class = getattr(
            torch.optim.lr_scheduler, type(self).__name__
        )
        if not is_torch_lr_scheduler(lr_class):
            TypeError(f"{lr_class} is not a scheduler.")

        kwargs = {
            kwarg: getattr(self, kwarg)
            for kwarg in inspect.signature(lr_class).parameters
            if kwarg != "optimizer"
        }

        self.eval_lambdas(kwargs)

        return lr_class(optimizer=optimizer, **kwargs)

    def eval_lambdas(self, kwargs: Dict[str, Any]) -> None:
        """To allow string-based configuration, we need decoder to convert
        from strings to Callables.
        """
        pass
