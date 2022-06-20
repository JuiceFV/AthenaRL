from typing import List

import nastenka_solnishko.optim.uninferrable_schedulers as cbi
import torch
from nastenka_solnishko.core.config import create_config_class, param_hash
from nastenka_solnishko.core.discriminated_union import DiscriminatedUnion

from nastenka_solnishko.optim.scheduler import LRSchedulerConfig
from nastenka_solnishko.optim.utils import is_torch_lr_scheduler


def retrieve_torch_lr_schedulers() -> List[str]:
    return [
        lr_scheduler
        for lr_scheduler in dir(torch.optim.lr_scheduler)
        if is_torch_lr_scheduler(getattr(torch.optim.lr_scheduler, lr_scheduler))
    ]


lr_scheduler_classes = {}
for lr_scheduler_name in retrieve_torch_lr_schedulers():
    if hasattr(cbi, lr_scheduler_name):
        subclass = getattr(cbi, lr_scheduler_name)
    else:
        lr_scheduler_class = getattr(
            torch.optim.lr_scheduler, lr_scheduler_name
        )
        subclass = type(
            lr_scheduler_name,
            (LRSchedulerConfig,),
            {"__module__": __name__},
        )
        create_config_class(
            lr_scheduler_class, blocklist=["optimizer"]
        )(subclass)

    subclass.__hash__ = param_hash
    lr_scheduler_classes[lr_scheduler_name] = subclass


@LRSchedulerConfig.register()
class LRSchedulerRoster(DiscriminatedUnion):
    pass
