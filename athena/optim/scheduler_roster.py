from typing import List, Dict

import athena.optim.uninferrable_schedulers as cbi
import torch
from athena.core.config import create_config_class, param_hash
from athena.core.registry import DiscriminatedUnion

from athena.optim.scheduler import LRSchedulerConfig
from athena.optim.utils import is_torch_lr_scheduler


def retrieve_torch_lr_schedulers() -> List[str]:
    return [
        lr_scheduler
        for lr_scheduler in dir(torch.optim.lr_scheduler)
        if is_torch_lr_scheduler(getattr(torch.optim.lr_scheduler, lr_scheduler))
    ]


lr_scheduler_classes: Dict[str, torch.optim.lr_scheduler._LRScheduler] = {}
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
