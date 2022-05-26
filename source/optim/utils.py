import inspect

import torch


def is_strict_subclass(_cls: object, _base_cls: object):
    if not inspect.isclass(_cls) or not inspect.isclass(_base_cls):
        return False
    return issubclass(_cls, _base_cls) and _cls != _base_cls


def is_torch_optimizer(cls):
    return is_strict_subclass(cls, torch.optim.Optimizer)


def is_torch_lr_scheduler(cls):
    return is_strict_subclass(cls, torch.optim.lr_scheduler._LRScheduler)
