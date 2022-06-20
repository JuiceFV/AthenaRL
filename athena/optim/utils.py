import inspect

import torch


def is_strict_subclass(_cls: object, _base_cls: object) -> bool:
    """Check if a class is exactly a subclass of its base.

    Args:
        _cls (object): Tentatively subclass.
        _base_cls (object): Base class.

    Returns:
        bool: Is the _cls exactly the subclass of the base.
    """
    if not inspect.isclass(_cls) or not inspect.isclass(_base_cls):
        return False
    return issubclass(_cls, _base_cls) and _cls != _base_cls


def is_torch_optimizer(cls: object) -> bool:
    """Check if optimizer is torch optimizer.

    Returns:
        bool: If an optimizer is instantiated from torch optimizer.
    """
    return is_strict_subclass(cls, torch.optim.Optimizer)


def is_torch_lr_scheduler(cls: object) -> bool:
    """Check if LR scheduler is torch LR scheduler.

    Returns:
        bool: If an scheduler is instantiated from torch scheduler.
    """
    return is_strict_subclass(cls, torch.optim.lr_scheduler._LRScheduler)
