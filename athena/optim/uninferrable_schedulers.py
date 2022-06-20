"""This module consists of optimizer definitions ones could be inferred, because
currently we're unable infer some default values provided by PyTorch. Here listed
listed types that cannot be inferred:
* tuple
* None
* required parameters (no default value)
TODO: Once these types could be inferred - remove this file.
"""
from typing import Any, Callable, Dict, List, Optional, Union

from athena.core.dataclasses import dataclass
from athena.optim.scheduler import LRSchedulerConfig


class _LRLambdaMixin(object):
    def eval_lambdas(self, kwargs: Dict[str, Any]) -> None:
        lr_lambda = kwargs.get("lr_lambda")
        if type(lr_lambda) is str:
            kwargs["lr_lambda"] = eval(lr_lambda)


class _ScaleFnLambdaMixin(object):
    def eval_lambdas(self, kwargs: Dict[str, Any]) -> None:
        scale_fn = kwargs.get("scale_fn")
        if type(scale_fn) is str:
            kwargs["scale_fn"] = eval(scale_fn)


@dataclass(frozen=True)
class LambdaLR(_LRLambdaMixin, LRSchedulerConfig):
    lr_lambda: Union[str, Callable[[int], float], List[Callable[[int], float]]]
    last_epoch: int = -1
    verbose: bool = False


@dataclass(frozen=True)
class MultiplicativeLR(_LRLambdaMixin, LRSchedulerConfig):
    lr_lambda: Union[str, Callable[[int], float], List[Callable[[int], float]]]
    last_epoch: int = -1
    verbose: bool = False


@dataclass(frozen=True)
class CyclicLR(_ScaleFnLambdaMixin, LRSchedulerConfig):
    base_lr: Union[float, List[float]]
    max_lr: Union[float, List[float]]
    step_size_up: int = 2000
    step_size_down: Optional[int] = None
    mode: str = "triangular"
    gamma: float = 1.0
    scale_fn: Optional[Union[str, Callable[[int], float]]] = None
    scale_mode: str = "cycle"
    cycle_momentum: bool = True
    base_momentum: float = 0.8
    max_momentum: float = 0.9
    last_epoch: int = -1
    verbose: bool = False


@dataclass(frozen=True)
class StepLR(LRSchedulerConfig):
    step_size: int
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False


@dataclass(frozen=True)
class MultiStepLR(LRSchedulerConfig):
    milestones: List[int]
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False


@dataclass(frozen=True)
class ExponentialLR(LRSchedulerConfig):
    gamma: float
    last_epoch: int = -1
    verbose: bool = False


@dataclass(frozen=True)
class CosineAnnealingLR(LRSchedulerConfig):
    T_max: int
    eta_min: float = 0
    last_epoch: int = -1
    verbose: bool = False


@dataclass(frozen=True)
class OneCycleLR(LRSchedulerConfig):
    max_lr: Union[float, List[float]]
    total_steps: Optional[int] = None
    epochs: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    cycle_momentum: bool = True
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 10000.0
    last_epoch: int = -1
    three_phase: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class CosineAnnealingWarmRestarts(LRSchedulerConfig):
    T_0: int
    T_mult: int = 1
    eta_min: float = 0
    last_epoch: int = -1
    verbose: bool = False
