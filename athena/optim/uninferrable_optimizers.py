"""This module consists of optimizer definitions ones could be inferred, because
currently we're unable infer some default values provided by PyTorch. Here listed
listed types that cannot be inferred:
* tuple
* None
TODO: Once these types could be inferred - remove this file.
"""
from typing import Optional, Tuple

from athena.core.dataclasses import dataclass

from athena.optim.optimizer import OptimizerConfig


@dataclass(frozen=True)
class Adam(OptimizerConfig):
    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    amsgrad: bool = False
    maximize: bool = False
    foreach: Optional[bool] = None
    capturable: bool = False
    differentiable: bool = False
    fused: bool = False


@dataclass(frozen=True)
class NAdam(OptimizerConfig):
    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    momentum_decay: float = 4e-3
    maximize: bool = False
    foreach: Optional[bool] = None


@dataclass(frozen=True)
class RAdam(OptimizerConfig):
    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    maximize: bool = False
    foreach: Optional[bool] = None


@dataclass(frozen=True)
class SGD(OptimizerConfig):
    lr: float = 0.001
    momentum: float = 0.0
    weight_decay: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    maximize: bool = False
    foreach: Optional[bool] = None
    differentiable: bool = False


@dataclass(frozen=True)
class AdamW(OptimizerConfig):
    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.01
    amsgrad: bool = False
    maximize: bool = False
    foreach: Optional[bool] = None
    capturable: bool = False


@dataclass(frozen=True)
class SparseAdam(OptimizerConfig):
    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    maximize: bool = False


@dataclass(frozen=True)
class Adamax(OptimizerConfig):
    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    maximize: bool = False
    foreach: Optional[bool] = None


@dataclass(frozen=True)
class LBFGS(OptimizerConfig):
    lr: float = 1
    max_iter: int = 20
    max_eval: Optional[int] = None
    tolerance_grad: float = 1e-07
    tolerance_change: float = 1e-09
    history_size: int = 100
    line_search_fn: Optional[str] = None
    maximize: bool = False


@dataclass(frozen=True)
class Rprop(OptimizerConfig):
    lr: float = 0.01
    etas: Tuple[float, float] = (0.5, 1.2)
    step_sizes: Tuple[float, float] = (1e-06, 50)
    maximize: bool = False
    foreach: Optional[bool] = None
