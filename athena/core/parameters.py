from typing import Dict, List, Optional

import torch.nn as nn

import athena.core.dtypes as adt
from athena.core.base_dclass import BaseDataClass
from athena.core.config import param_hash
from athena.core.dataclasses import dataclass

SortedList = List[float]
EnumValues = List[int]


@dataclass(frozen=True)
class TransformerParams(BaseDataClass):
    nheads: int = 1
    dim_model: int = 64
    dim_feedforward: int = 32
    nlayers: int = 2
    state_embed_dim: Optional[int] = None


@dataclass(frozen=True)
class Seq2SlateParams(BaseDataClass):
    on_policy: bool = True
    version: adt.Seq2SlateVersion = adt.Seq2SlateVersion.REINFORCEMENT_LEARNING
    ips_blur: Optional[adt.IPSBlur] = None


@dataclass(frozen=True)
class NormalizationParams(BaseDataClass):
    __hash__ = param_hash

    ftype: adt.Ftype
    boxcox_lambda: Optional[float] = None
    boxcox_shift: Optional[float] = None
    mean: Optional[float] = None
    stdev: Optional[float] = None
    possible_values: Optional[EnumValues] = None
    quantiles: Optional[SortedList] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass(frozen=True)
class NormalizationData(BaseDataClass):
    __hash__ = param_hash
    dense_normalization_params: Dict[int, NormalizationParams]


class NormalizationKey:
    STATE = "state"
    ACTIONS = "actions"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class EvaluationParams(BaseDataClass):
    cpe: bool = False
    propensity_network: Optional[nn.Module] = None
