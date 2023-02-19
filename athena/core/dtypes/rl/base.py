import torch
from typing import Optional, Dict
from dataclasses import fields
from athena.core.dataclasses import dataclass
from enum import Enum

from athena.core.base_dclass import BaseDataClass
from athena.core.enum_meta import AthenaEnumMeta
from athena.core.dtypes.base import TensorDataClass


class IPSBlurMethod(Enum, metaclass=AthenaEnumMeta):
    r"""
    Importance Sampling bluring method.
    """
    #: Restricts highest magnitude with a maximum possible.
    UNIVERSAL = "universal"
    #: Blends highest magnitude with 0.
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class IPSBlur(BaseDataClass):
    r"""
    Importance Sampling bluring configuration.
    """
    #: Bluring method.
    blur_method: IPSBlurMethod
    #: Highest admittable policy difference.
    blur_max: float


@dataclass
class ExtraData(TensorDataClass):
    r"""
    Exta data is used in marginal cases or for the data comprehension.
    """
    #: Unique episode id.
    mdp_id: Optional[torch.Tensor] = None
    #: Episode's iteration.
    sequence_number: Optional[torch.Tensor] = None
    #: Action probability is used in Temporal Difference algorithms.
    actions_probability: Optional[torch.Tensor] = None
    #: Maximum number of actions an agent capable to commit.
    max_num_actions: Optional[torch.Tensor] = None

    @classmethod
    def from_dict(cls, d: Dict[str, torch.Tensor]) -> "ExtraData":
        return cls(**{f.name: d.get(f.name, None) for f in fields(cls)})
