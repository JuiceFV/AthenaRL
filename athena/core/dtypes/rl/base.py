import torch
from typing import Optional, Dict
from dataclasses import fields
from athena.core.dataclasses import dataclass
from enum import Enum

from athena.core.base_dclass import BaseDataClass
from athena.core.enum_meta import AthenaEnumMeta
from athena.core.dtypes.base import TensorDataClass


class IPSBlurMethod(Enum, metaclass=AthenaEnumMeta):
    UNIVERSAL = "universal"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class IPSBlur(BaseDataClass):
    blur_method: IPSBlurMethod
    blur_max: float


@dataclass
class ExtraData(TensorDataClass):
    mdp_id: Optional[torch.Tensor] = None
    sequence_number: Optional[torch.Tensor] = None
    actions_probability: Optional[torch.Tensor] = None
    max_num_actions: Optional[torch.Tensor] = None

    @classmethod
    def from_dict(cls, d: Dict[str, torch.Tensor]) -> "ExtraData":
        return cls(**{f.name: d.get(f.name, None) for f in fields(cls)})
