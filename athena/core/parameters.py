from typing import Optional

import athena.core.dtypes as adt
from athena.core.base_dclass import BaseDataClass
from athena.core.config import param_hash
from athena.core.dataclasses import dataclass


@dataclass(frozen=True)
class TransformerParams(BaseDataClass):
    nheads: int = 1
    dim_model: int = 64
    dim_feedforward: int = 32
    nlayers: int = 2
    latent_state_embed_dim: Optional[int] = None


@dataclass(frozen=True)
class Seq2SlateParameters(BaseDataClass):
    on_policy: bool = True
    version: adt.Seq2SlateVersion = adt.Seq2SlateVersion.REINFORCEMENT_LEARNING
    ips_blur: Optional[adt.IPSBlur] = None
