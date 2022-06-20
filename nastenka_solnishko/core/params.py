from typing import Optional

from nastenka_solnishko.core.config import param_hash
from nastenka_solnishko.core.base_dclass import BaseDataClass
from nastenka_solnishko.core.dataclasses import dataclass

@dataclass(frozen=True)
class TransformerParams(BaseDataClass):
    nheads: int = 1
    dim_model: int = 64
    dim_feedforward: int = 32
    nlayers: int = 2
    latent_state_embed_dim: Optional[int] = None