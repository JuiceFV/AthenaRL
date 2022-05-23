from typing import Optional

from source.core.base_dclass import BaseDataClass
from source.core.dataclasses import dataclass

@dataclass(frozen=True)
class TransformerParams(BaseDataClass):
    nheads: int = 1
    dim_model: int = 64
    dim_feedforward: int = 32
    nlayers: int = 2
    latent_state_embed_dim: Optional[int] = None