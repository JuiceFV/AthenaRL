from typing import Tuple
import torch
from athena.core.dtypes import Ftype
from athena.core.parameters import NormalizationParams


def fake_norm():
    return NormalizationParams(ftype=Ftype.CONTINUOUS, mean=0.0, stdev=1.0)


def apply_variable_slate_size(
    input_prototype: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], length: int
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    state_proto, cand_proto = input_prototype
    cand_proto = (
        cand_proto[0][:, :1, :].repeat(1, length, 1),
        cand_proto[1][:, :1, :].repeat(1, length, 1),
    )
    return (
        (torch.rand_like(state_proto[0]), torch.ones_like(state_proto[1])),
        (
            torch.rand_like(cand_proto[0]),
            torch.ones_like(cand_proto[1])
        )
    )
