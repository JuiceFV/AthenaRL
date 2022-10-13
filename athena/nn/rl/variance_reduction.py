from typing import Optional

import torch
import torch.nn as nn

import athena.core.dtypes as adt


class BaselineNetwork(nn.Module):
    def __init__(self, state_dim: int, dim_feedforward: int, nlayers: int) -> None:
        super().__init__()
        layers = [nn.Linear(state_dim, dim_feedforward), nn.ReLU()]
        if nlayers < 1:
            raise ValueError("Zero-layer perceptron? are you fkn kidding me?")
        for _ in range(nlayers - 1):
            layers.extend([nn.Linear(dim_feedforward, dim_feedforward), nn.ReLU()])
        layers.append(nn.Linear(dim_feedforward, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, input: adt.PreprocessedRankingInput) -> torch.Tensor:
        x = input.state.dense_features
        return self.mlp(x)


def ips_ratio(dist1: torch.Tensor, dist2: torch.Tensor) -> torch.Tensor:
    r"""
    `Monte Carlo Methods and Importance Sampling
    <https://ib.berkeley.edu/labs/slatkin/eriq/classes/guest_lect/mc_lecture_notes.pdf>`_

    Args:
        dist1 (torch.Tensor): _description_
        dist2 (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    if dist1.shape != dist2.shape or len(dist1.shape) != 2 or dist1.shape[1] != 1:
        ValueError(f"Inapropriate distributions shapes. {dist1.shape} {dist2.shape}")
    return dist1 / dist2


def ips_blur(importance_weights: torch.Tensor, ips_blur: Optional[adt.IPSBlur]) -> torch.Tensor:
    if not ips_blur:
        return importance_weights.clone()
    if ips_blur.blur_method == adt.IPSBlurMethod.UNIVERSAL:
        return torch.clamp(importance_weights, 0, ips_blur.blur_max)
    elif ips_blur.blur_method == adt.IPSBlurMethod.AGGRESSIVE:
        return torch.where(
            importance_weights > ips_blur.blur_max, torch.zeros_like(importance_weights), importance_weights
        )
