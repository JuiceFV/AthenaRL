from typing import List

import numpy as np
import torch


def check_consistent_length(*tensors: List[torch.Tensor]) -> None:
    lengths = [tensor.size(0) * tensor.size(1) for tensor in tensors]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Inconsistent numbers of samples")


def preprocess_ir(y_true: torch.Tensor, y_score: torch.Tensor) -> torch.Tensor:
    check_consistent_length(y_true, y_score)
    predicted_order = torch.argsort(y_score, descending=True, dim=-1)
    sorted_ground_truth = torch.gather(y_true, dim=-1, index=predicted_order)
    return sorted_ground_truth


def nan2num(tensor: torch.Tensor, val: float = 0.0) -> torch.Tensor:
    tensor = torch.where(torch.isnan(tensor), torch.ones_like(tensor) * val, tensor)
    return tensor
