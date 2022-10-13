import torch

from athena.metrics.functional.utils import nan2num, preprocess_ir


def mrr(y_true: torch.Tensor, y_score: torch.Tensor, topk: int) -> torch.Tensor:
    k = min(y_score.size(1), topk)
    sorted_k_ground_truth = preprocess_ir(y_true, y_score)[:, :k]
    values, indices = torch.max(sorted_k_ground_truth, dim=1)
    indices = indices.type_as(values).unsqueeze(dim=0).t()
    return nan2num(torch.tensor(1.0) / (indices + torch.tensor(1.0)))
