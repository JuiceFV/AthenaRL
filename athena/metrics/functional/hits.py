import torch

from athena.metrics.functional.utils import nan2num, preprocess_ir


def hit(y_true: torch.Tensor, y_score: torch.Tensor, topk: int, hit_if_empty: bool = False) -> torch.Tensor:
    sorted_ground_truth = preprocess_ir(y_true, y_score)
    k = min(y_score.size(1), topk)
    hit_score = torch.sum(sorted_ground_truth[:, :k], dim=1) / y_true.sum(dim=1)
    return nan2num(hit_score, float(hit_if_empty))
