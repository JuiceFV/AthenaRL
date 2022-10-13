import torch

from athena.metrics.functional.utils import nan2num, preprocess_ir


def dcg(y_true: torch.Tensor, y_score: torch.Tensor) -> torch.Tensor:
    sorted_ground_truth = preprocess_ir(y_true, y_score)
    device = sorted_ground_truth.device
    def gain_function(x: torch.Tensor) -> torch.Tensor: return torch.pow(2, x) - 1
    gains = gain_function(sorted_ground_truth)
    discounts = torch.tensor(1) / torch.log2(
        torch.arange(sorted_ground_truth.shape[1], dtype=torch.float, device=device) + 2.0
    )
    dcg_score = gains * discounts
    return dcg_score


def ndcg(y_true: torch.Tensor, y_score: torch.Tensor, topk: int) -> torch.Tensor:
    ideal_dcgs = dcg(y_true, y_true)[:, :topk]
    predicted_dcgs = dcg(y_true, y_score)[:, :topk]
    ndcg_score = torch.sum(predicted_dcgs, dim=1) / torch.sum(ideal_dcgs, dim=1)
    return nan2num(ndcg_score)
