import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gumbel

from athena.core.config import resolve_defaults


class Sampler(nn.Module):
    """Base sampler class from which one all samplers should be forked.
    """

    def __init__(self) -> None:
        super().__init__()


class SimplexSampler(Sampler):
    r"""
    The sampler picks a vertex of n-dimensional simplex defined as

    .. math::

        \Delta^n = \left\{(\theta_0v_0, \ldots, \theta_nv_n) \in \mathbb{R}^{n + 1}
        | \sum_{i=0}^n{\theta_i} = 1 \text{ and } \theta_i \geq 0 \right\}

    There are two ways to pick a vertex, in purpose to find
    `Fréchet mean <https://en.wikipedia.org/wiki/Fréchet_mean>`_ one minimizes the global
    error:

    1. Greedy way. (i.e. :class:`FrechetSort` with high shape of Frechet Distribution):

        .. math::

            s = \arg \max_{v \in \Delta^n}{(\theta)}

    2. Multinomial sampling (i.e. :class:`FrechetSort` with low shape of Frechet Distribution):

        .. math::

            s = v_j, \qquad
            \arg \min_{j \in [0; n]}{\left\{\sum_{i = 0}^{j}{\theta_i} - X \geq 0 \right\}}

    Where :math:`X \sim U(0, 1)`.

    Examples::

        >>> sampler = SimplexSampler()
        >>> input = torch.Tensor([[[0.1, 0.3, 0.05, 0.07, 0.08, 0.5]]])
        >>> sampler(input, greedy=False)
        (tensor([[0]]), tensor([[0.1000, 0.3000, 0.0500, 0.0700, 0.0800, 0.5000]]))

        >>> sampler = SimplexSampler()
        >>> input = torch.Tensor([[[0.1, 0.3, 0.05, 0.07, 0.08, 0.5]]])
        >>> sampler(input, greedy=True)
        (tensor([[5]]), tensor([[0.1000, 0.3000, 0.0500, 0.0700, 0.0800, 0.5000]]))
    """

    def forward(self, scores: torch.Tensor, greedy: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Get index of a sampled vertex and considered :math:`\theta` distribution.

        Args:
            scores (torch.Tensor): :math:`\theta` distribution.
            greedy (bool): Sampling method.

        Shape:
            - scores: :math:`(B, M, n)`
            - output: :math:`((B, 1), (B, n))`.

        Notations:
            - :math:`B` - batch size.
            - :math:`M` - simplical manifold. It could be simplices distributed over time or whatever you want.
            - :math:`n` - simplex dimensionality.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Chosen vertex & Generative probabilities of last step.
        """
        batch_size = scores.shape[0]
        # Retrieve the last observed probabilities
        probas_dist = scores[:, -1, :]
        if greedy:
            _, vertex = torch.max(probas_dist, dim=1)
        else:
            # idx = min({i in {1, ..., len(probas_dist)}: sum(probas_dist, i) - X >= 0}})
            # Where X ~ U(0,1) and probas_dist sorted descendically.
            vertex = torch.multinomial(probas_dist, num_samples=1, replacement=False)
        vertex = vertex.reshape(batch_size, 1)
        return vertex, probas_dist


class FrechetSort(Sampler):

    @resolve_defaults
    def __init__(
        self, shape: float = 1.0, topk: Optional[int] = None, equiv_len: Optional[int] = None, log_scores: bool = False
    ) -> None:
        super().__init__()
        self.shape = shape
        self.topk = topk
        self.upto = equiv_len
        if topk is not None:
            if equiv_len is None:
                self.upto = topk
            if self.upto > self.topk:
                raise ValueError(f"Equiv length {equiv_len} cannot exceed topk={topk}.")
        self.gumbel_noise = Gumbel(0, 1.0 / shape)
        self.log_scores = log_scores

    def forward(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if scores.dim() != 2:
            raise ValueError("Sample only accepts batches")
        log_scores = scores if self.log_scores else torch.log(scores)
        perturbed = log_scores + self.gumbel_noise.sample(scores.shape)
        action = torch.argsort(perturbed.detach(), descending=True)
        log_proba = self.log_proba(scores, action)
        if self.topk is not None:
            action = action[: self.topk]
        return action, log_proba

    def log_proba(
        self, scores: torch.Tensor, action: torch.Tensor, equiv_len_override: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        upto = self.upto
        if equiv_len_override is not None:
            if equiv_len_override.shape != scores.shape[0]:
                raise ValueError(
                    f"Invalid shape {equiv_len_override.shape}, compared to scores"
                    f"{scores.shape}. equiv_len_override {equiv_len_override}"
                )
            upto = equiv_len_override.long()
            if self.topk is not None and torch.any(equiv_len_override > self.topk):
                raise ValueError(f"Override {equiv_len_override} cannot exceed topk={self.topk}.")

        if len(scores.shape) == 1:
            scores = scores.unsqueeze(0)
            action = action.unsqueeze(0)

        if len(action.shape) != len(scores.shape) != 2:
            raise ValueError("Scores should be batch")
        if action.shape[1] > scores.shape[1]:
            raise ValueError(
                f"Action cardinality ({action.shape[1]}) is larger than the number of scores ({scores.shape[1]})"
            )
        elif action.shape[1] < scores.shape[1]:
            raise NotImplementedError(
                "This semantic is ambiguous. If you have shorter"
                f" slate, pad it with scores.shape[1] ({scores.shape[1]})"
            )

        log_scores = scores if self.log_scores else torch.log(scores)
        n = log_scores.shape[-1]
        log_scores = torch.cat(
            [
                log_scores,
                torch.full((log_scores.shape[0], 1), -math.inf, device=log_scores.device)
            ],
            dim=1
        )
        log_scores = torch.gather(log_scores, 1, action) * self.shape

        p = upto if upto is not None else n

        if isinstance(p, int):
            log_proba = sum(
                torch.nan_to_num(F.log_softmax(log_scores[:, i:], dim=1)[:, 0], neginf=0.0)
                for i in range(p)
            )
        elif isinstance(p, torch.Tensor):
            log_proba = sum(
                torch.nan_to_num(F.log_softmax(log_scores[:, i:], dim=1)[:, 0], neginf=0.0) * (i < p).float()
                for i in range(p)
            )
        else:
            raise RuntimeError(f"p is {p}")

        if torch.any(log_proba.isnan()):
            raise RuntimeError(f"Nan is {log_proba}")
        return log_proba
