import torch
import torch.nn as nn

from typing import Tuple

class SimplexSampler(nn.Module):
    r"""
    The sampler picks a vertex of n-dimensional simplex defined as
    
    .. math::
    
        \Delta^n = \left\{(\theta_0v_0, \ldots, \theta_nv_n) \in \mathbb{R}^{n + 1}
        | \sum_{i=0}^n{\theta_i} = 1 \text{ and } \theta_i \geq 0 \right\}
        
    There are two ways to pick a vertex, in purpose to find 
    `Fréchet mean <https://en.wikipedia.org/wiki/Fréchet_mean>`_ one minimizes the global
    error:
    
    1. Greedy way. At each time step we pick most probable vertex:

        .. math::

            s = \arg \max_{v \in \Delta^n}{(\theta)}

    2. Multinomial sampling:
    
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

    def forward(self, probas: torch.Tensor, greedy: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Get index of a sampled vertex and considered :math:`\theta` distribution.

        Args:
            probas (torch.Tensor): :math:`\theta` distribution.
            greedy (bool): Sampling method.
            
        Shape:
            - probas: :math:`(B, M, n)`
            - output: :math:`((B, 1), (B, n))`.
            
        Notations:
            - :math:`B` - batch size.
            - :math:`M` - simplical manifold. It could be simplices distributed over time or whatever you want.
            - :math:`n` - simplex dimensionality.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Chosen vertex & Generative probabilities of last step.
        """
        batch_size = probas.shape[0]
        # Retrieve the last observed probabilities
        probas_dist = probas[:, -1, :]
        if greedy:
            _, candidate = torch.max(probas_dist, dim=1)
        else:
            # idx = min({i in {1, ..., len(probas_dist)}: sum(probas_dist, i) - X >= 0}})
            # Where X ~ U(0,1) and probas_dist sorted descendically.
            candidate = torch.multinomial(probas_dist, num_samples=1, replacement=False)
        candidate = candidate.reshape(batch_size, 1)
        return candidate, probas_dist