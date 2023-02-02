import torch


def prod_probas(
    probas: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    r"""
    Accumulate items probabilities into overall sequence probability.

    .. math::

        P(s) = \prod_{j=1}^{|s|}{P(i_j)}

    Args:
        probas (torch.Tensor): Probability of each symbol to be placed.
        indices (torch.Tensor): Result arangement indices.

    Shape:
        - probas: :math:`(B, S, I)`
        - indices: :math:`(B, 1)`

    Notations:
        - :math:`B` - batch size.
        - :math:`S` - length of sequence.
        - :math:`I` - number of entities which form probability distribution.

    Returns:
        torch.Tensor: Probability of the sequence.
    """
    return torch.clamp(
        torch.prod(
            torch.gather(probas, 2, indices.unsqueeze(2)).squeeze(2),
            dim=1,
            keepdim=True
        ),
        # Due to torch.log(0) = -inf replace it with a tiny value
        min=1e-40
    )
