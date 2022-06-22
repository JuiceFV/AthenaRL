import torch


def prod_probas(
    probas: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Accumulate items probabilities into overall sequence probability.
    :math:`P_{s} = P_{i_1} x ... x P_{i_n}`

    Args:
        probas (torch.Tensor): Probability of each symbol in the target_output_inds.
            shape: batch_size, seq_len, num_of_items
        indices (torch.Tensor): Result arangement indices.
            shape: batch_size, seq_len

    Returns:
        torch.Tensor: Probability of the sequence.
    """
    return torch.clamp(
        torch.prod(
            torch.gather(probas, 2, indices.unsqueeze(2)).squeeze(1),
            dim=2,
            keepdim=True
        ),
        # Due to torch.log(0) = -inf replace it with a tiny value
        min=1e-40
    )
