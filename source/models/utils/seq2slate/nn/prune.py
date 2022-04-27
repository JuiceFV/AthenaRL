"""Module describes some constraints logic like sequence masking
or probabilities summarization e.t.c.
"""
import torch
from typing import Tuple


def mask_by_index(input: torch.Tensor, target_input_indcs: torch.Tensor) -> torch.Tensor:
    """Mask out a tensor according to the given indices. Especially it's used 
    to mask already placed items in purpose to set their selection probabilities
    to 0, s.t. we never replace an item if it's placed already.

    masked_input[i][j][k] = -inf if item already attends in the target seq; 
    original value otherwise 

    Args:
        input (torch.Tensor): Input tensor which should be masked.
            shape: batch_size, seq_len, num_of_candidates
        target_input_indcs (torch.Tensor): Indices of input target seqence.
            shape: batch_size, seq_len 


    Returns:
        torch.Tensor: Masked input tensor.
    """
    # Ignore starting/padding symbols
    input[:, :, :2] = float("-inf")
    batch_size, sequence_len = input.shape
    # Getting indicies ones should be masked
    # Overlay lower triangular matrix and choose
    # the elements under main diagonal (include the diagonal)
    mask_indices = torch.tril(
        target_input_indcs.repeat(1, sequence_len).reshape(
            batch_size, sequence_len, sequence_len
        ),
        diagonal=0
    )
    # Fill masked positions with -inf, s.t. softmax(-inf) = 0
    input = input.scatter(2, mask_indices, float("-inf"))
    return input


def subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    """Mask out subsequent positions. Mainly used in the decoding process,
    in which an item should not attend subsequent items.

    mask[i][j][k] = 0 if the item should be ignored; 1 if the item should be paid attention
    Args:
        size (int): Size of the masking sequence.
        device (torch.device): Device where computations happen.

    Returns:
        torch.Tensor: Lower triangular mask-tensor (diagonal elements included).
            shape: 1, seq_len, seq_len
    """
    mask = torch.tril(
        torch.ones(1, size, size, device=device, dtype=torch.bool), diagonal=0
    )
    return mask


def decoder_mask(
    memory: torch.Tensor, target_input_indcs: torch.Tensor, nheads: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the masks used in the PyTorch Transformer-based decoder for
    self-attention and attention over encoder outputs.

    mask[i][j][k] = 1 if the item should be ignored; 0 otherwise.

    Args:
        memory (torch.Tensor): Encoder outputs.
            shape: batch_size, source_seq_len, dim_model
        target_input_indcs (torch.Tensor): Indices of input target seqence.
            (+2 offseted) shape: batch_size, target_seq_len
        nheads (int): Number of transformer heads.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mask over target sequence & mask over encoder output
            target2target_mask shape: batch_size * nheads, target_seq_len, target_seq_len
            target2source_mask shape: batch_size * nheads, target_seq_len, source_seq_len
    """
    device = memory.device
    batch_size, source_seq_len = memory.shape[:2]
    target_seq_len = target_input_indcs.shape[1]
    # Build lower triangular matrix to prevent
    # attention payment to the already aranged items
    # at each time step.
    mask_indices = torch.tril(
        target_input_indcs.repeat(1, target_seq_len).reshape(
            batch_size, target_seq_len, target_seq_len
        ),
        diagonal=0
    ).to(device)
    # Consider start and padding symbol to make target and source lengths equal.
    # And mark masked items in the source sequence as not attractive ones. 'Cause
    # they're already set in the target sequence and they have 0 probability to be
    # choosen in contrast to remaining elements ones would be sampled
    target2source_mask_augmented = torch.zeros(
        batch_size, target_seq_len, source_seq_len + 2, dtype=torch.bool, device=device
    ).scatter(2, mask_indices, 1)
    # Copy the mask for every single head in transformer
    target2source_mask = target2source_mask_augmented[:, :, 2:].repeat_interleave(
        nheads, dim=0
    )
    # As for the target mask we want to mark those items
    # that will represent already aranged output sequence
    # thus self attention represents cross-item attention
    # at each time step.
    target2target_mask = (subsequent_mask(target_seq_len, device) == 0).repeat(
        batch_size * nheads, 1, 1
    )
    return target2target_mask, target2source_mask


def per_item_to_per_seq_probas(
    per_item_probas: torch.Tensor, target_output_inds: torch.Tensor
) -> torch.Tensor:
    """Accumulate items probabilities into overall sequence probability.
    :math:`P_{s} = P_{i_1} x ... x P_{i_n}`

    Args:
        per_item_probas (torch.Tensor): Probability of each symbol in the target_output_inds.
            shape: batch_size, seq_len, num_of_items
        target_output_inds (torch.Tensor): Result arangement indices.
            shape: batch_size, seq_len

    Returns:
        torch.Tensor: Probability of the sequence.
    """
    return torch.clamp(
        torch.prod(
            torch.gather(
                per_item_probas, 2, target_output_inds.unsqueeze(2)
            ).squeeze(1),
            dim=2,
            keepdim=True
        ),
        # Due to torch.log(0) = -inf replace it with a tiny value
        min=1e-40
    )
