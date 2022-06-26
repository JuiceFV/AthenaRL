import torch

from typing import Tuple


def subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    r"""
    Mask out subsequent positions. Mainly used in the decoding process,
    in which an item should not attend subsequent items.

    .. code-block:: 
        
        mask[i][j][k] = 0 # if the item should be ignored; 
        mask[i][j][k] = 1 # if the item should be paid attention
        
    .. math:: 
    
        \begin{bmatrix}
            1      & 0      & \ldots & 0       \\
            1      & 1      & \ldots & 0       \\
            \vdots & \vdots & \ddots & \vdots  \\
            1      & 1      & \ldots & 1       \\
        \end{bmatrix}
            
    
    Args:
        size (int): Size of the masking sequence.
        device (torch.device): Device where computations happen.
        
    Shape:
        - output: :math:`(1, S, S)`
        
    Notations:
        - :math:`S` - sequence length.

    Returns:
        torch.Tensor: Lower triangular mask-tensor (diagonal elements included).
    """
    mask = torch.tril(
        torch.ones(1, size, size, device=device, dtype=torch.bool), diagonal=0
    )
    return mask


def decoder_mask(
    memory: torch.Tensor, target_input_indcs: torch.Tensor, nheads: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute the masks used in the PyTorch Transformer-based decoder for
    self-attention and attention over encoder outputs.

    .. code-block:: 
        
        mask[i][j][k] = 0 # if the item should be ignored; 
        mask[i][j][k] = 1 # if the item should be paid attention
        
    Example::
    
        >>> memory = torch.rand(1, 10, 512)
        >>> target_input_indcs = torch.randint(0, 10, (1, 5))
        >>> output = decoder_mask(memory, target_input_indcs, nheads=8)
        >>> output[0].shape
        torch.Size([8, 5, 5])
        
        >>> output[1].shape
        torch.Size([8, 5, 10])

    Args:
        memory (torch.Tensor): Encoder outputs.
        target_input_indcs (torch.Tensor): Indices of input target seqence.
        nheads (int): Number of transformer heads.
        
    Shape:
        - memory: :math:`(B, S, d_{model})`
        - target_input_indcs: :math:`(B, T)`
        - output: :math:`((B \times nheads, T, T), (B \times nheads, T, S))`

    Notations:
        - :math:`B` - batch size.
        - :math:`S` - source sequence length.
        - :math:`T` - target sequence length.
        - :math:`d_{model}` - dimension of learnable weights matrix.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mask over target sequence & mask over encoder output.
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
