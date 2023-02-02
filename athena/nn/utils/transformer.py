import torch

from typing import Tuple

PADDING_SYMBOL = 0
DECODER_START_SYMBOL = 1


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


def padding_mask(indcs: torch.Tensor, padding_symbol: int = PADDING_SYMBOL) -> torch.Tensor:
    r"""
    Mask out padding positions. Commonly, a padding mask is combined with an additive mask.
    Thus it has to be 3-dimensional.

    .. code-block::

        mask[i][0][j] = 0 # if the item should be ignored;
        mask[i][0][j] = 1 # if the item should be paid attention

    Args:
        indcs (torch.Tensor): Indices of a sequence.
        padding_symbol (int, optional): An index is treated as padding.
          Defaults to PADDING_SYMBOL.

    Shape:
        - indcs: :math:`(B, S)`
        - output: :math:`(B, 1, S)`

    Notations:
        - :math:`B` - batch size.
        - :math:`S` - sequence length.

    Returns:
        torch.Tensor: Padding mask-tensor.
    """
    mask = (indcs != padding_symbol).unsqueeze(-2).type(torch.bool)
    return mask


def subsequent_and_padding_mask(
    target_input_indcs: torch.Tensor,
    padding_symbol: int = PADDING_SYMBOL
) -> torch.Tensor:
    """_summary_

    Args:
        target_input_indcs (torch.Tensor): _description_
        padding_symbol (int, optional): _description_. Defaults to PADDING_SYMBOL.

    Returns:
        torch.Tensor: _description_
    """
    padd_mask = padding_mask(target_input_indcs, padding_symbol)
    subseq_mask = subsequent_mask(target_input_indcs.size(-1), target_input_indcs.device)
    target2target_mask = padd_mask & subseq_mask
    return target2target_mask


def encoder_mask(
    source_input_indcs: torch.Tensor,
    nheads: int,
    padding_symbol: int = PADDING_SYMBOL
) -> torch.Tensor:
    """_summary_

    Args:
        source_input_indcs (torch.Tensor): _description_
        nheads (int): _description_
        padding_symbol (int, optional): _description_. Defaults to PADDING_SYMBOL.

    Returns:
        torch.Tensor: _description_
    """
    device = source_input_indcs.device
    source_seq_len = source_input_indcs.shape[1]
    source2source_mask = torch.ones(1, source_seq_len, source_seq_len, device=device, dtype=torch.bool)
    padd_mask = padding_mask(source_input_indcs, padding_symbol)
    source2source_mask = ((source2source_mask & padd_mask) == 0).repeat_interleave(nheads, dim=0)
    return source2source_mask


def mask_by_index(input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    r"""
    Mask out a tensor according to the given indices.

    .. warning::

        Currently input tensor is considered as 3-D tensor, one represents sequential data
        so for the higher dimension tensors the behaviour is indeterministic. The masking,
        apparently is featurewise (occurs for the last dimension).

    Args:
        input (torch.Tensor): Input tensor which should be masked.
        indices (torch.Tensor): Indices of input target seqence.

    Shape:
        - input: :math:`(B, S, I)`
        - indices: :math:`(B, S)`

    Notations:
        - :math:`B` - batch size.
        - :math:`S` - sequence length.
        - :math:`I` - item's vector dimensionality.

    Returns:
        torch.Tensor: Masked input tensor.
    """
    batch_size, size = indices.shape
    # Getting indicies ones should be masked
    # Overlay lower triangular matrix and choose
    # the elements under main diagonal (include the diagonal)
    mask_indices = torch.tril(
        indices.repeat(1, size).reshape(batch_size, size, size),
        diagonal=0
    )
    # Fill masked positions with -inf, s.t. softmax(-inf) = 0
    input = input.scatter(2, mask_indices, 1)
    return input


def decoder_mask(
    memory: torch.Tensor,
    target_input_indcs: torch.Tensor,
    nheads: int,
    padding_symbol: int = PADDING_SYMBOL,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute the masks used in the PyTorch Transformer-based decoder for
    self-attention and attention over encoder outputs.

    .. code-block::

        mask[i][j][k] = 1 # if the item should be ignored;
        mask[i][j][k] = 0 # if the item should be paid attention

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
        padding_symbol (int, optional): An index is treated as padding.
          Defaults to PADDING_SYMBOL.

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
    # at each time step. Zero-index is also treated
    # as being ignored due to it's padding symbol
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
    target2target_mask = (
        subsequent_and_padding_mask(target_input_indcs, padding_symbol) == 0
    ).repeat(nheads, 1, 1)
    return target2target_mask, target2source_mask
