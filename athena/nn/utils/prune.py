import torch

def mask_by_index(input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    r"""
    Mask out a tensor according to the given indices.

    .. note:: 
        
        Masked items are set to :math:`-\infty` so that :func:`torch.nn.functional.softmax` 
        application gives zero probability to pick it.
        
        .. code-block:: python 
        
            >>> masked_input[i][j][k] = float("-inf")
            >>> probas = torch.softmax(masked_input, dim=2)
            >>> probas[i][j][k]
            0

    .. warning::

        Currently input tensor is considered as 3-D tensor, one represents sequential data 
        so for the higher dimension tensors the behaviour is indeterministic. The masking,
        apparently is featurewise (occurs for the last dimension).
        
    Example::

        >>> input = torch.rand(3, 2, 3)
        >>> mask = torch.full((3, 1), 1, dtype=torch.long)
        >>> mask_by_index(input, mask)
        tensor([[[0.4436,   -inf, 0.2505],
                [0.0239, 0.7337, 0.2961]],

                [[0.7820,   -inf, 0.1948],
                [0.8769, 0.5264, 0.6567]],

                [[0.0469,   -inf, 0.3706],
                [0.4661, 0.5410, 0.9782]]])
    
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
    input = input.scatter(2, mask_indices, float("-inf"))
    return input