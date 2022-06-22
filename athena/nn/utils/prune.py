import torch

def mask_by_index(input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Mask out a tensor according to the given indices. Especially it's used 
    to mask already placed items in purpose to set their selection probabilities
    to 0, s.t. we never replace an item if it's placed already.

    masked_input[i][j][k] = -inf if item already attends in the target seq; 
    original value otherwise 

    Args:
        input (torch.Tensor): Input tensor which should be masked.
            shape: batch_size, seq_len, num_of_candidates
        indices (torch.Tensor): Indices of input target seqence.
            shape: batch_size, seq_len 


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