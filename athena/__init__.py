import torch

from athena.version import __version__ as __version__


def gather(data: torch.Tensor, indices_2d: torch.Tensor) -> torch.Tensor:
    """Gather data alongs the second dim. Assume data is 3d with shape 
    :attr:`(batch_size, dim1, dim2)`, and indices_2d's shape is 
    :attr:`(batch_size, dim1)`. The output specified by::

        output[i][j] = data[i][indices_2d[i][j]]

    .. note::

        This function does not require :attr:`data`, output, or :attr:`index_2d` 
        having the same shape, which is mandated by :func:`torch.gather`.

    Args:
        data (torch.Tensor): The data gather from.
            ``shape: batch_size, dim1, dim2``
        indices_2d (torch.Tensor): _description_
            ``shape: batch_size, dim1``

    Raises:
        RuntimeError: If data is not 3-dimensional.

    Returns:

        torch.Tensor: 
            The values of data at given indicies.
            ``shape: batch_size, dim1``
    """
    if len(data.shape) != 3:
        raise RuntimeError("We assume that data is 3-dimensional.")
    device = data.device
    batch_size, data_dim, indices_len = data.shape[0], data.shape[2], indices_2d.shape[1]
    output = data[
        torch.arange(batch_size, device=device).repeat_interleave(
            # index_len has to be moved to the device explicitly, otherwise
            # error will throw during jit.trace
            torch.tensor([indices_len], device=device)
        ),
        indices_2d.flatten(),
    ].view(batch_size, indices_len, data_dim)
    return output
