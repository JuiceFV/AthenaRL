from typing import Any, Callable, Optional

import torch

from athena.version import __version__ as __version__  # noqa


def gather(data: torch.Tensor, indices_2d: torch.Tensor) -> torch.Tensor:
    r"""
    Gather data alongs the second dim. The output specified by::

        output[i][j] = data[i][indices_2d[i][j]]

    .. warning::

        Assume data is 3d with shape :math:`(B, N, M)`, and ``indices_2d``'s
        shape is :math:`(B, N)`.

    .. note::

        This function does not require ``data``, output, or ``index_2d``
        having the same shape, which is mandated by :func:`torch.gather`.

    Args:
        data (torch.Tensor): The source data gather from.
        indices_2d (torch.Tensor): The indices of elements to gather.

    Shape:
        - data: :math:`(B, N, M)`
        - indices_2d: :math:`(B, N)`
        - output: :math:`(B, N)`

    Notations:
        - :math:`B` - batch size.
        - :math:`N` - first dimension.
        - :math:`M` - second dimension.

    Returns:
        torch.Tensor: The values of data at given indicies.
    """
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


class lazy_property:

    def __init__(self, retreiver: Callable[[Optional[object]], Any]) -> None:
        self._retreiver = retreiver
        self.__doc__ = retreiver.__doc__
        self.__name__ = retreiver.__name__

    def __get__(self, __o: Optional[object], __t: type) -> Optional[Any]:
        if __o is None:
            return None
        value = self._retreiver(__o)
        setattr(__o, self.__name__, value)
        return value
