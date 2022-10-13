import torch
import torch.nn as nn


class Embedding(nn.Module):
    r"""
    Learnable embeddings ones make model able to learn.
    Thecnically, we apply linear transformation to given
    vector projecting it to a fixed dimension :math:`e = xW^T + b`.

    Args:
        in_features (int): The dimensionality of the original vector.
        out_features (int): The row-wise dimensionality of learnable weight matrix.

    Example::

        >>> embed = Embedding(20, 30)
        >>> input = torch.randn(128, 20)
        >>> out = embed(input)
        >>> out.size
        torch.Size([128, 30])
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The projection of original vector to the fixed-dimensionally space.

        Args:
            input (torch.Tensor): Original vector.

        Shape:
            - input: :math:`(B, I, H_{in})`
            - output: :math:`(B, I, H_{out})`

        Notations:
            - :math:`B` - Batch size.
            - :math:`I` - Number of tokens in a sequence.
            - :math:`H_{in}` - Dimensionality of input data element.
            - :math:`H_{out}` - Dimensionality of a fixed space.

        Returns:
            torch.Tensor: Embedded input.
        """
        return self.linear(input)
