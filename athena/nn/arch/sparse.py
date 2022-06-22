import torch
import torch.nn as nn

class Embedding(nn.Module):
    """Learnable embedings is the first step in both
    encoder and decoder parts. The input and output
    tokens project to the dimensionally fixed space (d_model)
    via linear transformations, thus making these embeddings
    learnable. For the details see [2] Section 3.4.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Embeddings parameters intialization.

        Args:
            in_features (int): The dimensionality of the original 
                input/output tokens.
            out_features (int): The row-wise dimensionality 
                of learnable weight matrix (d_model). In [2] d_model = 512.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The projection of original input/output tokens of a sequence
        to the fixed-dimensionally space via linear transformation.
        Due to the embedding layers (2 layers) and last linear transformation layer
        share the same weight matrix we scale the weights by the factor sqrt(d_model).

        Args:
            input (torch.Tensor): Original input/output tokens.
                shape: batch_size, seq_len, candidate_dim

        Returns:
            torch.Tensor: Scaled embeddings.
                shape: batch_size, seq_len, out_features
        """
        return self.linear(input)