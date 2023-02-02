import math

import torch
import torch.nn as nn
import torch.nn.modules.transformer as transformer

from typing import Optional
from athena.nn.arch import Embedding


class TransformerEmbedding(Embedding):
    r"""
    The copy of :class:`Embedding` except the projection process.
    Due to the embedding layers and last linear transformation layer
    share the same weight matrix we scale the weights by the factor
    :math:`\sqrt{d_{model}}`. See details in “`Attention Is All You Need
    <https://arxiv.org/abs/1706.03762>`_” section 3.4.

    Example::

        >>> embed = TransformerEmbedding(136, 512)
        >>> input = torch.rand(10, 10,136)
        >>> out = embed(input) # input was scaled
        >>> out.shape
        torch.Size([10, 10, 512])
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The projection of original input/output tokens of a sequence
        to the fixed-dimensionally space via linear transformation.

        Args:
            input (torch.Tensor): Original input/output tokens.

        Shape:
            - input: :math:`(B, I, H_{in})`
            - output: :math:`(B, I, H_{out})`

        Notification:
            - :math:`B` - Batch size.
            - :math:`I` - Number of tokens in a sequence.
            - :math:`H_{in}` - Dimensionality of input data element.
            - :math:`H_{out}` - Dimensionality of a fixed space.

        Returns:
            torch.Tensor: Scaled embeddings.
        """
        output = self.linear(input) * math.sqrt(self.out_features)
        return output


class PTEncoder(nn.Module):
    r"""
    Transformer encoder implementation based on PyTorch officials
    :class:`torch.nn.TransformerEncoder`.

    Args:
        dim_model (int): Dimension of learnable weights matrix :math:`W^{d_{model} \times d_*}`.
        dim_feedforward (int): Dimension of hidden layers of feedforward network.
        nheads (int): Number of heads in self attention mechanism.
        nlayers (int): Number of stacked layers in the encoder.

    .. important::

        Note that ``dropout`` is set to 0.

    Example::

        >>> encoder = PTEncoder(512, 2048, 8, 6)
        >>> input = torch.rand(10, 32, 512)
        >>> out = encoder(input)
    """

    def __init__(self, dim_model: int, dim_feedforward: int, nheads: int, nlayers: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            dim_feedforward=dim_feedforward,
            nhead=nheads,
            dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=nlayers
        )

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
        Pass the input tokens embedings through the ``nlayers`` stacked encoder.

        Args:
            input (torch.Tensor): Embedded input tokens (items) combined with its positions.
            mask (Optional[torch.Tensor], optional): Ensures that position is allowed to attend.

        Shape:
            - input: :math:`(B, S, d_{model})`
            - mask: :math:`(B \times nheads, S, S)`
            - output: :math:`(B, S, d_{model})`

        Notations:
            - :math:`B` - batch size.
            - :math:`S` - source sequence length.
            - :math:`d_{model}` - Dimension of learnable weights matrix.

        Returns:
            torch.Tensor: Encoded representation of a sequence.
        """
        # Adjust the input for the PyTorch format (batch_size as second dim)
        input = input.transpose(0, 1)
        output: torch.Tensor = self.encoder(input, mask)
        return output.transpose(0, 1)


class PTDecoder(nn.Module):
    r"""
    Transformer decoder implementation based on PyTorch officials
    :class:`torch.nn.TransformerDecoder`.

    Args:
        dim_model (int): Dimension of learnable weights matrix :math:`W^{d_{model} \times d_*}`.
        dim_feedforward (int): Dimension of hidden layers of feedforward network.
        nheads (int): Number of heads in self attention mechanism.
        nlayers (int): Number of stacked layers in the encoder.

    .. important::

        Note that ``dropout`` is set to 0.

    Example::

        >>> decoder = PTDecoder(512, 2048, 8, 6)
        >>> memory = torch.rand(10, 32, 512)
        >>> target = torch.rand(10, 10, 512)
        >>> out = decoder(target, memory)
        >>> out.shape
        torch.Size([10, 10, 512])
    """

    def __init__(self, dim_model: int, dim_feedforward: int, nheads: int, nlayers: int) -> None:
        super().__init__()

        self.layer = nn.TransformerDecoderLayer(
            d_model=dim_model,
            dim_feedforward=dim_feedforward,
            nhead=nheads,
            dropout=0.0
        )
        self.decoder = nn.TransformerDecoder(
            self.layer, num_layers=nlayers
        )

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Pass the input tokens embedings through the ``nlayers`` stacked decoder.

        Args:
            target (torch.Tensor): The sequence to the decoder layer.
            memory (torch.Tensor): The sequence from the last encoder layer.
            tgt_mask (Optional[torch.Tensor], optional): The mask for the target sequence.
            memory_mask (Optional[torch.Tensor], optional): The mask for the memory sequence.

        Shape:
            - target: :math:`(B, T, d_{model})`
            - memory: :math:`(B, S, d_{model})`
            - target_mask: :math:`(B \times nheads, T, T)`
            - memory_mask: :math:`(B \times nheads, T, S)`
            - output: :math:`(B, T, d_{model})`

        Notations:
            - :math:`B` - batch size.
            - :math:`T` - target sequence length.
            - :math:`S` - source sequence length.
            - :math:`d_{model}` - dimension of learnable weights matrix.

        Returns:
            torch.Tensor: Decoded representation of a sequence.
        """
        # Adjust the input for the PyTorch format (batch_size as second dim)
        target = target.transpose(0, 1)
        memory = memory.transpose(0, 1)
        output: torch.Tensor = self.decoder(
            target,
            memory,
            target_mask,
            memory_mask
        )
        return output.transpose(0, 1)


class PointwisePTDecoderLayer(transformer.TransformerDecoderLayer):
    r"""
    We use a bit different decoder output if vocabulary doesn't
    have fixed size. Model doesn't learn to point at position
    in the vocabulary of fixed size, instead it samples the items from
    original sequence of variable length. Thus we want decoder
    outputs the attention weights instead of attention embedings.
    These weights would directly be used in the items sampling.
    """

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Re-implement official PyTorch transformer decoder layer.
        Normally, in the decoder layer we pass the cross-attention
        results to the feedforward to get the weights distributed over
        overall vocabulary, afterward we normalize them within :math:`[0;1]`.
        However, some problem's definitions state that we don't have vocabulary
        of fixed size, and it changes at each time step. But the goal remains
        the same: we want to get probability distribution over vocabulary.
        Technically, original decoder does it during attention process. So the
        pointwise decoder output defines as follow:

        .. math::

            softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)

        For some details see “`Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_”
        and as example check “`Seq2Slate: Re-ranking and Slate Optimization with RNNs
        <https://arxiv.org/abs/1810.02019>`_”.

        Args:
            target (torch.Tensor): The sequence to the decoder layer.
            memory (torch.Tensor): The sequence from the last encoder layer.
            tgt_mask (Optional[torch.Tensor], optional): The mask for the target sequence.
            memory_mask (Optional[torch.Tensor], optional): The mask for the memory sequence.

        Shape:
            - target: :math:`(B, T, d_{model})`
            - memory: :math:`(B, S, d_{model})`
            - tgt_mask: :math:`(B \times nheads, T, T)`
            - memory_mask: :math:`(B, T, S)`
            - output: :math:`(B, T, V)`

        Notations:
            - :math:`B` - batch size.
            - :math:`T` - target sequence length.
            - :math:`S` - source sequence length.
            - :math:`V` - current vocabulary length.
            - :math:`d_{model}` - dimension of learnable weights matrix.

        Returns:
            torch.Tensor: Attention weights are probabilities over symbols.
        """
        # Firstly, apply masked self attention to the target sequence
        # getting d_model attention embeddings.
        attn_values: torch.Tensor = self.self_attn(
            target, target, target, attn_mask=tgt_mask
        )[0]
        # Second, residuals connection [5] and layer normalization [4]
        target = target + self.dropout1(attn_values)
        target = self.norm1(target)
        # Third, multihead attention but the weights are extracted
        # instead of embeddings
        attn_weights: Optional[torch.Tensor] = self.multihead_attn(
            target, memory, memory, attn_mask=memory_mask
        )[1]
        if attn_weights is None:
            raise RuntimeError("Set need_weights=True in PyTorch MultiheadAttention.")
        # We don't really need to optimize the attention embedding values
        # Because we already have prob dist over vocab:)
        return attn_weights


class PointwisePTDecoder(nn.Module):
    r"""
    Transformer decoder implementation based on PyTorch officials with slight
    modification dedicated to sample variable sequence instead of overall vocabulary.

    Args:
        dim_model (int): Dimension of learnable weights matrix :math:`W^{d_{model} \times d_*}`.
        dim_feedforward (int): Dimension of hidden layers of feedforward network.
        nheads (int): Number of heads in self attention mechanism.
        nlayers (int): Number of stacked layers in the encoder.

    .. important::

        Feedforward network isn't in use for the last decoder layer. Therefore if decoder consists
        of the only layer (pointwise layer) this parameter is meaningless.

    Example::

        >>> pointwise_decoder = PointwisePTDecoder(512, 2048, 8, 6)
        >>> memory = torch.rand(10, 32, 512)
        >>> target = torch.rand(10, 10, 512)
        >>> out = pointwise_decoder(target, memory)
        >>> out.shape
        torch.Size([10, 10, 34])

        >>> torch.sum(out[0][0]) # Ensure that PDF is formed
        tensor(1., grad_fn=<SumBackward0>)
    """

    def __init__(self, dim_model: int, dim_feedforward: int, nheads: int, nlayers: int) -> None:
        super().__init__()
        # Stacking size - 1 decoder layers
        default_transformer_decoders = [
            transformer.TransformerDecoderLayer(
                d_model=dim_model,
                dim_feedforward=dim_feedforward,
                nhead=nheads,
                dropout=0.0
            )
            for _ in range(nlayers - 1)
        ]
        # Define last pointwise layer
        pointwise_tranformer_decoder = [
            PointwisePTDecoderLayer(
                d_model=dim_model,
                dim_feedforward=dim_feedforward,
                nhead=nheads,
                dropout=0.0
            )
        ]
        # Define decoder as stack of ordinar layers and pointwise on top
        self.layers = nn.ModuleList(
            default_transformer_decoders + pointwise_tranformer_decoder
        )
        self.nlayers = nlayers

    def forward(
        self,
        target_embed: torch.Tensor,
        memory: torch.Tensor,
        target2source_mask: Optional[torch.Tensor] = None,
        target2target_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Pass target sequence along with encoder latent state and produce the probability
        distribution over the variable sequence. For the details check “`Pointer Networks
        <https://arxiv.org/abs/1506.03134>`_” and *Pointer-Network Architecture for
        Ranking* section in the“`Seq2Slate: Re-ranking and Slate Optimization with RNNs
        <https://arxiv.org/abs/1810.02019>`_”.

        Args:
            target_embed (torch.Tensor): Embedded target sequence.
            memory (torch.Tensor): Latent state of the encoder.
            target2source_mask (Optional[torch.Tensor], optional): Mask for the latent state.
            target2target_mask (Optional[torch.Tensor], optional): Mask for the target sequence.

        Shape:
            - target_embed: :math:`(B, T, d_{model})`
            - memory: :math:`(B, S, d_{model})`
            - target2source_mask: :math:`(B, T, S)`
            - target2target_mask: :math:`(B, T, T)`
            - output: :math:`(B, T, V)`

        .. note::

            Currently, padding and start vectors are not learnable, therefore
            treats them as zero vectors.

        Notations:
            - :math:`B` - batch size.
            - :math:`T` - target sequence length.
            - :math:`S` - source sequence length.
            - :math:`V` - current vocabulary length.
            - :math:`d_{model}` - Dimension of learnable weights matrix.

        Returns:
            torch.Tensor: Probability distribution over rest of items.
        """
        batch_size, target_seq_len = target_embed.shape[:2]

        # Make suitable for the PyTorch
        target_embed = target_embed.transpose(0, 1)
        memory = memory.transpose(0, 1)

        output = target_embed
        # Passing throug the decoder layers
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=target2target_mask,
                memory_mask=target2source_mask
            )
        # We don't really want to sample the placeholders (padding/starting symbols)
        # NOTE: The final sequence length is num_of_candidates + 2
        zero_probas_for_placeholders = torch.zeros(
            batch_size, target_seq_len, 2, device=target_embed.device
        )
        # Final probabilities pointing to the rest of the sequence items
        probas = torch.cat((zero_probas_for_placeholders, output), dim=2)
        return probas


class VLPositionalEncoding(nn.Module):
    r"""
    Special non-learnable positional encoding specified
    for the handling variable length vocabulary. To do so
    we fold joint representation of featurewise sequence
    and item positions into original dimension afterward
    project it back.

    Args:
        dim_model (int): Dimension of learnable weights matrix
          :math:`W^{d_{model} \times d_*}`.
    """

    def __init__(self, dim_model: int) -> None:
        super().__init__()
        self.pos_proj = nn.Linear(dim_model + 1, dim_model)
        self.activation = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Encode input sequence taking item positions into account.

        Args:
            input (torch.Tensor):

        Shape:
            - input: :math:`(B, S, d_{model})`
            - output: :math:`(B, S, d_{model})`

        Notations:
            - :math:`B` - batch size.
            - :math:`S` - sequence length.
            - :math:`d_{model}` - Dimension of learnable weights matrix.


        Returns:
            torch.Tensor: Encoded sequence positions.
        """
        device = input.device
        batch_size, seq_len = input.shape[:2]
        # Obtain the vector of positions
        # for each input in the batch.
        # The lengths of these vectors may differ
        pos_idx = (
            torch.arange(0, seq_len, device=device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .reshape(batch_size, seq_len, 1)
        )
        # Stack the input values (items of sequences)
        # onto its positions in the input vector
        # shape: batch_size, seq_len, dim_model + 1
        input_pos = torch.cat((input, pos_idx), dim=2)
        return self.activation(self.pos_proj(input_pos))
