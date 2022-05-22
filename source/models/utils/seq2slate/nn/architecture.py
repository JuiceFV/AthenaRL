"""Implementation of architecture entities of the Seq2Slate model.

[1] Seq2Slate: https://arxiv.org/abs/1810.02019
[2] Transformer: https://arxiv.org/abs/1706.03762
[3] Pointer Networks: https://arxiv.org/abs/1506.03134
[4] Layer Normalization: https://arxiv.org/abs/1607.06450
[5] Residual Connection: https://arxiv.org/abs/1512.03385
"""
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.modules.transformer as transformer


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
        output = self.linear(input) * math.sqrt(self.out_features)
        return output


class PTEncoder(nn.Module):
    """Transformer encoder implementation based on PyTorch officials.
    """

    def __init__(self, dim_model: int, dim_feedforward: int, nheads: int, nlayers: int) -> None:
        """Initialization of transformer-based encoder.

        Args:
            dim_model (int): Dimension of learnable weights matrix :math:`W^{d_model x d_*}`.
            dim_feedforward (int): Dimension of hidden layers of feedforward network.
            nheads (int): Number of heads in self attention mechanism.
            nlayers (int): Number of stacked layers in the encoder.
        """
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            dim_feedforward=dim_feedforward,
            nhead=nheads,
            dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(
            self.layer, num_layers=nlayers
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Pass the input tokens embedings through the `self.nlayers`
        stacked encoder to get the latent state. In [1] it's described
        as :math:`{x_i}^n -> {e_i}^n`, where :math:`e_i = Encoder(x_i)`.

        Args:
            input (torch.Tensor): Embedded input tokens (items) combined w/ its positions.
                shape: batch_size, source_seq_len, dim_model

        Returns:
            torch.Tensor: p-dimensional latent state for every token in sequence.
                shape: batch_size, source_seq_len, dim_model
        """
        # Adjust the input for the PyTorch format (batch_size as second dim)
        input = input.transpose(0, 1)
        # w/o mask due to currently have no paddings
        output: torch.Tensor = self.encoder(input)
        return output.transpose(0, 1)


class PointwisePTDecoderLayer(transformer.TransformerDecoderLayer):
    """Seq2Slate doesn't learn to point at position in the
    vocabulary of fixed size. Instead it samples the items from
    original sequence of variable length. Thus we wanna decoder
    outputs the attention weights instead of attention embedings.
    These weights would directly be used in the items sampling.
    """

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: torch.Tensor,
        memory_mask: torch.Tensor
    ) -> torch.Tensor:
        """Re-implement official PyTorch transformer decoder layer. 
        Normally, in the decoder layer we pass the cross-attention
        results to the feedforward to get the weights of affection
        afterward we normalize them within :math: `[0;1]`. We do this
        'cause vocabulary size differs from target sequnce and we need
        to get probability distribution over entire vocab. But this is 
        not the case, it's sufficient to use self attention weights
        :math:`softmax(QK^T/sqrt(d_model))`. For details see Figure 1 in the [2].

        Args:
            target (torch.Tensor): The sequence to the decoder layer.
                shape: batch_size, target_seq_len, dim_model
            memory (torch.Tensor): The sequence from the last layer.
                shape: batch_size, source_seq_len, dim_model
            target_mask (torch.Tensor): The mask for the target sequence.
                shape: batch_size, target_seq_len, target_seq_len
            memory_mask (torch.Tensor): The mask for the memory sequence.
                shape: batch_size, target_seq_len, source_seq_len

        Returns:
            torch.Tensor: Attention weights are probabilities over symbols.
                shape: batch_size, target_seq_len, num_ofcandidates
        """
        # Firstly, apply masked self attention to the target sequence
        # getting d_model attention embeddings.
        attn_values: torch.Tensor = self.self_attn(
            target, target, target, attn_mask=target_mask
        )[0]
        # Second, residuals connection [5] and layer normalization [4]
        target = target + self.dropout1(attn_values)
        target = self.norm1(target)
        # Third, multihead attention but the weights are extracted
        # instead of embeddings
        attn_weights = self.multihead_attn(
            target,
            memory,
            memory,
            attn_mask=memory_mask
        )[1]
        # We don't really need to optimize the attention embedding values
        # Because we already have prob dist over vocab:)
        return attn_weights


class PTDecoder(nn.Module):
    """Transformer decoder implementation based on PyTorch officials
    with slight modification dedicated to sample the target input
    instead of overall vocabulary. For the details see [1] Figure 1 or [3].
    """

    def __init__(self, dim_model: int, dim_feedforward: int, nheads: int, nlayers: int) -> None:
        """Initialize pointwise decoder.

        Args:
            dim_model (int): Dimension of learnable weights matrix :math:`W^{d_model x d_*}`.
            dim_feedforward (int): Dimension of hidden layers of feedforward network.
                NOTE: Feedforward network isn't in use for the last decoder layer. Therefore 
                if decoder consists of the only layer (pointwise layer) this parameter is meaningless.
            nheads (int): Number of heads in self attention mechanism.
            nlayers (int): Number of stacked layers in the encoder.
        """
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
        target2source_mask: torch.Tensor,
        target2target_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pass target sequence along with encoder latent state
        and produce the probability distribution over the target
        sequence. For the details check the section 
        "Pointer-Network Architecture for Ranking" in [1].

        Args:
            target_embed (torch.Tensor): Embedded target sequence.
                shape: batch_size, target_seq_len, dim_model
            memory (torch.Tensor): Latent state of the encoder, i.e. :math:`{e_i}^n`.
                shape: batch_size, source_seq_len, dim_model
            target2source_mask (torch.Tensor): Mask for the latent state.
                shape: batch_size, target_seq_len, source_seq_len            
            target2target_mask (torch.Tensor): Mask for the target sequence.
                shape: batch_size, target_seq_len, target_seq_len
        Returns:
            torch.Tensor: Probability distribution over target sequence.
                shape: batch_size, target_seq_len + 2, num_of_candidates
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
                target_mask=target2target_mask,
                memory_mask=target2source_mask
            )
        # We don't really want to sample the placeholders (padding/starting symbols)
        # NOTE: The final sequence length is target_seq_len + 2
        zero_probas_for_placeholders = torch.zeros(
            batch_size, target_seq_len, 2, device=target_embed.device
        )
        # Final probabilities pointing to the target sequence items
        probas = torch.cat((zero_probas_for_placeholders, output), dim=2)
        return probas


class CandidateGenerator(nn.Module):
    """One way to train seq2slate model is autoregressive.
    I.e. at each time step we choose j candidate, s.t. the 
    generative probability of resulting permutation is optimal 
    :math:`perm_* = argmax(P(perm_j|perm_{<j}, candidate))`. 
    Authors suggest two ways to sample the candidate:
    * Greedy: At each time step we choose the item with highest proability
    * Sampling: Sample candidate, theoretically converges to the expected 
        value (one would be optimized)
    """

    def forward(self, probas: torch.Tensor, greedy: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode one-step

        Args:
            probas (torch.Tensor): Probability distributions of decoder.
                Shape: batch_size, target_seq_len, num_of_candidates
            greedy (bool): Whether to greedily pick or sample the next symbol.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Chosen candidate & Generative probabilities of last step
                shape: batch_size, 1
                shape: batch_size, num_of_candidates
        """
        batch_size = probas.shape[0]
        # Retrieve the last observed probabilities
        probas_dist = probas[:, -1, :]
        if greedy:
            _, candidate = torch.max(probas_dist, dim=1)
        else:
            # idx = min({i in {1, ..., len(probas_dist)}: sum(probas_dist, i) - X >= 0}})
            # Where X ~ U(0,1) and probas_dist sorted descendically.
            candidate = torch.multinomial(probas_dist, replacement=False)
        candidate = candidate.reshape(batch_size, 1)
        return candidate, probas_dist


class VLPositionalEncoding(nn.Module):
    """Special non-learnable positional encoding specified 
    for the handling variable length vocabulary. To do so
    we fold the input positions to the lower dimension
    afterward project them back to the original dimension.
    """

    def __init__(self, dim_model: int) -> None:
        """Constructor.

        Args:
            dim_model (int): Dimension of learnable weights matrix :math:`W^{d_model x d_*}`.
        """
        super().__init__()
        self.pos_proj = nn.Linear(dim_model + 1, dim_model)
        self.activation = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Encode input sequence taking item positions into account.

        Args:
            input (torch.Tensor): 
                shape: batch_size, seq_len, dim_model

        Returns:
            torch.Tensor: Encoded sequence positions.
                shape: batch_size, seq_len, dim_model
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
