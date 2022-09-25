"""The implementation of Transformer-based Seq2Slate network.
The network itself is pointer encoder-decoder, i.e. instead of
learning to point to an item (word) in fixed-length vocabulary
it learns to point to an item in the input sequence one could
be variable length. Encoder takes feature-wise sequence of
candidates as input, decoder in its turn takes the sequence and
latent state of encoder and outputs ordered list of candidates.
The final order is learned in reinforce maner to increase
sequence-wise reward (i.e. it learns to find the best permutation).
As options, the network could be optimized in weak supervised and
teacher forcing maners.

As instance, this model could be used to rank given candidate
items to a specific user s.t. final list of items maximizes the
user engagment.

[1] Seq2Slate: https://arxiv.org/abs/1810.02019
[2] Transformer: https://arxiv.org/abs/1706.03762
[3] Pointer Networks: https://arxiv.org/abs/1506.03134
[4] Layer Normalization: https://arxiv.org/abs/1607.06450
[5] Residual Connection: https://arxiv.org/abs/1512.03385
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from athena import gather
from athena.core.config import param_hash
from athena.core.dataclasses import dataclass
from athena.core.dtypes import (PreprocessedRankingInput, RankingOutput,
                                Seq2SlateMode, Seq2SlateOutputArch,
                                Seq2SlateTransformerOutput)
from athena.nn.arch.transformer import DECODER_START_SYMBOL, PADDING_SYMBOL
from athena.core.logger import LoggerMixin
from athena.models.base import BaseModel
from athena.nn.arch import (PointwisePTDecoder, PTEncoder, SimplexSampler,
                            TransformerEmbedding, VLPositionalEncoding)
from athena.nn.functional import prod_probas
from athena.nn.utils.prune import mask_by_index
from athena.nn.utils.transformer import decoder_mask


class Seq2SlateTransformerModel(nn.Module):
    r"""
    The implementation of Seq2Slate model. Architecture is based on "`Seq2Slate: Re-ranking and Slate
    Optimization with RNNs <https://arxiv.org/abs/1810.02019>`_". Irwan Bello, Sayali Kulkarni, Sagar
    Jain, Craig Boutilier, Ed Chi, Elad Eban, Xiyang Luo, Alan Mackey, Ofer Meshi. 2019. This
    implementation is based on transformer encoder-decoder architecture instead of RNN proposed by
    authors.

    The generalized problem is to find a point in n-dimensional simplex which maximizes a metric.
    This problem could be mutated s.t. it describes basic ranking problem, i.e. we look for such
    permutation :math:`\pi` which maximizes user engagement :math:`\mathcal{R}(\pi) = \arg \max{(\Delta^n)}`.
    The obvious solution is to evaluate every vertex in the simplex. The obvious con of such
    method is its factorial computational cost. Thus, alongside with optimal solution we need
    to obtain an optimal complexity (it strongly correlates with problem definition) which
    one could be decreased down to :math:`O(n)` (see :func:`_greedy_decoding`).

    The architecture defined as follow:

    .. image:: ../_static/images/seq2slate_arch.png
        :scale: 45 %

    The model aplplies encoder and modified decoder blocks to an input sequence. Encoder takes input
    sequence :math:`\{x_i\}_{i=1}^n` and transforms it into embedded candidate items set representation
    :math:`\{e_i\}_{i=1}^n`. Decoder at each time step combines query with encoder output and produce
    probabilities over the rest items to include in the result sequence. For the detailes check `Pointer
    Networks <https://arxiv.org/abs/1506.03134>`_ by Oriol Vinyals, Meire Fortunato, Navdeep Jaitly. 2015.

    The model implements three slight different variations:

    1. Auregressive decoder. Pick most probable candidate.
    2. Iterative softmax (Frechet sort). Decoder takes encoder scores and performs descending sort.
    3. No decoder and the output order is solely based on encoder scores.

    Some modifications to the default transformer blocks were applied in purpose to handle variable length
    of vocabulary (source sequence itself). Check :class:`athena.nn.PointwisePTDecoder` for the details.

    Args:
        latent_state_dim (int): The dimension of the current state.
        candidate_dim (int): The dimension of a candidate.
        nlayers (int): The number of stacked encoder and decoder layers.
        nheads (int): The number of heads used in transformer.
          Defines the enrichment of state representations.
        dim_model (int): The dimension of model (:math:`d_{model}` in
          "`Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_").
        dim_feedforward (int): The dimension of hidden layers in the FF network.
        max_source_seq_len (int): The maximum length of input sequences.
        max_target_seq_len (int): The maximum length of output sequences.
        output_arch (Seq2SlateOutputArch): The output architecture of the model.
            Specifically, used to generalize the output for the different model variations.
        temperature (float, optional): The temperature of decoder softmax. Defaults to ``1.0``.
        latent_state_embed_dim (Optional[int], optional): Embedding dimension of the
            latent state. In case it's not specified ``latent_state_embed_dim = dim_model/2``.
            Defaults to ``None``.

    .. warning::

        ``temperature`` isn't in use, currently.

    .. important::

        Here's presented private methods which implicitly used in the encoder/decoder process.
        The purpose of their exhibition is the detailed description of the model performance.
    """
    __constants__ = [
        "latent_state_dim",
        "candidate_dim",
        "nlayers",
        "nheads",
        "dim_model",
        "dim_feedforward",
        "max_source_seq_len",
        "max_target_seq_len",
        "temperature",
        "_padding_symbol",
        "_decoder_start_symbol",
        "_rank_mode_val",
        "_per_item_log_prob_dist_mode_val",
        "_per_seq_log_prob_mode_val",
        "_encoder_score_mode_val",
        "_decode_one_step_mode_val"
    ]

    def __init__(
        self,
        latent_state_dim: int,
        candidate_dim: int,
        nlayers: int,
        nheads: int,
        dim_model: int,
        dim_feedforward: int,
        max_source_seq_len: int,
        max_target_seq_len: int,
        output_arch: Seq2SlateOutputArch,
        temperature: float = 1.0,
        latent_state_embed_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        self.latent_state_dim = latent_state_dim
        self.candidate_dim = candidate_dim
        self.nlayers = nlayers
        self.nheads = nheads
        self.dim_model = dim_model
        self.dim_feedforward = dim_feedforward
        self.max_source_seq_len = max_source_seq_len
        self.max_target_seq_len = max_target_seq_len
        self.output_arch = output_arch

        if latent_state_embed_dim is None:
            latent_state_embed_dim = dim_model // 2
        candidate_embed_dim = dim_model - latent_state_embed_dim
        self.latent_state_embedding = TransformerEmbedding(
            self.latent_state_dim, latent_state_embed_dim
        )
        self.candidate_embedding = TransformerEmbedding(
            self.candidate_dim, candidate_embed_dim
        )
        self._padding_symbol = PADDING_SYMBOL
        self._decoder_start_symbol = DECODER_START_SYMBOL

        self._rank_mode_val = Seq2SlateMode.RANK_MODE.value
        self._per_item_log_prob_dist_mode_val = Seq2SlateMode.PER_ITEM_LOG_PROB_DIST_MODE.value
        self._per_seq_log_prob_mode_val = Seq2SlateMode.PER_SEQ_LOG_PROB_MODE.value
        self._encoder_score_mode_val = Seq2SlateMode.ENCODER_SCORE_MODE.value
        self._decode_one_step_mode_val = Seq2SlateMode.DECODE_ONE_STEP_MODE.value

        # Placeholder for different model variations
        self._output_placeholder = torch.zeros(1)

        self.encoder = PTEncoder(
            self.dim_model, self.dim_feedforward, self.nheads, self.nlayers
        )
        self.encoder_scorer = nn.Linear(self.dim_model, 1)
        self.decoder = PointwisePTDecoder(
            self.dim_model, self.dim_feedforward, self.nheads, self.nlayers
        )
        self.gc = SimplexSampler()
        self.vl_positional_encoding = VLPositionalEncoding(self.dim_model)
        self._initialize_learnable_params()

    def _initialize_learnable_params(self):
        r"""
        Initialize parameters with `Glorot / fan_avg <https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_.
        """
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        mode: str,
        latent_state: torch.Tensor,
        source_seq: torch.Tensor,
        target_seq_len: Optional[int] = None,
        target_input_seq: Optional[torch.Tensor] = None,
        target_input_indcs: Optional[torch.Tensor] = None,
        target_output_indcs: Optional[torch.Tensor] = None,
        greedy: Optional[bool] = None
    ) -> Seq2SlateTransformerOutput:
        r"""Pass the input through the stack of encoders and ptr-decoders.

        Args:
            mode (str): The mode one depicts how is model performing. For the details see
              :class:`athena.core.dtypes.ranking.seq2slate.Seq2SlateMode`.
            latent_state (torch.Tensor): Current latent state of the model.
            source_seq (torch.Tensor): Source sequence.
            target_seq_len (Optional[int], optional): The length of output sequence to be decoded.
              Only used in RANK mode. Defaults to ``None``.
            target_input_seq (Optional[torch.Tensor], optional): Target input sequence one passed
              to the docder input. Used in TEACHER FORCING or REINFORCE mode. Defaults to ``None``.
            target_input_indcs (Optional[torch.Tensor], optional): The indices of the given
              target sequences. Used in TEACHER FORCING/REINFORCE mode. Defaults to ``None``.
            target_output_indcs (Optional[torch.Tensor], optional): The indicies over ones
              the final probabilities distribution will be performed. Defaults to ``None``.
            greedy (Optional[bool], optional): The greedy sample. Defaults to ``None``.

        Shape:
            - latent_state: :math:`(B, H)`
            - source_seq: :math:`(B, S, C)`
            - target_input_seq: :math:`(B, T, C)`
            - target_input_indcs: :math:`(B, T)`
            - target_output_indcs: :math:`(B, T)`

        Notations:
            - :math:`B` - batch size.
            - :math:`H` - dimensionality of hidden state.
            - :math:`S` - length of source sequence.
            - :math:`C` - candidate (item) dimensionality.
            - :math:`T` - length of target sequence.

        Raises:
            ValueError: If greedy hasn't been passed in RANKING mode.
            ValueError: If target data hasn't been passed in REINFORCE mode.
            ValueError: If target input hasn't been passed in ENCODER_SCORE mode.
            NotImplementedError: Other variations of the model not implemented.

        Returns:
            Seq2SlateTransformerOutput: Generalized model output.
        """
        if mode == self._rank_mode_val:
            if target_seq_len is None:
                target_seq_len = self.max_target_seq_len
            if greedy is None:
                raise ValueError(
                    "Ranking mode requires sampling method (greedy or not)"
                )
            return self._rank(
                latent_state,
                source_seq,
                target_seq_len,
                greedy
            )
        elif mode in (
            self._per_item_log_prob_dist_mode_val, self._per_seq_log_prob_mode_val
        ):
            if (target_input_seq is None) or (target_input_indcs is None) or (target_output_indcs is None):
                raise ValueError(
                    "Expected target_input_seq, target_input_indcs, target_output_indcs; "
                    f"Given {target_input_seq, target_input_indcs, target_output_indcs}"
                )
            return self._log_probas(
                latent_state,
                source_seq,
                target_input_seq,
                target_input_indcs,
                target_output_indcs,
                mode
            )
        elif mode == self._encoder_score_mode_val:
            if target_output_indcs is None:
                raise ValueError(
                    f"Expected target_input_indcs; Given {target_input_indcs}"
                )
            return self.encoder_output_to_scores(
                latent_state,
                source_seq,
                target_output_indcs
            )
        else:
            raise NotImplementedError()

    def _rank(
        self,
        latent_state: torch.Tensor,
        source_seq: torch.Tensor,
        target_seq_len: int,
        greedy: bool
    ) -> Seq2SlateTransformerOutput:
        r"""
        Decode and arange sequence according the given input.
        The re-aranged sequence is just a sole permutation
        but we're wanna to get the "best" one so represent
        resulted sequence as its occurance probability by taking
        product of all item's probabilities in the sequence.

        .. math::

            \Delta_i^n = p_{\pi}(\pi | \pi_{|\pi| - 1}, x) =
            \prod_{j=1}^{|\pi|}{p(\pi_j | \pi_{<j}, x)}

        To get :math:`\max{(\Delta^n)}` we want to pick most probable individuals
        :math:`p(\pi_j | \pi_{<j}, x)`. However, to catch higher order interactions
        between the elements we employee three diferent methods.

        .. important::

            Such way of drawing most probable item minimizes local error (or Karcher
            means of simplical manifold) but we want to minimize the global one
            (`Fréchet mean`_). Here's the implementation of these methods:

            - :func:`_greedy_decoding`: Applies Fréchet sort over encoding scores.
            - :func:`_encoder_decoding`: Use only encoder scores to arange a sequence.
            - :func:`_autoregressive_decoding`: Pick an item in autoregressive way.

        Args:
            latent_state (torch.Tensor): Current latent state.
            source_seq (torch.Tensor): Featurewise source sequence.
            target_seq_len (int): Length of the given target sequence.
            greedy (bool): Greedily rank the items.

        Shape:
            - latent_state: :math:`(B, H)`
            - source_seq: :math:`(B, S, C)`

        Notations:
            - :math:`B` - batch size.
            - :math:`H` - dimensionality of hidden state.
            - :math:`S` - length of source sequence.
            - :math:`C` - candidate (item) dimensionality.

        Returns:
            Seq2SlateTransformerOutput: Generalized model output specified for the ranking decoding.

        .. _Fréchet mean:
            https://en.wikipedia.org/wiki/Fréchet_mean
        """
        device = source_seq.device
        # Extract the dimensionality of source sequence
        batch_size, source_seq_len, candidate_dim = source_seq.shape
        # We're considering feeatures of start symbol and padding symbol
        featurewise_seq = torch.zeros(
            batch_size, source_seq_len + 2, candidate_dim, device=device
        )
        # Fill the lookup table with feature values, ignoring start and padding symbols
        # TODO: probably it worths to create learnable vectors for the start/padding symbol
        featurewise_seq[:, 2:, :] = source_seq
        # Obtain the latent memory states {e_i}^source_seq_len of the source sequence
        # memory shape: batch_size, source_seq_len, dim_model
        memory = self.encode(latent_state, source_seq)

        if self.output_arch == Seq2SlateOutputArch.FRECHET_SORT and greedy:
            # Greedy decoding but not in the autoregressive way i.e. we decode
            # entire sequence at one decoder step
            target_output_indcs, ordered_per_item_probas = self._greedy_decoding(
                latent_state, memory, featurewise_seq, target_seq_len
            )
        elif self.output_arch == Seq2SlateOutputArch.ENCODER_SCORE:
            # use only the encode process one doesn't consider
            # high-order items' interactions
            target_output_indcs, ordered_per_item_probas = self._encoder_decoding(
                memory, target_seq_len
            )
        else:
            if greedy is None:
                raise ValueError(
                    "Autoregressive decoding implies greedy way to select adjacent item"
                )
            target_output_indcs, ordered_per_item_probas = self._autoregressive_decoding(
                latent_state, memory, featurewise_seq, target_seq_len, greedy
            )
        # Sequence probability is the product of each sequence's individual
        ordered_per_seq_probas = prod_probas(ordered_per_item_probas, target_output_indcs)
        return Seq2SlateTransformerOutput(
            ordered_per_item_probas=ordered_per_item_probas,
            ordered_per_seq_probas=ordered_per_seq_probas,
            ordered_target_out_indcs=target_output_indcs,
            per_item_log_probas=self._output_placeholder,
            per_seq_log_probas=self._output_placeholder,
            encoder_scores=self._output_placeholder
        )

    def _encoder_decoding(
        self,
        memory: torch.Tensor,
        target_seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Arange items according to its encoder scores
        i.e. self attention scores excluding high-order
        interactions between the items in sequence.

        .. math::

            \{e_i\}_{i=1}^m = E(\{x_i\}_{i=1}^{m})

        Thus encoder exhibits most attractive items and blends the opposite
        ones. However, we miss high-order interaction between the elements
        by erasing the decoding step. In other words, we don't consider other
        permutations relying only on the original ordering, which isn't the
        "best" *a priori*.

        Args:
            memory (torch.Tensor): Encoder output depicts how important
              each item in the sequence at the current time step.
            target_seq_len (int): Length of the target sequence.

        Shape:
            - memory: :math:`(B, S, d_{model})`
            - output: :math:`((B, T), (B, T))`

        Notations:
            - :math:`B` - batch size.
            - :math:`S` - source sequence length.
            - :math:`d_{model}` - Dimension of learnable weights matrix.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Re-aranged permutation and
            generative item probabilitites in the permutation.
        """
        device = memory.device
        batch_size, source_seq_len = memory.shape[:2]
        num_of_candidates = source_seq_len + 2

        ordered_per_item_probas = torch.zeros(
            batch_size, target_seq_len, num_of_candidates, device=device
        )
        # memory is the encoder output intended to be incorporated
        # in decoding process therefore its shape is: batch_size,
        # src_seq_len, dim_model. But we want use them directly
        # s.t. embed them into 1D making their shape batch_size, src_seq_len
        encoder_scores = self.encoder_scorer(memory).squeeze(dim=2)
        target_output_indcs = torch.argsort(
            encoder_scores, dim=1, descending=True
        )
        # deem start and padding symbols
        target_output_indcs += 2
        # every position has propensity of 1 because we are just using argsort
        # i.e. each item 100% holds the proper position
        ordered_per_item_probas = ordered_per_item_probas.scatter(
            2, target_output_indcs.unsqueeze(2), 1.0
        )
        return target_output_indcs, ordered_per_item_probas

    def _autoregressive_decoding(
        self,
        latent_state: torch.Tensor,
        memory: torch.Tensor,
        featurewise_seq: torch.Tensor,
        target_seq_len: int,
        greedy: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Autoregressive decoding implies an element selection at
        each decoding step without considering remaining ones. I.e.
        at each step we pick most probable elment under assumption
        that target sequence's elements are independent. In constrast
        :func:`_encoder_decoding` we produce probability distribution
        over remaining items at each time step, by that encorporating
        all permutations.

        Args:
            latent_state (torch.Tensor): Current latent state.
            memory (torch.Tensor): Encoder output depicts how important
                each item in the sequence at the current time step.
            featurewise_seq (torch.Tensor): The source sequence adjusted
                for the learning purpose. I.e. added start and padding symbols' vectors.
            target_seq_len (int): Length of the target sequence.
            greedy (bool): The way how to choose next candidate. :class:`athena.nn.SimplexSampler`.

        Shape:
            - latent_state: :math:`(B, H)`
            - memory: :math:`(B, S, d_{model})`
            - featurewise_seq: :math:`(B, S + 2, C)`
            - output: :math:`((B, T), (B, T))`

        Notations:
            - :math:`B` - batch size.
            - :math:`H` - dimensionality of hidden state.
            - :math:`S` - length of source sequence.
            - :math:`C` - candidate (item) dimensionality.
            - :math:`d_{model}` - Dimension of learnable weights matrix.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Re-aranged permutation and
            generative item probabilitites in the permutation.
        """
        device = featurewise_seq.device
        batch_size, num_of_candidates = featurewise_seq.shape[:2]
        # we will process the entire sequence step by step
        target_input_indcs = torch.full(
            (batch_size, 1), self._decoder_start_symbol, dtype=torch.long, device=device
        )
        ordered_per_item_probas = torch.zeros(
            batch_size, target_seq_len, num_of_candidates, device=device
        )
        for step in torch.arange(target_seq_len, device=device):
            # take already choosen items to recalculate generative probabilities
            # considering the item selected at last step.
            target_input_seq = gather(featurewise_seq, target_input_indcs)
            # Extract sequential probability distribution accounting most
            # probable item drawn at previous step. Therefore at each step
            # we will get varying distributions over remaining items s.t.
            # taking higher order items' interactions into account.
            # probas shape: batch_size, step + 1, num_of_candidates
            probas = self.decode(
                memory, latent_state, target_input_indcs, target_input_seq
            )
            # Choose most probable (or sampling) item at each step.
            # Obviously it could vary step to step
            # candidate shape: batch_size, 1
            # probas_dist shape: batch_size, nom_of_candidates
            candidate, probas_dist = self.gc(probas, greedy)
            # Store generative probability for the current step (distribution)
            ordered_per_item_probas[:, step, :] = probas_dist
            target_input_indcs = torch.cat(
                [target_input_indcs, candidate], dim=1
            )
        # remove the decoder start symbol
        # target_output_indcs shape: batch_size, target_seq_len
        # ordered_per_item_probas shape: batch_size, target_seq_len, num_of_candidates
        target_output_indcs = target_input_indcs[:, 1:]
        return target_output_indcs, ordered_per_item_probas

    def _greedy_decoding(
        self,
        latent_state: torch.Tensor,
        memory: torch.Tensor,
        featurewise_seq: torch.Tensor,
        target_seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Use one-step decoding scores to greedily rank items. In
        other words we emplyee decoder only once and produce
        the distribution over symbols, considering only some
        of permutations.

        Args:
            latent_state (torch.Tensor): Current latent state.
            memory (torch.Tensor): Encoder output depicts how important
                each item in the sequence at the current time step.
            featurewise_seq (torch.Tensor): The source sequence adjusted
                for the learning purpose. I.e. added start and padding symbols' vectors.
            target_seq_len (int): Length of the target sequence.

        Shape:
            - latent_state: :math:`(B, H)`
            - memory: :math:`(B, S, d_{model})`
            - featurewise_seq: :math:`(B, S + 2, C)`
            - output: :math:`((B, T), (B, T))`

        Notations:
            - :math:`B` - batch size.
            - :math:`H` - dimensionality of hidden state.
            - :math:`S` - length of source sequence.
            - :math:`C` - candidate (item) dimensionality.
            - :math:`d_{model}` - Dimension of learnable weights matrix.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Re-aranged permutation and
            generative item probabilitites in the permutation.
        """
        device = featurewise_seq.device
        batch_size, num_of_candidates = featurewise_seq.shape[:2]
        target_input_indcs = torch.full(
            (batch_size, 1), self._decoder_start_symbol, dtype=torch.long, device=device
        )
        # take the first element at each input (decoder start symbol)
        # thus decoder will output the probabilities over the entire
        # sequence in one step.
        target_input_seq = gather(featurewise_seq, target_input_indcs)
        # decoder outputs probabilities over each symbol (item) for
        # each target sequnce position. We consider the current position.
        # probs shape: batch_size, num_of_candidates
        probas = self.decode(
            memory, latent_state, target_input_indcs, target_input_seq
        )[:, -1, :]

        # arange the items in the greedy way
        target_output_indcs = torch.argsort(
            probas, dim=1, descending=True
        )[:, :target_seq_len]

        # Due to it's greedy ranking we don't really
        # care about already choosen items. Further
        # we consider only remaining items. Therefore
        # we set items' probas to 1.
        ordered_per_item_probas = torch.zeros(
            batch_size, target_seq_len, num_of_candidates, device=device
        ).scatter(2, target_output_indcs.unsqueeze(2), 1.0)
        return target_output_indcs, ordered_per_item_probas

    def _log_probas(
        self,
        latent_state: torch.Tensor,
        source_seq: torch.Tensor,
        target_input_seq: torch.Tensor,
        target_input_indcs: torch.Tensor,
        target_output_indcs: torch.Tensor,
        mode: str
    ) -> Seq2SlateTransformerOutput:
        r"""
        For the REINFORCE training we're required for the log of
        generative probabilities, but not the aranged sequence.
        The ordering should maximize :math:`\mathcal{R}(\pi)`, one
        is commonly NDCG.

        Args:
            latent_state (torch.Tensor): Current latent state.
            source_seq (torch.Tensor): Featurewise source sequence.
            target_input_seq (torch.Tensor): Target input sequence one passed to the docder input.
            target_input_indcs (torch.Tensor): The indices of the given target sequences.
            target_output_indcs (torch.Tensor): The indicies over ones the final probabilities distribution will be
                performed.
            mode (str): The way how to optimize the network. Either calculate sequence or item distribution reward.

        Shape:
            - latent_state: :math:`(B, H)`
            - source_seq: :math:`(B, S, C)`
            - target_input_seq: :math:`(B, T, C)`
            - target_input_indcs: :math:`(B, T)`
            - target_output_indcs: :math:`(B, T)`

        Notations:
            - :math:`B` - batch size.
            - :math:`H` - dimensionality of hidden state.
            - :math:`S` - length of source sequence.
            - :math:`C` - candidate (item) dimensionality.
            - :math:`T` - length of target sequence.

        Raises:
            ValueError: In case if target sequence length is greater than source one.

        Returns:
            Seq2SlateTransformerOutput: Generalized model output.
        """
        # Transform source sequence into embeddings
        # shape: batch_size, source_seq_len, dim_model
        memory = self.encode(latent_state, source_seq)
        target_seq_len = target_input_seq.shape[1]
        source_seq_len = source_seq.shape[1]
        if target_seq_len > source_seq_len:
            raise ValueError(
                f"Target sequence len is greater than source sequence len. "
                f"{target_seq_len} > {source_seq_len}"
            )
        # Extract the probability distribution over items
        # decoder_probs shape: batch_size, target_seq_len, num_of_candidates
        probas = self.decode(
            memory, latent_state, target_input_indcs, target_input_seq
        )
        if mode == self._per_item_log_prob_dist_mode_val:
            # to prevent log(P) = -inf, clamp it with extremely small value
            # shape: batch_size, target_seq_len, num_of_candidates
            per_item_log_probas = torch.log(torch.clamp(probas, min=1e-40))
            return Seq2SlateTransformerOutput(
                ordered_per_item_probas=None,
                ordered_per_seq_probas=None,
                ordered_target_out_indcs=None,
                per_item_log_probas=per_item_log_probas,
                per_seq_log_probas=None,
                encoder_scores=None
            )
        # shape: batch_size, 1
        per_seq_log_probas = torch.log(prod_probas(probas, target_output_indcs))
        return Seq2SlateTransformerOutput(
            ordered_per_item_probas=None,
            ordered_per_seq_probas=None,
            ordered_target_out_indcs=None,
            per_item_log_probas=None,
            per_seq_log_probas=per_seq_log_probas,
            encoder_scores=None
        )

    def encoder_output_to_scores(
        self,
        latent_state: torch.Tensor,
        source_seq: torch.Tensor,
        target_output_indcs: torch.Tensor
    ) -> Seq2SlateTransformerOutput:
        """Similar to the :func:`_encoder_decoding`, except it's
        not sorting according to the encoder scores.

        Args:
            latent_state (torch.Tensor): Current latent state.
            source_seq (torch.Tensor): Featurewise source sequence.
            target_output_indcs (torch.Tensor): The indicies over ones
                the final probabilities distribution will be performed.

        Shape:
            - latent_state: :math:`(B, H)`
            - source_seq: :math:`(B, S, C)`
            - target_output_indcs: :math:`(B, T)`

        Notations:
            - :math:`B` - batch size.
            - :math:`H` - dimensionality of hidden state.
            - :math:`S` - length of source sequence.
            - :math:`C` - candidate (item) dimensionality.
            - :math:`T` - length of target sequence.

        Returns:
            Seq2SlateTransformerOutput: Generalized model output.
        """
        # encode source sequence into model memory state
        # shape: batch_size, source_seq_len, dim_model
        memory = self.encode(latent_state, source_seq)
        # order the memory scores according to the target_output_indcs
        # excluding starting and padding symbols
        # shape: batch_size, target_seq_len, dim_model
        slate_encoder_scores = gather(memory, target_output_indcs - 2)
        # Embed the scores into 1D
        # shape: batch_size, target_seq_len
        encoder_scores = self.encoder_scorer(slate_encoder_scores).squeeze()
        return Seq2SlateTransformerOutput(
            ordered_per_item_probas=None,
            ordered_per_seq_probas=None,
            ordered_target_out_indcs=None,
            per_item_log_probas=None,
            per_seq_log_probas=None,
            encoder_scores=encoder_scores
        )

    def encode(self, latent_state: torch.Tensor, source_seq: torch.Tensor) -> torch.Tensor:
        r"""
        Seq2Slate encoding process. The process consists of two steps:

        1. Combine current latent model state with new input by
        stacking one over another. S.t. resulted embedding
        dimensionality will be equal to the d_model.

        2. Pass this embedding through the default transformer
        encoder layers. As result we get vectorized sequence
        representation :math:`\{e_i\}_{i=1}^n`.

        Args:
            latent_state (torch.Tensor): Current latent state.
            source_seq (torch.Tensor): Featurewise source sequence.

        Shape:
            - latent_state: :math:`(B, H)`
            - source_seq: :math:`(B, S, C)`
            - output: :math:`(B, S, d_{model})`

        Notations:
            - :math:`B` - batch size.
            - :math:`H` - dimensionality of hidden state.
            - :math:`S` - length of source sequence.
            - :math:`C` - candidate (item) dimensionality.
            - :math:`d_{model}` - Dimension of learnable weights matrix.

        Returns:
            torch.Tensor: Sequence represented as embeddings (memory).
        """
        batch_size, max_source_seq_len = source_seq.shape[:2]
        # candidate_embed: batch_size, source_seq_len, dim_model/2
        candidate_embed = self.candidate_embedding(source_seq)
        # latent_state_embed: batch_size, dim_model/2
        latent_state_embed = self.latent_state_embedding(latent_state)
        # transform latent_state_embed into shape: batch_size, source_seq_len, dim_model/2
        latent_state_embed = latent_state_embed.repeat(1, max_source_seq_len).reshape(
            batch_size, max_source_seq_len, -1
        )
        # Encoder input at each step is the concatenation of latent_state_embed
        # and candidate_embed. The latent state is replicated at each encoder step.
        # source_state_embed shape: batch_size, source_seq_len, dim_model
        source_state_embed = torch.cat(
            (latent_state_embed, candidate_embed), dim=2
        )
        return self.encoder(source_state_embed)

    def decode(
        self,
        memory: torch.Tensor,
        latent_state: torch.Tensor,
        target_input_indcs: torch.Tensor,
        target_input_seq: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Seq2Slate decoding process.
        The process splits over two slightly different implementations.

        1. First way is intended for the sampling purpose, we just embed
        encoder scores into 1D vector and apply softmax to it, by one
        making scores distribution as PDF. This method is not adapted
        for the full high-order inference, learning in one decoder step
        but eventually it should converge.

        2. The second option is autoregressive
        where each time step will change the probability distribution
        over remaining items due to the attention values are varying.
        For this option we use stack of decoder layers to combine
        the information about the self attention values from encoder
        output to the cross-attention values for the remaining items
        obtained from attention sublayer of the decoder. By these
        we evaluate high-order inference between the items taking
        already set ones into the account.

        .. note::
            Last decoder layer is modified s.t. it outputs the
            probabilities over rest of source sequence (vocabulary).


        Args:
            memory (torch.Tensor): Encoder output depicts how important each item
                in the sequence at the current time step.
            latent_state (torch.Tensor): Current latent state.
            target_input_indcs (torch.Tensor): The indices of the given target sequences.
            target_input_seq (torch.Tensor): Target input sequence one passed to the docder input.

        Shape:
            - memory: :math:`(B, S, d_{model})`
            - latent_state: :math:`(B, H)`
            - target_input_indcs: :math:`(B, T)`
            - target_input_seq: :math:`(B, T, d_{model})`
            - output: :math:`(B, T, C)`

        Notations:
            - :math:`B` - batch size.
            - :math:`H` - dimensionality of hidden state.
            - :math:`S` - length of source sequence.
            - :math:`C` - candidate (item) dimensionality.
            - :math:`T` - length of target sequence.
            - :math:`d_{model}` - Dimension of learnable weights matrix.

        Returns:
            torch.Tensor: Probabilities over symbols.
        """
        batch_size, source_seq_len = memory.shape[:2]
        target_seq_len = target_input_indcs.shape[1]
        # taking starting and padding symbols into account
        num_of_candidates = source_seq_len + 2
        if self.output_arch == Seq2SlateOutputArch.FRECHET_SORT:
            # Project d_model encoder scores to the 1d line
            # Thus getting attention value for each item in the sequence
            encoder_scores = self.encoder_scorer(memory).squeeze(dim=2)
            input_scores = torch.zeros(batch_size, target_seq_len, num_of_candidates).to(
                encoder_scores.device
            )
            # Starting and padding symbols has probability 0
            input_scores[:, :, :2] = float("-inf")
            # Probability distribution over each symbol for each position
            input_scores[:, :, 2:] = encoder_scores.repeat(1, target_seq_len).reshape(
                batch_size, target_seq_len, source_seq_len
            )
            # Mask already choosen items
            input_scores = mask_by_index(input_scores, target_input_indcs)
            # Normalize the scores within [0;1]
            probas = torch.softmax(input_scores, dim=2)
        elif self.output_arch == Seq2SlateOutputArch.AUTOREGRESSIVE:
            # Embed candidate vector in purpose to combine it further
            # shape: batch_size, target_seq_len, dim_model/2
            candidate_embed = self.candidate_embedding(target_input_seq)
            # Embed latent state vector in purpose to combine it further
            # shape: batch_size, dim_model/2
            latent_state_embed = self.latent_state_embedding(latent_state)
            # Replicate latent state for every open position to choose a candidate
            # shape: batch_size, target_seq_len, dim_model/2
            latent_state_embed = latent_state_embed.repeat(1, target_seq_len).reshape(
                batch_size, target_seq_len, -1
            )
            # Add position vectors to the embeddings
            # shape: batch_size, target_seq_len, dim_model
            target_embed = self.vl_positional_encoding(
                torch.cat((latent_state_embed, candidate_embed), dim=2)
            )
            # Mask already choosen items
            # target2target_mask shape: batch_size * nheads, target_seq_len, target_seq_len
            # target2source_mask shape: batch_size * nheads, target_seq_len, source_seq_len
            target2target_mask, target2source_mask = decoder_mask(
                memory, target_input_indcs, self.nheads
            )
            # Apply decoder layers
            probas = self.decoder(
                target_embed, memory, target2source_mask, target2target_mask
            )
        else:
            raise NotImplementedError()
        return probas


@dataclass
class Seq2SlateNetwork(BaseModel, LoggerMixin):
    __hash__ = param_hash

    latent_state_dim: int
    candidate_dim: int
    nlayers: int
    dim_model: int
    max_source_seq_len: int
    max_target_seq_len: int
    output_arch: Seq2SlateOutputArch
    temperature: float

    def __post_init_post_parse__(self) -> None:
        super().__init__()
        self.seq2slate = self._build_model()

    def _build_model(self) -> Seq2SlateTransformerModel:
        return None

    def input_prototype(self) -> PreprocessedRankingInput:
        return PreprocessedRankingInput.from_tensors(
            state=torch.randn(1, self.latent_state_dim),
            source_seq=torch.randn(1, self.max_source_seq_len, self.candidate_dim),
            target_input_seq=torch.randn(1, self.max_target_seq_len, self.candidate_dim),
            target_output_seq=torch.randn(1, self.max_target_seq_len, self.candidate_dim),
            slate_reward=torch.rand(1)
        )

    def forward(
        self,
        input: PreprocessedRankingInput,
        mode: Seq2SlateMode,
        target_seq_len: Optional[int] = None,
        greedy: Optional[bool] = None
    ):
        if mode == Seq2SlateMode.RANK_MODE:
            result: Seq2SlateTransformerOutput = self.seq2slate(
                mode=mode.value,
                latent_state=input.latent_state.repr,
                source_seq=input.source_seq.repr,
                target_seq_len=target_seq_len,
                greedy=greedy
            )
            return RankingOutput(
                ordered_target_out_indcs=result.ordered_target_out_indcs,
                ordered_per_item_probas=result.ordered_per_item_probas,
                ordered_per_seq_probas=result.ordered_per_seq_probas,
            )
        elif mode in (
            Seq2SlateMode.PER_ITEM_LOG_PROB_DIST_MODE,
            Seq2SlateMode.PER_SEQ_LOG_PROB_MODE
        ):
            if None in (
                input.target_input_seq,
                input.target_input_indcs,
                input.target_output_indcs
            ):
                raise ValueError(
                    "For the REINFORCE learning "
                    "target_input_seq, target_input_indcs, target_output_indcs "
                    "required."
                )
            result: Seq2SlateTransformerOutput = self.seq2slate(
                mode=mode.value,
                latent_state=input.latent_state.repr,
                source_seq=input.source_seq.repr,
                target_input_seq=input.target_input_seq.repr,
                target_input_indcs=input.target_input_indcs,
                target_output_indcs=input.target_output_indcs
            )
            if result.per_item_log_probas is not None:
                log_probas = result.per_item_log_probas
            else:
                log_probas = result.per_seq_log_probas
            return RankingOutput(log_probas=log_probas)
        elif mode == Seq2SlateMode.ENCODER_SCORE_MODE:
            if input.target_output_indcs is None:
                raise ValueError(
                    "For the ENCODER_SCORE_MODE target_output_indcs required."
                )
            result: Seq2SlateTransformerOutput = self.seq2slate(
                mode=mode.value,
                latent_state=input.latent_state.repr,
                source_seq=input.source_seq.repr,
                target_output_indcs=input.target_output_indcs
            )
            return RankingOutput(encoder_scores=result.encoder_scores)
        else:
            raise NotImplementedError()

    def get_distributed_data_parallel_model(self):
        raise NotImplementedError()  # TODO: Implement


@dataclass
class Seq2SlateTransformerNetwork(Seq2SlateNetwork):
    __hash__ = param_hash

    nheads: int
    dim_feedforward: int
    latent_state_embed_dim: Optional[int] = None

    def _build_model(self) -> Seq2SlateTransformerModel:
        return Seq2SlateTransformerModel(
            latent_state_dim=self.latent_state_dim,
            candidate_dim=self.candidate_dim,
            nlayers=self.nlayers,
            nheads=self.nheads,
            dim_model=self.dim_model,
            dim_feedforward=self.dim_feedforward,
            max_source_seq_len=self.max_source_seq_len,
            max_target_seq_len=self.max_target_seq_len,
            output_arch=self.output_arch,
            temperature=self.temperature,
            latent_state_embed_dim=self.latent_state_embed_dim
        )
