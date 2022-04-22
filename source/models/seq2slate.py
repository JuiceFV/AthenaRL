import logging
import math
from turtle import forward
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.modules.transformer as transformer
import source.core.dtypes as dt
import source.models.utils.seq2slate.constatnts as const

from source.core.dataclasses import dataclass
from source.core.confg import param_hash
from source.models.base import BaseModel
from source.core.logger import LoggerMixin
from source.models.utils.seq2slate.dtypes import (
    Seq2SlateMode,
    Seq2SlateOutputArch,
    Seq2SlateTransformerOutput
)

try:
    from typing import Final
except ImportError:
    from typing_extensions import Final


class Embedding(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.linear(input) * math.sqrt(self.out_features)
        return output


class PTEncoder(nn.Module):
    def __init__(self, dim_model: int, dim_feedforward: int, nheads: int, nlayers: int) -> None:
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
        input = input.transpose(0, 1)
        # w/o mask due to currently have no paddings
        output: torch.Tensor = self.encoder(input)
        return output.transpose(0, 1)


class PointwisePTDecoderLayer(transformer.TransformerDecoderLayer):
    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: torch.Tensor,
        memory_mask: torch.Tensor
    ) -> torch.Tensor:
        attn_values: torch.Tensor = self.self_attn(
            target, target, target, attn_mask=target_mask
        )[0]
        target += self.dropout1(attn_values)
        target = self.norm1(target)
        attn_weights = self.multihead_attn(
            target,
            memory,
            memory,
            attn_mask=memory_mask
        )[1]
        return attn_weights


class PTDecoder(nn.Module):
    def __init__(self, dim_model: int, dim_feedforward: int, nheads: int, nlayers: int) -> None:
        super().__init__()
        default_transformer_decoders = [
            transformer.TransformerDecoderLayer(
                d_model=dim_model,
                dim_feedforward=dim_feedforward,
                nhead=nheads,
                dropout=0.0
            )
            for _ in range(nlayers - 1)
        ]
        pointwise_tranformer_decoder = [
            PointwisePTDecoderLayer(
                d_model=dim_model,
                dim_feedforward=dim_feedforward,
                nhead=nheads,
                dropout=0.0
            )
        ]
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
    ):
        batch_size, target_seq_len = target_embed.shape[:2]

        target_embed = target_embed.transpose(0, 1)
        memory = memory.transpose(0, 1)

        output = target_embed

        for layer in self.layers:
            output = layer(
                output,
                memory,
                target_mask=target2target_mask,
                memory_mask=target2source_mask
            )

        zero_probs_for_placeholders = torch.zeros(
            batch_size, target_seq_len, 2, device=target_embed.device
        )
        probas = torch.cat((zero_probs_for_placeholders, output), dim=2)
        return probas


class CandidateGenerator(nn.Module):
    def forward(self, probas: torch.Tensor, greedy: bool):
        batch_size = probas.shape[0]
        last_prob = probas[:, -1, :]
        if greedy:
            _, candidate = torch.max(last_prob, dim=1)
        else:
            candidate = torch.multinomial(last_prob, replacement=False)
        candidate = candidate.reshape(batch_size, 1)
        return candidate, last_prob


class VariableLengthPositionalEncoding(nn.Module):
    def __init__(self, dim_model: int) -> None:
        super().__init__()
        self.pos_proj = nn.Linear(dim_model + 1, dim_model)
        self.activation = nn.ReLU()

    def forward(self, input: torch.Tensor):
        device = input.device
        batch_size, seq_len = input.shape[:2]
        pos_idx = (
            torch.arange(0, seq_len, device=device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .reshape(batch_size, seq_len, 1)
        )
        input_pos = torch.cat((input, pos_idx), dim=2)
        return self.activation(self.pos_proj(input_pos))


class Seq2SlateTransformerModel(nn.Module):
    __constants__ = [
        "latent_state_dim",
        "candidate_dim",
        "nlayers",
        "nheads",
        "dim_model",
        "dim_feedforward",
        "max_source_seq_len",
        "max_target_seq_len",
        "output_arch",
        "latent_state_embedding",
        "candidate_embedding",
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
        """_summary_
        # TODO: print_model_info

        Args:
            latent_state_dim (int): _description_
            candidate_dim (int): _description_
            nlayers (int): _description_
            nheads (int): _description_
            dim_model (int): _description_
            dim_feedforward (int): _description_
            max_source_seq_len (int): _description_
            max_target_seq_len (int): _description_
            output_arch (Seq2SlateOutputArch): _description_
            temperature (float, optional): _description_. Defaults to 1.0.
            latent_state_embed_dim (Optional[int], optional): _description_. Defaults to None.
        """
        super().__init__()
        self.latent_state_dim: Final[int] = latent_state_dim
        self.candidate_dim: Final[int] = candidate_dim
        self.nlayers: Final[int] = nlayers
        self.nheads: Final[int] = nheads
        self.dim_model: Final[int] = dim_model
        self.dim_feedforward: Final[int] = dim_feedforward
        self.max_source_seq_len: Final[int] = max_source_seq_len
        self.max_target_seq_len: Final[int] = max_target_seq_len
        self.output_arch: Final[Seq2SlateOutputArch] = output_arch

        if latent_state_embed_dim is None:
            latent_state_embed_dim = dim_model // 2
        candidate_embed_dim = dim_model - latent_state_embed_dim
        self.latent_state_embedding: Final[Embedding] = Embedding(
            self.latent_state_dim, latent_state_embed_dim
        )
        self.candidate_embedding: Final[Embedding] = Embedding(
            self.candidate_dim, candidate_embed_dim
        )

        self._padding_symbol: Final[int] = const.PADDING_SYMBOL
        self._decoder_start_symbol: Final[int] = const.DECODER_START_SYMBOL

        self._rank_mode_val: Final[str] = Seq2SlateMode.RANK_MODE.value
        self._per_item_log_prob_dist_mode_val: Final[str] = Seq2SlateMode.PER_ITEM_LOG_PROB_DIST_MODE.value
        self._per_seq_log_prob_mode_val: Final[str] = Seq2SlateMode.PER_SEQ_LOG_PROB_MODE.value
        self._encoder_score_mode_val: Final[str] = Seq2SlateMode.ENCODER_SCORE_MODE.value
        self._decode_one_step_mode_val: Final[str] = Seq2SlateMode.DECODE_ONE_STEP_MODE.value

        self._output_placeholder = torch.zeros(1)

        self.encoder = PTEncoder(
            self.dim_model, self.dim_feedforward, self.nheads, self.nlayers
        )
        self.encoder_scorer = nn.Linear(self.dim_model, 1)
        self.decoder = PTDecoder(
            self.dim_model, self.dim_feedforward, self.nheads, self.nlayers
        )
        self.gc = CandidateGenerator()
        self.vl_positional_encoding = VariableLengthPositionalEncoding(
            self.dim_model
        )
        self._initialize_learnable_params()

    def _initialize_learnable_params(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)


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

    def forward(
        self,
        input: dt.PreprocessedRankingInput,
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
            return dt.RankingOutput(
                ordered_target_out_idcs=result.ordered_target_out_idcs,
                ordered_per_item_probs=result.ordered_per_item_probs,
                ordered_per_seq_probs=result.ordered_per_seq_probs,
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
                raise ValueError  # TODO: specify exception explaination
            result: Seq2SlateTransformerOutput = self.seq2slate(
                mode=mode.value,
                latent_state=input.latent_state.repr,
                source_seq=input.source_seq.repr,
                target_input_seq=input.target_input_seq.repr,
                target_input_indcs=input.target_input_indcs,
                target_output_indcs=input.target_output_indcs
            )
            # Obtain log probabilities of target sequences
            # item-log: autoregressive (batch_size, 1)
            # seq-log: entire target sequence (batch_size, target_sequence_len)
            if result.per_item_log_probs is not None:
                log_probs = result.per_item_log_probs
            else:
                log_probs = result.per_seq_log_probs
            return dt.RankingOutput(log_probs=log_probs)
        elif mode == Seq2SlateMode.ENCODER_SCORE_MODE:
            if input.target_output_indcs is None:
                raise ValueError  # TODO: specify exception explaination
            result: Seq2SlateTransformerOutput = self.seq2slate(
                mode=mode.value,
                latent_state=input.latent_state.repr,
                source_seq=input.source_seq.repr,
                target_output_indcs=input.target_output_indcs
            )
            return dt.RankingOutput(encoder_scores=result.encoder_scores)
        else:
            raise NotImplementedError(
                "Poshli nahyi i tak dohuya variantov nakrutil =)"
            )

    def get_distributed_data_parallel_model(self):
        raise NotImplementedError()  # TODO: Implement


@dataclass
class Seq2SlateTransformerNetwork(Seq2SlateNetwork):
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
