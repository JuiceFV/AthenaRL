from source.core.dataclasses import dataclass, field
from source.core.params import TransformerParams, param_hash
from source.models.base import BaseModel
from source.models.seq2slate import Seq2SlateTransformerNetwork
from source.models.utils.seq2slate.dtypes import Seq2SlateOutputArch
from source.net_builder.slate_reranking_builder import SlateRerankingNetBuilder


@dataclass
class Seq2SlateReranking(SlateRerankingNetBuilder):
    __hash__ = param_hash
    
    output_arch: Seq2SlateOutputArch = Seq2SlateOutputArch.AUTOREGRESSIVE
    temperature: float = 1.0
    transformer: TransformerParams = field(
        default_factory=lambda: TransformerParams(
            nheads=8, dim_model=512, dim_feedforward=2048, nlayers=8
        )
    )

    def build_slate_reranking_network(
        self, latent_state_dim: int, candidate_dim: int, num_of_candidates: int, slate_size: int
    ) -> BaseModel:
        return Seq2SlateTransformerNetwork(
            latent_state_dim=latent_state_dim,
            candidate_dim=candidate_dim,
            nlayers=self.transformer.nlayers,
            dim_model=self.transformer.dim_model,
            max_source_seq_len=num_of_candidates,
            max_target_seq_len=slate_size,
            output_arch=self.output_arch,
            temperature=self.temperature,
            nheads=self.transformer.nheads,
            dim_feedforward=self.transformer.dim_feedforward,
            latent_state_embed_dim=self.transformer.latent_state_embed_dim
        )
