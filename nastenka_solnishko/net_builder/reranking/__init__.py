from typing import Optional

from source.core.dataclasses import dataclass
from source.core.discriminated_union import DiscriminatedUnion
from source.net_builder.reranking.seq2slate import (
    Seq2SlateReranking as Seq2SlateRerankingType
)


@dataclass
class SlateRerankingNetBuilderRoster(DiscriminatedUnion):
    Seq2SlateReranking: Optional[Seq2SlateRerankingType] = None
