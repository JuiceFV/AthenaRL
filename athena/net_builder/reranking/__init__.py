from typing import Optional

from athena.core.dataclasses import dataclass
from athena.core.registry import DiscriminatedUnion
from athena.net_builder.reranking.seq2slate import (
    Seq2SlateReranking as Seq2SlateRerankingType
)


@dataclass
class SlateRerankingNetBuilderRoster(DiscriminatedUnion):
    Seq2SlateReranking: Optional[Seq2SlateRerankingType] = None
