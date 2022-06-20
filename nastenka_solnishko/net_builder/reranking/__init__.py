from typing import Optional

from nastenka_solnishko.core.dataclasses import dataclass
from nastenka_solnishko.core.discriminated_union import DiscriminatedUnion
from nastenka_solnishko.net_builder.reranking.seq2slate import (
    Seq2SlateReranking as Seq2SlateRerankingType
)


@dataclass
class SlateRerankingNetBuilderRoster(DiscriminatedUnion):
    Seq2SlateReranking: Optional[Seq2SlateRerankingType] = None
