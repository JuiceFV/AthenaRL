from typing import Optional

from athena.core.dataclasses import dataclass
from athena.core.registry import DiscriminatedUnion
from athena.net_builder.ranking.seq2slate import Seq2SlateRanking as Seq2SlateRankingType


@dataclass
class SlateRankingNetBuilderRoster(DiscriminatedUnion):
    Seq2SlateRanking: Optional[Seq2SlateRankingType] = None
