from typing import Optional
from athena.core.dataclasses import dataclass
from athena.core.registry import DiscriminatedUnion
from athena.model_managers.ranking.seq2slate import Seq2Slate as Seq2SlateType


@dataclass(frozen=True)
class ModelManagerRoster(DiscriminatedUnion):
    Seq2Slate: Optional[Seq2SlateType] = None
