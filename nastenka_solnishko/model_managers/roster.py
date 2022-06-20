from typing import Optional
from source.core.dataclasses import dataclass
from source.core.discriminated_union import DiscriminatedUnion
from source.model_managers.ranking.seq2slate import Seq2Slate as Seq2SlateType


@dataclass(frozen=True)
class ModelManagerRoster(DiscriminatedUnion):
    Seq2Slate: Optional[Seq2SlateType] = None
