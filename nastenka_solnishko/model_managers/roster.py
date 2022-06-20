from typing import Optional
from nastenka_solnishko.core.dataclasses import dataclass
from nastenka_solnishko.core.discriminated_union import DiscriminatedUnion
from nastenka_solnishko.model_managers.ranking.seq2slate import Seq2Slate as Seq2SlateType


@dataclass(frozen=True)
class ModelManagerRoster(DiscriminatedUnion):
    Seq2Slate: Optional[Seq2SlateType] = None
