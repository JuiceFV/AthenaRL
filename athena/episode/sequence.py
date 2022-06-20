import numpy as np
from typing import Any, Dict, List, Generator
from dataclasses import asdict, field
from athena.core.dataclasses import dataclass


@dataclass
class SequenceEntity:
    seq_id: int
    seq_num: int
    action: int
    state: Any
    score: float
    is_last: bool

    def asdict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class Sequence:
    entities: List[SequenceEntity] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entities)

    def __getattr__(self, attr: str) -> List[Any]:
        return list(map(lambda entity: getattr(entity, attr), self.entities))

    def __iter__(self) -> Generator[SequenceEntity, None, None]:
        for entity in self.entities:
            yield entity

    def _calculate_cumulative_shift(self, gamma: float = 1.0) -> float:
        num_entities = len(self)
        if num_entities <= 0:
            raise ValueError("Empty sequence")
        discounts = [gamma ** pos for pos in range(num_entities)]
        return sum([discount * ent.score + ent.score for ent, discount in zip(self, discounts)])

    def recalculate_positional_scores(self, gamma: float = 1.0) -> None:
        pos_shift = np.abs(self._calculate_cumulative_shift(gamma))
        for entity in self:
            entity.score += pos_shift

    def add_enitity(self, entity: SequenceEntity) -> None:
        self.entities.append(entity)
