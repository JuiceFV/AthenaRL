from typing import Any, Dict, List
from dataclasses import asdict, field
from source.core.dataclasses import dataclass


@dataclass
class SeqEntity:
    seq_id: int
    action: int
    state: Any
    score: float

    def asdict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class Sequence:
    entities: List[SeqEntity] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entities)

    def __getattr__(self, attr: str) -> List[Any]:
        return list(map(lambda entity: getattr(entity, attr), self.entities))

    def add_enitity(self, entity: SeqEntity) -> None:
        self.entities.append(entity)
