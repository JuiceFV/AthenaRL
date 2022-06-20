import numpy as np
from typing import Any, Union, Dict

from athena.core.dataclasses import dataclass, field


@dataclass
class RandGen:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init_post_parse__(self):
        if not hasattr(np.random, self.name):
            raise ValueError(f"No such random generator: {self.name}")
        generator = getattr(np.random, self.name)
        try:
            generator(**self.params)
        except Exception as e:
            raise ValueError(f"Unable retrieve {self.name} for: {self.params}")
        self.generator = generator

    def draw(self) -> Union[float, int]:
        return self.generator(**self.params)


@dataclass(frozen=True)
class SimFeature:
    id: int
    generator: RandGen

    def draw(self) -> Union[float, int]:
        return self.generator.draw()
