from nastenka_solnishko.core.dataclasses import dataclass
from nastenka_solnishko.model_managers.seq2slate_base import Seq2SlateBase


@dataclass
class Seq2Slate(Seq2SlateBase):

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
