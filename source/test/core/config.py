import unittest

from typing import Any
from source.core.dataclasses import field
from source.core.confg import make_config_class, resolve_defaults

class ABConfig:
    @resolve_defaults
    def __init__(
        self, a: int = 1, b : int = field(default_factory=lambda: 2)
    ) -> None:
        self.a = a
        self.b = b

    def __call__(self) -> int:
        return self.a * self.b

@make_config_class(ABConfig.__init__)
class ABConfigClass:
    pass

class TestConfigParser(unittest.TestCase):
    pass