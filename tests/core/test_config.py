import abc
import os
import unittest

from athena.core.config import create_config_class, resolve_defaults
from athena.core.dataclasses import dataclass, field
from athena.core.registry import RegistryMeta, DiscriminatedUnion


class AB:
    @resolve_defaults
    def __init__(
        self, a: int = 1, b: int = field(default_factory=lambda: 2)
    ) -> None:
        self.a = a
        self.b = b

    def __call__(self) -> int:
        return self.a + self.b


@create_config_class(AB.__init__)
class ABConfigClass:
    pass


class FooRegistry(metaclass=RegistryMeta):
    @abc.abstractmethod
    def foo(self) -> int:
        pass


@dataclass
class Foo(FooRegistry):
    ab_cfg: ABConfigClass = field(default_factory=ABConfigClass)

    def foo(self):
        ab = AB(**self.ab_cfg.asdict())
        return ab()


@dataclass
class Bar(FooRegistry):
    def foo(self) -> int:
        return 10


@FooRegistry.register()
class FooRoster(DiscriminatedUnion):
    pass


@dataclass
class Config:
    roster: FooRoster = field(
        default_factory=lambda: FooRoster(Foo=Foo())
    )


class TestConfigParser(unittest.TestCase):
    def test_parse_foo_default(self) -> None:
        raw_config = {}
        config = Config(**raw_config)
        self.assertEqual(config.roster.value.foo(), 3)

    def test_parse_foo(self) -> None:
        raw_config = {"roster": {"Foo": {"ab_cfg": {"a": 6}}}}
        config = Config(**raw_config)
        self.assertEqual(config.roster.value.foo(), 8)

    def test_parse_bar(self) -> None:
        raw_config = {"roster": {"Bar": {}}}
        config = Config(**raw_config)
        self.assertEqual(config.roster.value.foo(), 10)

    def test_frozen_registry(self) -> None:
        with self.assertRaises(RuntimeError):

            @dataclass
            class Baz(FooRegistry):
                def foo(self):
                    return 20

        self.assertListEqual(sorted(FooRegistry.REGISTRY.keys()), ["Bar", "Foo"])

    def test_frozen_registry_skip(self) -> None:
        _environ = dict(os.environ)
        os.environ.update({"SKIP_FROZEN_REGISTRY_CHECK": "1"})
        try:

            @dataclass
            class Baz(FooRegistry):
                def foo(self):
                    return 20

        finally:
            os.environ.clear()
            os.environ.update(_environ)

        self.assertListEqual(sorted(FooRegistry.REGISTRY.keys()), ["Bar", "Foo"])
