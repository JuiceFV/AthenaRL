import abc
from athena.core.registry import RegistryMeta
from athena.core.logger import LoggerMixin
from athena.core.singleton import Singleton


class FAPper(LoggerMixin, metaclass=type("RegSingleton", (RegistryMeta, Singleton), {})):
    @abc.abstractmethod
    def fap(self, *args, **kwargs) -> str:
        pass
