import abc
import sys
from types import TracebackType
from typing import Optional, Type


class BaseDebugClass(abc.ABC):
    @abc.abstractstaticmethod
    def start() -> None:
        pass


class DebugOnException(BaseDebugClass):
    """Debug arises drops on exception.
    """
    @staticmethod
    def start() -> None:
        def info(
            type: Type[BaseException],
            value: BaseException,
            traceback: Optional[TracebackType]
        ) -> None:
            if hasattr(sys, "ps1") or not sys.stderr.isatty():
                sys.__excepthook__(type, value, traceback)
            else:
                import pdb
                import traceback

                traceback.print_exception(type, value, traceback)
                print
                pdb.post_mortem(traceback)

        sys.excepthook = info
