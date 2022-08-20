r"""
The module implements discriminated union register.
"""
import abc
import os
from dataclasses import fields
from typing import Any, Dict, Optional, Tuple, Type

from athena.core.dataclasses import dataclass
from athena.core.logger import LoggerMixin


class DiscriminatedUnion:
    r"""
    `Discriminated union <https://en.wikipedia.org/wiki/Tagged_union>`_ allows us get 
    a value that could take one of several different but fixed types. Assuming that 
    subclasses are pydantic's dataclass. All the fields must be ``Optional``
    with ``None`` as default value. This doesn't support changing selected field/value.
    """

    @property
    def value(self) -> Any:
        r"""
        Returns the value of the existing instance of roster.

        Raises:
            ValueError: If several values are defined.

        Returns:
            object: Selected instance.
        """
        selected_fields = [
            field.name for field in fields(self) if getattr(self, field.name, None)
        ]
        # Check if Union is discriminated
        if len(selected_fields) != 1:
            raise ValueError(
                f"{self} Expecting one selected field, got {selected_fields}"
            )
        return getattr(self, selected_fields[0])


class RegistryMeta(abc.ABCMeta, LoggerMixin):
    r"""
    Metaclass dedicated to autofill the discriminated unions.
    It keeps track all subclasses and register them at one 
    union factory named roster (via :func:`register()` method). Once 
    a roster is registered with all subclasses it gets frozen s.t.
    new member cannot be added. If ``SKIP_FROZEN_REGISTRY_CHECK=1``
    warning appears instead of exception when new member is attempted 
    to be added to the roster.

    Examples::

        class Tracker(metaclass=RegistryMeta):
           def some_common_method(self):
               pass 

        class TrackedClass1(Tracker):
            a: str
            b: int  

        class TrackerClass2(Tracker):
            a: str
            b: int
            c: float  

        @Tracker.register()
        class ClassesRoster(DiscriminatedUnion):
            pass

        # If no instance is passed then the roster
        # will be empty and no value will be returned
        ClassesRoster()

        # If an instance is passed the roster will retain
        # it and the value is exactly the given instance
        ClassesRoster(TrackedClass1=TrackedClass1("Hello", 5)),
    """
    def __init__(cls, name: str, bases: Tuple[type, ...], attrs: Dict[str, Any]):
        if not hasattr(cls, "REGISTRY"):
            # Put REGISTRY on cls. This only happens once on the base class
            cls.info(f"Adding REGISTRY to type {name}")
            cls.REGISTRY: Dict[str, Type] = {}
            cls.REGISTRY_NAME = name
            cls.REGISTRY_FROZEN = False

        if cls.REGISTRY_FROZEN:
            # trying to add to a frozen registry
            if bool(int(os.environ.get("SKIP_FROZEN_REGISTRY_CHECK", 0))):
                cls.warning(
                    f"{cls.REGISTRY_NAME} has been used to fill a union and is now frozen. "
                    "Since environment variable SKIP_FROZEN_REGISTRY_CHECK was set, "
                    f"no exception was raised, but {name} wasn't added to the registry"
                )
            else:
                raise RuntimeError(
                    f"{cls.REGISTRY_NAME} has been used to register a factory and is now frozen, "
                    f"so {name} can't be added to the roster. "
                    "Please rearrange your import orders. Or set environment variable "
                    "SKIP_FROZEN_REGISTRY_CHECK=1 to replace this error with a warning if you "
                    f"don't need the {name} to be added to the registry.")
        else:
            if not cls.__abstractmethods__ and name != cls.REGISTRY_NAME:
                # Only register fully-defined classes
                cls.info(f"Registering {name} to {cls.REGISTRY_NAME}")
                if hasattr(cls, "__registry_name__"):
                    registry_name = cls.__registry_name__
                    cls.info(f"Using {registry_name} instead of {name}")
                    name = registry_name
                assert name not in cls.REGISTRY, f"{name} in REGISTRY {cls.REGISTRY}"
                cls.REGISTRY[name] = cls
            else:
                cls.info(
                    f"Not Registering {name} to {cls.REGISTRY_NAME}. Abstract "
                    f"methods {list(cls.__abstractmethods__)} are not implemented."
                )
        return super().__init__(name, bases, attrs)

    def register(cls):
        """
        Register all ``cls`` subclasses to the given roster.

        Args:
            cls (RegistryMeta): The tracker class. Generally, it 
                depicts the purpose of its subclasses.

        Returns:
            Callable[[DiscriminatedUnion], DiscriminatedUnion]: 
                Callable which modifies given roster by puting 
                all subclasses into it.
        """
        def wrapper(roster):
            cls.REGISTRY_FROZEN = True

            def make_roster_instance(
                inst: object, instance_class: Optional[type] = None
            ) -> object:
                inst_class = instance_class or type(inst)
                key = getattr(
                    inst_class, "__registry_name__",
                    inst_class.__name__
                )
                return roster(**{key: inst})

            roster.make_roster_instance = make_roster_instance

            roster.__annotations__ = {
                name: Optional[t]
                for name, t in cls.REGISTRY.items()
            }
            for name in cls.REGISTRY:
                setattr(roster, name, None)
            return dataclass(frozen=True)(roster)

        return wrapper
