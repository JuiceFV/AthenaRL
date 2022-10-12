"""Module especially dedicated for configuration handling.
When a user executes a runner YAML-config file is passed.
In purpose to properly handle the configuration w/o
flexibility damage we parse runner prototype and compare
its arguments to the configuration content. So that
a user can define its own runner function and create
config file compatible to the runner arguments (types including).
"""
import functools
from dataclasses import MISSING, Field, fields
from inspect import Parameter, isclass, signature
from typing import Any, Callable, List, Optional, Union

from torch import nn

from athena.core.dataclasses import dataclass

BLOCKLIST_TYPES = [nn.Module]


def _get_param_annotation(param: Parameter) -> type:
    """Retrieve parameter type annotation.

    Args:
        param (Parameter): A parameter of some type.

    Raises:
        ValueError: If not annotated and unable to infer type from default
        ValueError: If not annotated and has no default to infer type
        ValueError: If tuple then value could be any type
        ValueError: If dict then value could be any type

    Returns:
        Type: Type of a parameter
    """
    if param.annotation == Parameter.empty and param.default == Parameter.empty:
        raise ValueError(
            f"Param {param}: both annotation and default are empty, "
            "so cannot infer any useful annotation."
        )
    if param.annotation != Parameter.empty:
        return param.annotation
    if param.default is None:
        raise ValueError(
            f"Param {param}: default is None and annotation is empty, "
            "cannot infer useful annotation"
        )
    if isinstance(param.default, tuple):
        raise ValueError(f"Param {param}: default is tuple, cannot infer type")
    if isinstance(param.default, dict):
        raise ValueError(f"Param{param}: default is tuple, cannot infer type")
    return type(param.default)


def create_config_class(
    func: Callable,
    allowlist: Optional[List[str]] = None,
    blocklist: Optional[List[str]] = None,
    blocklist_types: List[type] = BLOCKLIST_TYPES,
) -> Callable[[type], object]:
    r"""
    Create a decorator to create dataclass with the arguments of ``func`` as fields.
    Only annotated arguments are converted to fields. If the default value is mutable,
    you must use ``dataclass.field(default_factory=default_factory)`` as default.
    In that case, the func has to be wrapped with :func:`resolve_defaults` below.

    .. note::

        ``allowlist`` & ``blocklist`` are mutually exclusive.

    Example::

        def func(a: int, b: str = "Hellow World"):
            pass

        @create_config_class(func)
        class FuncClass:
            pass

        FuncClass(1) # FuncClass(a=1, b='Hellow World')

    Args:
        func (Callable): Function from one's parameters dataclass is created.
        allowlist (Optional[List[str]], optional): Acceptable fields. Defaults to ``None``.
        blocklist (Optional[List[str]], optional): Prohibited fields. Defaults to ``None``.
        blocklist_types (List[Type], optional): Prohibited types. Defaults to ``BLOCKLIST_TYPES``.

    Raises:
        ValueError: If allowlist & blocklist are mutually exclusive.
        TypeError: If a param is ``Union`` it has to be ``Optional``.
        TypeError: If param isn't python class.

    Returns:
        Callable[[type], object]: Callable which makes class with fields from ``func`` params.
    """
    parameters = signature(func).parameters

    if allowlist and blocklist:
        raise ValueError("Allowlist & blocklist are mutually exclusive")

    blocklist_set = set(blocklist or [])

    def _is_type_in_blocklist(_type: type) -> bool:
        if getattr(_type, "__origin__", None) is Union:
            if len(_type.__args__) != 2 or _type.__args__[1] != type(None):
                raise TypeError(
                    "Only Unions of [X, None] (a.k.a. Optional[X]) are supported"
                )
            _type = _type.__args__[0]
        if hasattr(_type, "__origin__"):
            _type = _type.__origin__
        if not isclass(_type):
            raise TypeError(f"{_type} is not a class.")
        return any(issubclass(_type, blocklist_type) for blocklist_type in blocklist_types)

    def _is_valid_param(p: Parameter) -> bool:
        if p.name in blocklist_set:
            return False
        if p.annotation == Parameter.empty and p.default == Parameter.empty:
            return False
        ptype = _get_param_annotation(p)
        if _is_type_in_blocklist(ptype):
            return False
        return True

    allowlist = allowlist or [
        p.name for p in parameters.values() if _is_valid_param(p)
    ]

    def wrapper(config_cls: type) -> object:
        # Add __annotations__ for dataclass
        config_cls.__annotations__ = {
            field_name: _get_param_annotation(parameters[field_name])
            for field_name in allowlist
        }
        # Set default values
        for field_name in allowlist:
            default = parameters[field_name].default
            if default != Parameter.empty:
                setattr(config_cls, field_name, default)

        # Add hashing to support hashing list and dict
        config_cls.__hash__ = param_hash

        # Add non-recursive asdict(). dataclasses.asdict() is recursive
        def asdict(self):
            return {field.name: getattr(self, field.name) for field in fields(self)}

        config_cls.asdict = asdict

        return dataclass(frozen=True)(config_cls)

    return wrapper


def _resolve_default(val):
    if not isinstance(val, Field):
        return val
    if val.default != MISSING:
        return val.default
    if val.default_factory != MISSING:
        return val.default_factory()
    raise ValueError("No default value")


def resolve_defaults(func: Callable) -> Callable:
    r"""
     Use this decorator to resolve default field values in the constructor.

    Args:
        func (Callable): Function with mutable parameter one has to be
            resolved.

    Raises:
        ValueError: If number of given param values not fits to number of params.

    Returns:
        Callable: Callable one resolve the mutable defaults.
    """

    func_params = list(signature(func).parameters.values())

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > len(func_params):
            raise ValueError(
                f"There are {len(func_params)} parameters in total, "
                f"but args is {len(args)} long. \n"
                f"{args}"
            )
        # go through unprovided default kwargs
        for p in func_params[len(args):]:
            # only resolve defaults for Fields
            if isinstance(p.default, Field):
                if p.name not in kwargs:
                    kwargs[p.name] = _resolve_default(p.default)
        return func(*args, **kwargs)

    return wrapper


def param_hash(p: Parameter) -> int:
    r"""
    Use this to make parameters hashable. This is required because :func:`__hash__`
    is not inherited when subclass redefines :func:`__eq__`. We only need this when
    the parameter dataclass has a list or dict field.

    Args:
        p (Parameter): Parameter to hash.

    Returns:
        int: Hash of given parameter ``p``.
    """
    return hash(tuple(_hash_field(getattr(p, f.name)) for f in fields(p)))


def _hash_field(val: Any) -> Any:
    """
    Returns hashable value of the argument. A list is converted to a tuple.
    A dict is converted to a tuple of sorted pairs of key and value.
    """
    if isinstance(val, list):
        return tuple(val)
    elif isinstance(val, dict):
        return tuple(sorted(val.items()))
    else:
        return val
