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
from typing import Any, List, Optional, Type, Union


from nastenka_solnishko.core.dataclasses import dataclass
from torch import nn


BLOCKLIST_TYPES = [nn.Module]


def _get_param_annotation(param: Any) -> Type:
    """Retrieve parameter type annotation.

    Args:
        param (Any): A parameter of some type.

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
    func,
    allowlist: Optional[List[str]] = None,
    blocklist: Optional[List[str]] = None,
    blocklist_types: List[Type] = BLOCKLIST_TYPES,
):
    """
    Create a decorator to create dataclass with the arguments of `func` as fields.
    Only annotated arguments are converted to fields. If the default value is mutable,
    you must use `dataclass.field(default_factory=default_factory)` as default.
    In that case, the func has to be wrapped with @resolve_defaults below.

    `allowlist` & `blocklist` are mutually exclusive.
    """

    parameters = signature(func).parameters

    if allowlist and blocklist:
        raise ValueError("Allowlist & blocklist are mutually exclusive")

    blocklist_set = set(blocklist or [])

    def _is_type_in_blocklist(_type: Type):
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

    def _is_valid_param(p):
        if p.name in blocklist_set:
            return False
        if p.annotation == Parameter.empty and p.default == Parameter.empty:
            return False
        ptype = _get_param_annotation(p)
        if _is_type_in_blocklist(ptype):
            return False
        return True

    allowlist = allowlist or [
        p.name for p in parameters.values() if _is_valid_param(p)]

    def wrapper(config_cls):
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


def resolve_defaults(func):
    """
    Use this decorator to resolve default field values in the constructor.
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


def param_hash(p) -> int:
    """
    Use this to make parameters hashable. This is required because __hash__()
    is not inherited when subclass redefines __eq__(). We only need this when
    the parameter dataclass has a list or dict field.
    """
    return hash(tuple(_hash_field(getattr(p, f.name)) for f in fields(p)))


def _hash_field(val):
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
