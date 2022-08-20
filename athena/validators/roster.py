from athena.core.registry import DiscriminatedUnion
from athena.validators.validator_base import ModelValidator
from athena.validators.noop_validator import NoValidation # noqa

@ModelValidator.register()
class ModelValidatorRoster(DiscriminatedUnion):
    pass